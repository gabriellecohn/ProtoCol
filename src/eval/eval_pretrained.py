"""Evaluate pretrained DNABERT-2 (no fine-tuning) as a retrieval baseline.

Encodes sequences with the raw pretrained model, mean-pools token embeddings,
and scores query-document pairs by cosine similarity.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from src.utils.io import dump_json, load_yaml, parse_json_list
from src.utils.logging import get_logger
from src.utils.metrics import (
    binary_jaccard,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    safe_pearson,
    top_k_class_purity,
)
from src.utils.seed import set_global_seed

LOGGER = get_logger(__name__)
MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def load_pretrained(device: str):
    LOGGER.info("Loading pretrained %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    hf_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model_class = get_class_from_dynamic_module(
        "bert_layers.BertModel", MODEL_NAME, trust_remote_code=True,
    )
    model_class.config_class = type(hf_config)

    # Disable flash attention (Triton incompatibility)
    import sys
    for mod_name in list(sys.modules):
        mod = sys.modules[mod_name]
        if hasattr(mod, "flash_attn_qkvpacked_func"):
            mod.flash_attn_qkvpacked_func = None

    model = model_class.from_pretrained(MODEL_NAME, config=hf_config)
    model.to(device)
    model.eval()
    return tokenizer, model


def encode_sequences(
    tokenizer, model, sequences: list[str], device: str, batch_size: int = 32, max_length: int = 512,
) -> torch.Tensor:
    """Encode sequences into mean-pooled embeddings (N, hidden_dim)."""
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            mask = tokens["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = F.normalize(pooled, dim=-1)
            all_embeddings.append(pooled.cpu())
            if (i // batch_size) % 10 == 0:
                LOGGER.info("  encoded %d / %d sequences", min(i + batch_size, len(sequences)), len(sequences))
    return torch.cat(all_embeddings, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrained DNABERT-2 retrieval (no fine-tuning)")
    parser.add_argument("--config", default="configs/train/eval_dnabert2_small.yaml")
    parser.add_argument("--output", default="outputs/metrics/eval_pretrained.json")
    parser.add_argument("--tf-activity", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["seed"]))
    device = args.device or cfg.get("device", "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    manifest = pd.read_csv(cfg["data"]["manifest_path"])
    pairs = pd.read_csv(cfg["data"]["pair_path"])
    sequence_column = cfg["data"]["sequence_column"]
    top_k = [int(k) for k in cfg["evaluation"]["top_k"]]

    query_df = manifest[manifest["split"] == "test"].dropna(subset=[sequence_column]).copy()
    corpus_df = manifest[manifest["split"] == "test"].dropna(subset=[sequence_column]).copy()
    sample_queries = min(int(cfg["evaluation"]["sample_queries"]), len(query_df))
    sample_corpus = min(int(cfg["evaluation"]["sample_corpus"]), len(corpus_df))
    query_df = query_df.sample(n=sample_queries, random_state=int(cfg["seed"])) if sample_queries else query_df
    corpus_df = corpus_df.sample(n=sample_corpus, random_state=int(cfg["seed"])) if sample_corpus else corpus_df

    doc_ids = corpus_df["ccre_id"].tolist()
    doc_sequences = corpus_df[sequence_column].tolist()
    label_lookup = manifest.set_index("ccre_id")["ccre_class"].astype(str).to_dict()

    if args.tf_activity:
        LOGGER.info("Loading TF activity vectors from %s", args.tf_activity)
        tf_data = np.load(args.tf_activity, allow_pickle=True)
        tf_ccre_ids = tf_data["ccre_ids"]
        tf_matrix = tf_data["tf_matrix"]
        tf_lookup_map = {cid: i for i, cid in enumerate(tf_ccre_ids)}
        n_tfs = tf_matrix.shape[1]
        activity_lookup = {
            cid: tf_matrix[tf_lookup_map[cid]].astype(float).tolist() if cid in tf_lookup_map else [0.0] * n_tfs
            for cid in manifest["ccre_id"]
        }
    else:
        activity_lookup = manifest.set_index("ccre_id")["activity_vector"].apply(parse_json_list).to_dict()

    relevance_lookup = (
        pairs[(pairs["split"] == "test") & (pairs["label"] == 1)]
        .groupby("query_id")["doc_id"]
        .apply(set)
        .to_dict()
    )

    # Load model and encode
    tokenizer, model = load_pretrained(device)

    LOGGER.info("Encoding %d corpus documents...", len(doc_sequences))
    doc_embeddings = encode_sequences(tokenizer, model, doc_sequences, device)

    query_sequences = query_df[sequence_column].tolist()
    LOGGER.info("Encoding %d query sequences...", len(query_sequences))
    query_embeddings = encode_sequences(tokenizer, model, query_sequences, device)

    # Score all queries against all docs via cosine similarity
    LOGGER.info("Computing cosine similarity scores...")
    scores_matrix = (query_embeddings @ doc_embeddings.T).numpy()

    # Evaluate
    results = {f"Recall@{k}": [] for k in top_k}
    results.update({"MRR": [], "nDCG": []})
    for k in top_k:
        results[f"class_purity@{k}"] = []
        results[f"activity_jaccard@{k}"] = []
        results[f"activity_corr@{k}"] = []

    for q_idx, row in enumerate(query_df.itertuples(index=False)):
        if row.ccre_id not in relevance_lookup:
            continue
        scores = scores_matrix[q_idx]
        order = np.argsort(scores)[::-1]
        ranked_ids = [doc_ids[idx] for idx in order if doc_ids[idx] != row.ccre_id]
        relevant_ids = relevance_lookup[row.ccre_id]

        results["MRR"].append(reciprocal_rank(ranked_ids, relevant_ids))
        results["nDCG"].append(ndcg_at_k(ranked_ids, relevant_ids, max(top_k)))
        query_activity = activity_lookup.get(row.ccre_id, [])
        query_label = label_lookup.get(row.ccre_id, "unknown")

        for k in top_k:
            top_docs = ranked_ids[:k]
            results[f"Recall@{k}"].append(recall_at_k(ranked_ids, relevant_ids, k))
            results[f"class_purity@{k}"].append(top_k_class_purity(top_docs, label_lookup, query_label, k))
            activity_jaccards = [binary_jaccard(query_activity, activity_lookup.get(doc_id, [])) for doc_id in top_docs]
            activity_corrs = [safe_pearson(query_activity, activity_lookup.get(doc_id, [])) for doc_id in top_docs]
            results[f"activity_jaccard@{k}"].append(float(np.mean(activity_jaccards)) if activity_jaccards else 0.0)
            results[f"activity_corr@{k}"].append(float(np.mean(activity_corrs)) if activity_corrs else 0.0)

    summary = {metric: float(np.mean(values)) if values else 0.0 for metric, values in results.items()}
    summary["baseline"] = "pretrained_dnabert2_meanpool"
    summary["num_queries"] = int(len(query_df))

    dump_json(summary, args.output)
    LOGGER.info("Results:")
    for metric in ["MRR", "nDCG"] + [f"Recall@{k}" for k in top_k] + [f"class_purity@{k}" for k in top_k]:
        LOGGER.info("  %s: %.4f", metric, summary[metric])
    LOGGER.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
