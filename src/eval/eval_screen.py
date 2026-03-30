from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.eval.baselines import kmer_similarity_scores, random_scores
from src.train.trainer_utils import build_retriever
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


def load_checkpoint_model(checkpoint_path: str | Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_retriever(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def score_with_model(model: torch.nn.Module, query_sequence: str, doc_sequences: list[str]) -> np.ndarray:
    with torch.no_grad():
        scores = model.score_sequences([query_sequence], doc_sequences)
    return scores[0].detach().cpu().numpy()


def rank_ids_from_scores(doc_ids: list[str], scores: np.ndarray, query_id: str) -> list[str]:
    order = np.argsort(scores)[::-1]
    ranked = [doc_ids[idx] for idx in order if doc_ids[idx] != query_id]
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cCRE retrieval baselines and trained models")
    parser.add_argument("--config", default="configs/train/eval.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--baseline",
        choices=["random", "kmer", "model"],
        default="kmer",
    )
    parser.add_argument("--output", default="outputs/metrics/eval_screen.json")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["seed"]))
    device = "cuda" if cfg.get("device", "cpu") == "cuda" and torch.cuda.is_available() else "cpu"

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
    activity_lookup = manifest.set_index("ccre_id")["activity_vector"].apply(parse_json_list).to_dict()
    relevance_lookup = (
        pairs[(pairs["split"] == "test") & (pairs["label"] == 1)]
        .groupby("query_id")["doc_id"]
        .apply(set)
        .to_dict()
    )

    model = load_checkpoint_model(args.checkpoint, device) if args.checkpoint else None
    results = {f"Recall@{k}": [] for k in top_k}
    results.update({"MRR": [], "nDCG": []})
    for k in top_k:
        results[f"class_purity@{k}"] = []
        results[f"activity_jaccard@{k}"] = []
        results[f"activity_corr@{k}"] = []

    for row in query_df.itertuples(index=False):
        if row.ccre_id not in relevance_lookup:
            continue
        if args.baseline == "random":
            scores = random_scores(len(doc_ids), seed=int(cfg["evaluation"]["random_seed"]))
        elif args.baseline == "kmer":
            scores = kmer_similarity_scores(row.__getattribute__(sequence_column), doc_sequences, int(cfg["evaluation"]["kmer_size"]))
        else:
            if model is None:
                raise ValueError("--checkpoint is required for --baseline model")
            scores = score_with_model(model, row.__getattribute__(sequence_column), doc_sequences)

        ranked_ids = rank_ids_from_scores(doc_ids, scores, row.ccre_id)
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
    summary["baseline"] = args.baseline
    summary["num_queries"] = int(len(query_df))
    if args.checkpoint:
        summary["checkpoint"] = str(args.checkpoint)

    dump_json(summary, args.output)
    LOGGER.info("Saved evaluation summary to %s", args.output)


if __name__ == "__main__":
    main()
