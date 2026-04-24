"""Full-corpus retrieval evaluation for Pfam ColBERT/MeanPool models and baselines.

Reports for every query, against the full test-split corpus:

  Standard retrieval (relevance = same family):
    Recall@k, MRR, nDCG

  Distant homolog detection (relevance = same clan, different family):
    Recall@k, MRR, nDCG  -- tests if the model preserves clan-level signal
    despite contrastive training pushing these apart.

  Top-k composition:
    % of top-k that are same-family / same-clan-diff-family / diff-clan.

  Hard MRR:
    Rank of the same-family positive when restricted to same-clan distractors
    only (easy negatives removed). Isolates the discriminative task.

Three modes:

  --checkpoint <path>           Load a trained checkpoint (ColBERT or MeanPool).
  --baseline random             Score uniformly at random.
  --baseline kmer-K             K-mer TF-IDF (cosine similarity), K in 3,4,5,...
  --baseline meanpool-esm2-35m  Frozen pretrained ESM-2, mean-pool, cosine.
  --baseline meanpool-esm2-650m Same with the 650M model.
  --baseline randproj-esm2-35m  Frozen ESM-2 + random 64-d projection + MaxSim.
  --baseline randproj-esm2-650m Same with the 650M model.

Usage:
    python -m src.eval.eval_pfam \\
        --config configs/train/eval_pfam.yaml \\
        --checkpoint outputs/pfam_colbert_small/<run>/checkpoints/pfam_colbert_small_best.pt
    python -m src.eval.eval_pfam --baseline kmer-4 --output outputs/baselines/kmer_4.json
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.colbert import ColBERTRetriever
from src.models.meanpool import MeanPoolRetriever
from src.train.trainer_utils import build_retriever
from src.utils.io import dump_json, load_yaml
from src.utils.logging import get_logger
from src.utils.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from src.utils.seed import set_global_seed


LOGGER = get_logger(__name__)

ESM_MODEL_NAMES = {
    "35m": "facebook/esm2_t12_35M_UR50D",
    "650m": "facebook/esm2_t33_650M_UR50D",
}


# ---------- Scorer abstraction ----------------------------------------------

class Scorer(Protocol):
    """Encodes a corpus once, then scores each query against it.

    Concrete scorers implement ``encode_corpus`` (called once) and
    ``score_query`` (called per query); ``score_query`` returns a 1-D array of
    length len(corpus_seqs).
    """
    name: str
    def encode_corpus(self, sequences: list[str]) -> None: ...
    def score_query(self, query_seq: str) -> np.ndarray: ...


# ---------- Random ----------------------------------------------------------

class RandomScorer:
    name = "random"

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._n = 0

    def encode_corpus(self, sequences: list[str]) -> None:
        self._n = len(sequences)

    def score_query(self, query_seq: str) -> np.ndarray:
        return self._rng.random(self._n).astype(np.float32)


# ---------- K-mer TF-IDF ----------------------------------------------------

class KmerScorer:
    """Sparse TF-IDF over k-mers (sklearn). Cosine similarity for scoring."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.name = f"kmer-{k}"
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Treat the protein sequence like a string and slide a char window of
        # width k. min_df=2 drops vocabulary that appears once (sparser, faster).
        self._vec = TfidfVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        self._corpus_mat = None

    def encode_corpus(self, sequences: list[str]) -> None:
        self._corpus_mat = self._vec.fit_transform(sequences)
        LOGGER.info("  k=%d  vocab=%d  nnz=%d", self.k, len(self._vec.vocabulary_), self._corpus_mat.nnz)

    def score_query(self, query_seq: str) -> np.ndarray:
        q = self._vec.transform([query_seq])  # (1, V) sparse, L2-normalized
        # cos sim = dot product since both are L2-normalized
        return (self._corpus_mat @ q.T).toarray().squeeze(1).astype(np.float32)


# ---------- Frozen ESM-2 (mean-pool, no learned head) -----------------------

class FrozenESMMeanPoolScorer:
    """Pretrained ESM-2, mean-pool token hidden states, L2-normalize. Cosine."""

    def __init__(self, esm_size: str, device: str, batch_size: int, max_length: int) -> None:
        self.name = f"meanpool-esm2-{esm_size}"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.autocast = device.startswith("cuda")
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAMES[esm_size])
        self._model = AutoModel.from_pretrained(ESM_MODEL_NAMES[esm_size]).to(device).eval()
        self._corpus_emb = None  # (N, F) on device

    def _encode(self, sequences: list[str]) -> torch.Tensor:
        out = []
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = sequences[i : i + self.batch_size]
                tok = self._tokenizer(batch, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.max_length).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    h = self._model(**tok).last_hidden_state            # (B, T, H)
                m = tok["attention_mask"].unsqueeze(-1).float()
                pooled = (h.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
                pooled = F.normalize(pooled, dim=-1)
                out.append(pooled.cpu())
                if (i // self.batch_size) % 50 == 0:
                    LOGGER.info("  encoded %d/%d", min(i + self.batch_size, len(sequences)), len(sequences))
        return torch.cat(out, dim=0)

    def encode_corpus(self, sequences: list[str]) -> None:
        self._corpus_emb = self._encode(sequences).to(self.device)

    def score_query(self, query_seq: str) -> np.ndarray:
        q = self._encode([query_seq])[0].to(self.device)                # (F,)
        return (self._corpus_emb @ q).cpu().numpy().astype(np.float32)


# ---------- Frozen ESM-2 + random projection + MaxSim -----------------------

class RandProjMaxSimScorer:
    """Pretrained ESM-2, fixed random linear projection to 64d, ColBERT MaxSim."""

    def __init__(self, esm_size: str, device: str, batch_size: int, max_length: int,
                 projection_dim: int, doc_chunk: int, seed: int) -> None:
        self.name = f"randproj-esm2-{esm_size}"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.doc_chunk = doc_chunk
        self.autocast = device.startswith("cuda")
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAMES[esm_size])
        self._model = AutoModel.from_pretrained(ESM_MODEL_NAMES[esm_size]).to(device).eval()
        # Fixed random projection (Linear with no bias matches the trained ColBERT head).
        gen = torch.Generator(device="cpu").manual_seed(seed)
        hidden = self._model.config.hidden_size
        weight = torch.empty(projection_dim, hidden)
        torch.nn.init.normal_(weight, mean=0.0, std=(1.0 / hidden) ** 0.5, generator=gen)
        self._proj = torch.nn.Linear(hidden, projection_dim, bias=False).to(device)
        with torch.no_grad():
            self._proj.weight.copy_(weight)
        self._proj.eval()
        self._corpus_tokens = None  # (N, T_max, F) on CPU
        self._corpus_masks = None   # (N, T_max) bool on CPU

    def _encode(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        all_t, all_m = [], []
        with torch.no_grad():
            for i in range(0, len(sequences), self.batch_size):
                batch = sequences[i : i + self.batch_size]
                tok = self._tokenizer(batch, return_tensors="pt", padding=True,
                                      truncation=True, max_length=self.max_length).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    h = self._model(**tok).last_hidden_state            # (B, T, H)
                    p = self._proj(h)                                   # (B, T, F)
                p = F.normalize(p.float(), dim=-1)
                all_t.append(p.cpu())
                all_m.append(tok["attention_mask"].bool().cpu())
                if (i // self.batch_size) % 50 == 0:
                    LOGGER.info("  encoded %d/%d", min(i + self.batch_size, len(sequences)), len(sequences))
        max_len = max(t.size(1) for t in all_t)
        padded_t, padded_m = [], []
        for t, m in zip(all_t, all_m):
            pad = max_len - t.size(1)
            if pad > 0:
                t = F.pad(t, (0, 0, 0, pad))
                m = F.pad(m, (0, pad))
            padded_t.append(t); padded_m.append(m)
        return torch.cat(padded_t, dim=0), torch.cat(padded_m, dim=0)

    def encode_corpus(self, sequences: list[str]) -> None:
        self._corpus_tokens, self._corpus_masks = self._encode(sequences)

    def score_query(self, query_seq: str) -> np.ndarray:
        q_tok, q_mask = self._encode([query_seq])
        q_tok = q_tok.to(self.device); q_mask = q_mask.to(self.device)
        n = self._corpus_tokens.size(0)
        out = torch.empty(n, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, n, self.doc_chunk):
                d = self._corpus_tokens[i : i + self.doc_chunk].to(self.device)
                dm = self._corpus_masks[i : i + self.doc_chunk].to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    s = ColBERTRetriever._score_batched(q_tok, q_mask, d, dm)  # (1, D)
                out[i : i + s.size(1)] = s[0].float().cpu()
        return out.numpy()


# ---------- Trained checkpoint (ColBERT or MeanPool) ------------------------

class CheckpointScorer:
    """Wraps a trained ColBERTRetriever or MeanPoolRetriever checkpoint."""

    def __init__(self, ckpt_path: str | Path, device: str, batch_size: int, doc_chunk: int) -> None:
        self.ckpt_path = str(ckpt_path)
        self.device = device
        self.batch_size = batch_size
        self.doc_chunk = doc_chunk
        self.autocast = device.startswith("cuda")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self._model = build_retriever(ckpt["model_config"])
        self._model.load_state_dict(ckpt["model_state"], strict=False)
        self._model.to(device).eval()
        self.is_colbert = isinstance(self._model, ColBERTRetriever)
        self.is_meanpool = isinstance(self._model, MeanPoolRetriever)
        self.name = ("colbert-" if self.is_colbert else "meanpool-") + Path(ckpt_path).stem
        self._corpus_emb = None
        self._corpus_tokens = None
        self._corpus_masks = None

    def encode_corpus(self, sequences: list[str]) -> None:
        if self.is_colbert:
            all_t, all_m = [], []
            with torch.no_grad():
                for i in range(0, len(sequences), self.batch_size):
                    batch = sequences[i : i + self.batch_size]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                        t, m = self._model.encode(batch)
                    all_t.append(t.float().cpu()); all_m.append(m.bool().cpu())
                    if (i // self.batch_size) % 50 == 0:
                        LOGGER.info("  encoded %d/%d", min(i + self.batch_size, len(sequences)), len(sequences))
            max_len = max(t.size(1) for t in all_t)
            padded_t, padded_m = [], []
            for t, m in zip(all_t, all_m):
                pad = max_len - t.size(1)
                if pad > 0:
                    t = F.pad(t, (0, 0, 0, pad)); m = F.pad(m, (0, pad))
                padded_t.append(t); padded_m.append(m)
            self._corpus_tokens = torch.cat(padded_t, dim=0)
            self._corpus_masks = torch.cat(padded_m, dim=0)
        else:
            embs = []
            with torch.no_grad():
                for i in range(0, len(sequences), self.batch_size):
                    batch = sequences[i : i + self.batch_size]
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                        e = self._model.encode(batch)
                    embs.append(e.float().cpu())
                    if (i // self.batch_size) % 50 == 0:
                        LOGGER.info("  encoded %d/%d", min(i + self.batch_size, len(sequences)), len(sequences))
            self._corpus_emb = torch.cat(embs, dim=0).to(self.device)

    def score_query(self, query_seq: str) -> np.ndarray:
        if self.is_colbert:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    q_tok, q_mask = self._model.encode([query_seq])
            q_tok = q_tok.float().to(self.device); q_mask = q_mask.bool().to(self.device)
            n = self._corpus_tokens.size(0)
            out = torch.empty(n, dtype=torch.float32)
            with torch.no_grad():
                for i in range(0, n, self.doc_chunk):
                    d = self._corpus_tokens[i : i + self.doc_chunk].to(self.device)
                    dm = self._corpus_masks[i : i + self.doc_chunk].to(self.device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                        s = ColBERTRetriever._score_batched(q_tok, q_mask, d, dm)
                    out[i : i + s.size(1)] = s[0].float().cpu()
            return out.numpy()
        else:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
                    q = self._model.encode([query_seq])
            q = q.float().to(self.device)[0]
            return (self._corpus_emb @ q).cpu().numpy().astype(np.float32)


# ---------- Eval driver -----------------------------------------------------

def build_scorer(args, eval_cfg) -> Scorer:
    device = args.device
    encode_batch = int(eval_cfg.get("encode_batch_size", 32))
    doc_chunk = int(eval_cfg.get("doc_chunk_size", 256))
    max_length = int(eval_cfg.get("max_length", 512))
    proj_dim = int(eval_cfg.get("projection_dim", 64))
    if args.checkpoint:
        return CheckpointScorer(args.checkpoint, device, encode_batch, doc_chunk)
    b = args.baseline
    if b == "random":
        return RandomScorer(seed=int(args.seed_override or 0))
    if b.startswith("kmer-"):
        return KmerScorer(k=int(b.split("-")[1]))
    if b.startswith("meanpool-esm2-"):
        return FrozenESMMeanPoolScorer(esm_size=b.split("-")[2], device=device,
                                        batch_size=encode_batch, max_length=max_length)
    if b.startswith("randproj-esm2-"):
        return RandProjMaxSimScorer(esm_size=b.split("-")[2], device=device,
                                     batch_size=encode_batch, max_length=max_length,
                                     projection_dim=proj_dim, doc_chunk=doc_chunk,
                                     seed=int(args.seed_override or 0))
    raise ValueError(f"Unknown baseline: {b}")


def evaluate(scorer: Scorer, query_df: pd.DataFrame, corpus_df: pd.DataFrame,
             sequence_column: str, top_k: list[int]) -> dict:
    corpus_seqs = corpus_df[sequence_column].tolist()
    corpus_ids = corpus_df["sequence_id"].tolist()

    family_to_seqs: dict[str, set[str]] = defaultdict(set)
    clan_to_seqs: dict[str, set[str]] = defaultdict(set)
    for sid, fam, clan in zip(corpus_df["sequence_id"], corpus_df["family_acc"],
                              corpus_df["clan_acc"]):
        family_to_seqs[fam].add(sid)
        if clan:
            clan_to_seqs[clan].add(sid)

    LOGGER.info("Encoding corpus (%d seqs) with %s ...", len(corpus_seqs), scorer.name)
    t0 = time.time()
    scorer.encode_corpus(corpus_seqs)
    LOGGER.info("  done in %.1fs", time.time() - t0)

    metrics: dict[str, list[float]] = defaultdict(list)
    queries_with_clan = 0; skipped = 0
    t0 = time.time()
    for i, row in enumerate(query_df.itertuples(index=False)):
        qid = row.sequence_id
        qseq = getattr(row, sequence_column)
        qfam = row.family_acc
        qclan = row.clan_acc or ""

        same_family = family_to_seqs[qfam] - {qid}
        same_clan_diff_family = (clan_to_seqs[qclan] - same_family - {qid}) if qclan else set()
        if not same_family:
            skipped += 1
            continue

        scores = scorer.score_query(qseq)
        order = np.argsort(scores)[::-1]
        ranked_ids = [corpus_ids[idx] for idx in order if corpus_ids[idx] != qid]

        metrics["fam_mrr"].append(reciprocal_rank(ranked_ids, same_family))
        metrics["fam_ndcg@10"].append(ndcg_at_k(ranked_ids, same_family, 10))
        for k in top_k:
            metrics[f"fam_recall@{k}"].append(recall_at_k(ranked_ids, same_family, k))

        if same_clan_diff_family:
            queries_with_clan += 1
            metrics["clan_mrr"].append(reciprocal_rank(ranked_ids, same_clan_diff_family))
            metrics["clan_ndcg@10"].append(ndcg_at_k(ranked_ids, same_clan_diff_family, 10))
            for k in top_k:
                metrics[f"clan_recall@{k}"].append(recall_at_k(ranked_ids, same_clan_diff_family, k))

        for k in top_k:
            top = ranked_ids[:k]
            n_fam = sum(1 for d in top if d in same_family)
            n_hard = sum(1 for d in top if d in same_clan_diff_family)
            n_easy = k - n_fam - n_hard
            metrics[f"top{k}_pct_same_family"].append(100.0 * n_fam / k)
            metrics[f"top{k}_pct_hard_neg"].append(100.0 * n_hard / k)
            metrics[f"top{k}_pct_easy_neg"].append(100.0 * n_easy / k)

        if qclan and same_clan_diff_family:
            clan_pool = same_family | same_clan_diff_family
            ranked_clan_only = [d for d in ranked_ids if d in clan_pool]
            metrics["hard_mrr"].append(reciprocal_rank(ranked_clan_only, same_family))

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(query_df) - i - 1)
            LOGGER.info("  query %d/%d  (elapsed %.1fs, eta %.1fs)", i + 1, len(query_df), elapsed, eta)

    summary = {m: float(np.mean(v)) if v else 0.0 for m, v in metrics.items()}
    summary.update({
        "scorer": scorer.name,
        "n_queries": int(len(query_df)),
        "n_queries_evaluated": int(len(query_df) - skipped),
        "n_queries_with_clan": int(queries_with_clan),
        "n_corpus": int(len(corpus_df)),
    })
    return summary


def print_summary(summary: dict, top_k: list[int]) -> None:
    LOGGER.info("=" * 70)
    LOGGER.info("EVAL  scorer=%s  queries=%d  corpus=%d",
                summary["scorer"], summary["n_queries_evaluated"], summary["n_corpus"])
    LOGGER.info("=" * 70)
    LOGGER.info("Standard retrieval (same-family positives):")
    LOGGER.info("  MRR     : %.4f", summary.get("fam_mrr", 0.0))
    LOGGER.info("  nDCG@10 : %.4f", summary.get("fam_ndcg@10", 0.0))
    for k in top_k:
        LOGGER.info("  Recall@%-3d: %.4f", k, summary.get(f"fam_recall@{k}", 0.0))
    LOGGER.info("Distant homolog (same-clan diff-family positives, n=%d):",
                summary.get("n_queries_with_clan", 0))
    LOGGER.info("  MRR     : %.4f", summary.get("clan_mrr", 0.0))
    LOGGER.info("  nDCG@10 : %.4f", summary.get("clan_ndcg@10", 0.0))
    for k in top_k:
        LOGGER.info("  Recall@%-3d: %.4f", k, summary.get(f"clan_recall@{k}", 0.0))
    LOGGER.info("Top-k composition (%% same-fam / hard-neg / easy-neg):")
    for k in top_k:
        LOGGER.info("  @%-3d %5.1f / %5.1f / %5.1f", k,
                    summary.get(f"top{k}_pct_same_family", 0.0),
                    summary.get(f"top{k}_pct_hard_neg", 0.0),
                    summary.get(f"top{k}_pct_easy_neg", 0.0))
    LOGGER.info("Hard MRR (positive vs same-clan distractors only): %.4f",
                summary.get("hard_mrr", 0.0))


def default_output_path(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint).parent.parent / "metrics" / "eval_pfam.json"
    return Path("outputs/pfam_baselines") / f"eval_pfam_{args.baseline}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Pfam full-corpus retrieval eval")
    parser.add_argument("--config", default="configs/train/eval_pfam.yaml")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", default=None)
    src.add_argument("--baseline", default=None,
                     help="One of: random, kmer-K (K in {3,4,5,...}), "
                          "meanpool-esm2-{35m,650m}, randproj-esm2-{35m,650m}")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-override", default=None,
                        help="Override the seed used by random/randproj baselines")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["seed"]))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    manifest = pd.read_csv(cfg["data"]["manifest_path"]).fillna({"clan_acc": ""})
    sequence_column = cfg["data"]["sequence_column"]
    eval_cfg = cfg["evaluation"]
    top_k = [int(k) for k in eval_cfg["top_k"]]
    split = eval_cfg.get("split", "test")
    sample_queries = int(eval_cfg.get("sample_queries", 0))

    pool = manifest[manifest["split"] == split].dropna(subset=[sequence_column]).reset_index(drop=True)
    LOGGER.info("Pool: %d sequences in split=%s", len(pool), split)
    if sample_queries and sample_queries < len(pool):
        query_df = pool.sample(n=sample_queries, random_state=int(cfg["seed"])).copy()
    else:
        query_df = pool.copy()
    corpus_df = pool

    scorer = build_scorer(args, eval_cfg)
    summary = evaluate(scorer, query_df, corpus_df, sequence_column, top_k)
    out_path = Path(args.output) if args.output else default_output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_path)
    print_summary(summary, top_k)
    LOGGER.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
