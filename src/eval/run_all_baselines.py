"""Run all baselines for cCRE retrieval evaluation.

Baselines:
  1. random              — uniform random scores (done)
  2. gc_matched_random   — random within same GC bin
  3. kmer_cosine         — 6-mer cosine similarity (done)
  4. tfidf_kmer          — TF-IDF weighted 6-mer cosine
  5. minhash             — MinHash approximate k-mer Jaccard
  6. blastn              — BLASTn local alignment scores
  7. gkm_svm             — gapped k-mer SVM kernel similarity
  8. pretrained_dnabert2  — mean-pooled pretrained DNABERT-2 (done)
  9. oracle              — cosine on true TF binding vectors
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.baselines import kmer_counts, kmer_similarity_scores, random_scores
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


# ============================================================
# Scoring functions for each baseline
# ============================================================

def gc_content(seq: str) -> float:
    seq = seq.upper()
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq) if len(seq) > 0 else 0.5


def gc_matched_random_scores(
    query_seq: str, doc_sequences: list[str], doc_gc_bins: np.ndarray,
    query_gc_bin: int, seed: int,
) -> np.ndarray:
    """Random scores, but docs in different GC bin get -inf."""
    rng = np.random.default_rng(seed)
    scores = rng.random(len(doc_sequences))
    mask = doc_gc_bins != query_gc_bin
    scores[mask] = -1e9
    return scores


def tfidf_kmer_scores(
    query_seq: str, doc_sequences: list[str], k: int,
    idf_weights: dict[str, float],
) -> np.ndarray:
    """TF-IDF weighted k-mer cosine similarity."""
    def tfidf_vector(seq):
        counts = kmer_counts(seq, k)
        vec = {}
        for kmer, count in counts.items():
            if kmer in idf_weights:
                vec[kmer] = count * idf_weights[kmer]
        return vec

    def cosine_sparse(a, b):
        keys = set(a) & set(b)
        if not keys:
            return 0.0
        num = sum(a[k] * b[k] for k in keys)
        na = sum(v*v for v in a.values()) ** 0.5
        nb = sum(v*v for v in b.values()) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return num / (na * nb)

    q_vec = tfidf_vector(query_seq)
    return np.array([cosine_sparse(q_vec, tfidf_vector(d)) for d in doc_sequences])


def compute_idf(doc_sequences: list[str], k: int) -> dict[str, float]:
    """Compute IDF weights for k-mers across the corpus."""
    n_docs = len(doc_sequences)
    doc_freq: Counter[str] = Counter()
    for seq in doc_sequences:
        unique_kmers = set(seq[i:i+k] for i in range(len(seq) - k + 1))
        for kmer in unique_kmers:
            doc_freq[kmer] += 1
    idf = {}
    for kmer, df in doc_freq.items():
        idf[kmer] = float(np.log(n_docs / df))
    return idf


def build_minhash_signatures(sequences: list[str], k: int, n_hashes: int = 128):
    """Pre-compute MinHash signatures for all sequences. Returns (n_seqs, n_hashes) array."""
    # Pre-generate hash coefficients
    rng = np.random.RandomState(42)
    prime = (1 << 61) - 1
    a_coeffs = rng.randint(1, prime, size=n_hashes, dtype=np.int64)
    b_coeffs = rng.randint(0, prime, size=n_hashes, dtype=np.int64)

    # Map each unique k-mer to an integer
    kmer_to_int: dict[str, int] = {}
    counter = 0

    sigs = np.full((len(sequences), n_hashes), np.iinfo(np.int64).max, dtype=np.int64)
    for idx, seq in enumerate(sequences):
        seq = seq.upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if not all(c in "ACGT" for c in kmer):
                continue
            if kmer not in kmer_to_int:
                kmer_to_int[kmer] = counter
                counter += 1
            h = np.int64(kmer_to_int[kmer])
            # Vectorized hash: (a*h + b) mod prime
            hashes = (a_coeffs * h + b_coeffs) % prime
            sigs[idx] = np.minimum(sigs[idx], hashes)
    return sigs


def minhash_scores_precomputed(query_sig: np.ndarray, doc_sigs: np.ndarray) -> np.ndarray:
    """Approximate Jaccard from pre-computed MinHash signatures."""
    return (query_sig[None, :] == doc_sigs).mean(axis=1).astype(float)


def oracle_scores(
    query_id: str, doc_ids: list[str], tf_vectors: dict[str, np.ndarray],
) -> np.ndarray:
    """Score by cosine similarity on true TF binding vectors."""
    q_vec = tf_vectors.get(query_id)
    if q_vec is None or q_vec.sum() == 0:
        return np.zeros(len(doc_ids))
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    scores = np.zeros(len(doc_ids))
    for i, did in enumerate(doc_ids):
        d_vec = tf_vectors.get(did)
        if d_vec is not None and d_vec.sum() > 0:
            d_norm = d_vec / (np.linalg.norm(d_vec) + 1e-10)
            scores[i] = float(np.dot(q_norm, d_norm))
    return scores


def blastn_scores(
    query_seq: str, query_id: str, doc_sequences: list[str], doc_ids: list[str],
    blast_db_dir: str,
) -> np.ndarray:
    """Score using BLASTn. Returns bit scores."""
    scores = np.zeros(len(doc_ids))

    # Write query to temp fasta
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as qf:
        qf.write(f">{query_id}\n{query_seq}\n")
        query_file = qf.name

    try:
        result = subprocess.run(
            ["blastn", "-query", query_file, "-db", os.path.join(blast_db_dir, "corpus"),
             "-outfmt", "6 sseqid bitscore", "-max_target_seqs", str(len(doc_ids)),
             "-evalue", "10", "-dust", "no", "-word_size", "7"],
            capture_output=True, text=True, timeout=30,
        )
        id_to_idx = {did: i for i, did in enumerate(doc_ids)}
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0] in id_to_idx:
                scores[id_to_idx[parts[0]]] = max(scores[id_to_idx[parts[0]]], float(parts[1]))
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOGGER.warning("BLASTn failed for %s: %s", query_id, e)
    finally:
        os.unlink(query_file)

    return scores


def build_blast_db(doc_ids: list[str], doc_sequences: list[str], db_dir: str):
    """Build a BLASTn database from corpus sequences."""
    os.makedirs(db_dir, exist_ok=True)
    fasta_path = os.path.join(db_dir, "corpus.fa")
    with open(fasta_path, "w") as f:
        for did, seq in zip(doc_ids, doc_sequences):
            f.write(f">{did}\n{seq}\n")
    subprocess.run(
        ["makeblastdb", "-in", fasta_path, "-dbtype", "nucl",
         "-out", os.path.join(db_dir, "corpus")],
        capture_output=True, check=True,
    )
    LOGGER.info("Built BLAST db with %d sequences at %s", len(doc_ids), db_dir)


def _gapped_kmer_vector(seq: str, k: int = 10, l: int = 6) -> Counter:
    """Extract gapped k-mer features: choose l positions from windows of length k."""
    from itertools import combinations
    counts: Counter[str] = Counter()
    seq = seq.upper()
    positions_combos = list(combinations(range(k), l))
    for i in range(len(seq) - k + 1):
        window = seq[i:i+k]
        if not all(c in "ACGT" for c in window):
            continue
        for positions in positions_combos:
            gkmer = "".join(window[p] for p in positions)
            counts[gkmer] += 1
    return counts


def _cosine_counters(a: Counter, b: Counter) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    num = sum(a[k] * b[k] for k in keys)
    na = sum(v*v for v in a.values()) ** 0.5
    nb = sum(v*v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return num / (na * nb)


def _gkm_worker(seq):
    """Worker function for multiprocessing gkm vector computation."""
    return _gapped_kmer_vector(seq, k=10, l=6)


def build_gkm_vectors(sequences: list[str], n_workers: int = 12) -> list[Counter]:
    """Pre-compute gapped k-mer vectors for all sequences."""
    vectors = []
    for i, seq in enumerate(sequences):
        vectors.append(_gapped_kmer_vector(seq, k=10, l=6))
        if (i + 1) % 200 == 0:
            LOGGER.info("  gkm vectors: %d / %d", i + 1, len(sequences))
    return vectors


def gkm_svm_scores_precomputed(query_vec: Counter, doc_vectors: list[Counter]) -> np.ndarray:
    scores = np.zeros(len(doc_vectors))
    for i, d_vec in enumerate(doc_vectors):
        scores[i] = _cosine_counters(query_vec, d_vec)
    return scores


# ============================================================
# Main evaluation loop
# ============================================================

def evaluate_baseline(
    name: str, score_fn, query_df, doc_ids, doc_sequences,
    relevance_lookup, label_lookup, activity_lookup, top_k, cfg,
):
    """Run evaluation for a single baseline."""
    results = {f"Recall@{k}": [] for k in top_k}
    results.update({"MRR": [], "nDCG": []})
    for k in top_k:
        results[f"class_purity@{k}"] = []
        results[f"activity_jaccard@{k}"] = []
        results[f"activity_corr@{k}"] = []

    seq_col = cfg["data"]["sequence_column"]
    n_evaluated = 0

    for row in query_df.itertuples(index=False):
        if row.ccre_id not in relevance_lookup:
            continue

        scores = score_fn(row)

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
            act_jacs = [binary_jaccard(query_activity, activity_lookup.get(d, [0.0]*len(query_activity))) for d in top_docs]
            act_corrs = [safe_pearson(query_activity, activity_lookup.get(d, [0.0]*len(query_activity))) for d in top_docs]
            results[f"activity_jaccard@{k}"].append(float(np.mean(act_jacs)) if act_jacs else 0.0)
            results[f"activity_corr@{k}"].append(float(np.mean(act_corrs)) if act_corrs else 0.0)

        n_evaluated += 1

    summary = {metric: float(np.mean(values)) if values else 0.0 for metric, values in results.items()}
    summary["baseline"] = name
    summary["num_queries"] = n_evaluated
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/eval_dnabert2_small.yaml")
    parser.add_argument("--tf-activity", default="data/interim/tf_vectors/small_tf.npz")
    parser.add_argument("--output-dir", default="outputs/metrics")
    parser.add_argument("--baselines", nargs="+",
                        default=["gc_matched_random", "tfidf_kmer", "minhash",
                                 "blastn", "gkm_svm", "oracle"])
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_global_seed(int(cfg["seed"]))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg["data"]["manifest_path"])
    pairs = pd.read_csv(cfg["data"]["pair_path"])
    seq_col = cfg["data"]["sequence_column"]
    top_k = [int(k) for k in cfg["evaluation"]["top_k"]]

    query_df = manifest[manifest["split"] == "test"].dropna(subset=[seq_col]).copy()
    corpus_df = manifest[manifest["split"] == "test"].dropna(subset=[seq_col]).copy()
    sample_queries = min(int(cfg["evaluation"]["sample_queries"]), len(query_df))
    sample_corpus = min(int(cfg["evaluation"]["sample_corpus"]), len(corpus_df))
    query_df = query_df.sample(n=sample_queries, random_state=int(cfg["seed"]))
    corpus_df = corpus_df.sample(n=sample_corpus, random_state=int(cfg["seed"]))

    doc_ids = corpus_df["ccre_id"].tolist()
    doc_sequences = corpus_df[seq_col].tolist()
    label_lookup = manifest.set_index("ccre_id")["ccre_class"].astype(str).to_dict()

    # Load TF activity vectors
    tf_data = np.load(args.tf_activity, allow_pickle=True)
    tf_ccre_ids = tf_data["ccre_ids"]
    tf_matrix = tf_data["tf_matrix"]
    n_tfs = tf_matrix.shape[1]
    tf_lookup_map = {cid: i for i, cid in enumerate(tf_ccre_ids)}
    activity_lookup = {
        cid: tf_matrix[tf_lookup_map[cid]].astype(float).tolist() if cid in tf_lookup_map else [0.0] * n_tfs
        for cid in manifest["ccre_id"]
    }
    # Also build numpy dict for oracle
    tf_vectors = {
        cid: tf_matrix[tf_lookup_map[cid]].astype(float) if cid in tf_lookup_map else np.zeros(n_tfs)
        for cid in manifest["ccre_id"]
    }

    relevance_lookup = (
        pairs[(pairs["split"] == "test") & (pairs["label"] == 1)]
        .groupby("query_id")["doc_id"]
        .apply(set)
        .to_dict()
    )

    # Pre-compute shared resources
    doc_gc = np.array([gc_content(s) for s in doc_sequences])
    doc_gc_bins = pd.qcut(doc_gc, q=10, labels=False, duplicates="drop")
    query_gc_map = {row.ccre_id: gc_content(getattr(row, seq_col))
                    for row in query_df.itertuples(index=False)}
    query_gc_bin_map = {}
    bin_edges = np.percentile(doc_gc, np.linspace(0, 100, 11))
    for qid, qgc in query_gc_map.items():
        query_gc_bin_map[qid] = int(np.clip(np.digitize(qgc, bin_edges[1:-1]), 0, 9))

    kmer_k = int(cfg["evaluation"]["kmer_size"])

    # Pre-compute MinHash signatures
    minhash_doc_sigs = None
    minhash_query_sigs = None
    if "minhash" in args.baselines:
        LOGGER.info("Pre-computing MinHash signatures for %d corpus + %d query sequences...",
                     len(doc_sequences), len(query_df))
        all_seqs = doc_sequences + query_df[seq_col].tolist()
        all_sigs = build_minhash_signatures(all_seqs, kmer_k, n_hashes=128)
        minhash_doc_sigs = all_sigs[:len(doc_sequences)]
        minhash_query_sigs = all_sigs[len(doc_sequences):]
        minhash_query_map = {row.ccre_id: i for i, row in enumerate(query_df.itertuples(index=False))}
        LOGGER.info("MinHash signatures computed")

    # Pre-compute IDF for TF-IDF
    idf_weights = None
    if "tfidf_kmer" in args.baselines:
        LOGGER.info("Computing IDF weights for %d corpus documents...", len(doc_sequences))
        idf_weights = compute_idf(doc_sequences, kmer_k)
        LOGGER.info("IDF computed for %d unique k-mers", len(idf_weights))

    # Pre-compute gkm-SVM vectors
    gkm_doc_vectors = None
    gkm_query_vectors = None
    if "gkm_svm" in args.baselines:
        LOGGER.info("Pre-computing gapped k-mer vectors for %d corpus docs (12 workers)...",
                     len(doc_sequences))
        gkm_doc_vectors = build_gkm_vectors(doc_sequences, n_workers=12)
        LOGGER.info("Pre-computing gapped k-mer vectors for %d queries...", len(query_df))
        gkm_query_vectors = build_gkm_vectors(query_df[seq_col].tolist(), n_workers=12)
        gkm_query_map = {row.ccre_id: i for i, row in enumerate(query_df.itertuples(index=False))}
        LOGGER.info("gkm vectors computed")

    # Build BLAST db if needed
    blast_db_dir = None
    if "blastn" in args.baselines:
        blast_db_dir = "outputs/blast_db"
        try:
            build_blast_db(doc_ids, doc_sequences, blast_db_dir)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            LOGGER.warning("Could not build BLAST db: %s. Skipping blastn.", e)
            args.baselines = [b for b in args.baselines if b != "blastn"]

    # Define score functions
    seed = int(cfg["evaluation"].get("random_seed", 13))

    def make_scorer(baseline_name):
        def scorer(row):
            seq = getattr(row, seq_col)
            if baseline_name == "gc_matched_random":
                qbin = query_gc_bin_map.get(row.ccre_id, 0)
                return gc_matched_random_scores(seq, doc_sequences, doc_gc_bins, qbin, seed)
            elif baseline_name == "tfidf_kmer":
                return tfidf_kmer_scores(seq, doc_sequences, kmer_k, idf_weights)
            elif baseline_name == "minhash":
                qi = minhash_query_map[row.ccre_id]
                return minhash_scores_precomputed(minhash_query_sigs[qi], minhash_doc_sigs)
            elif baseline_name == "blastn":
                return blastn_scores(seq, row.ccre_id, doc_sequences, doc_ids, blast_db_dir)
            elif baseline_name == "gkm_svm":
                qi = gkm_query_map[row.ccre_id]
                return gkm_svm_scores_precomputed(gkm_query_vectors[qi], gkm_doc_vectors)
            elif baseline_name == "oracle":
                return oracle_scores(row.ccre_id, doc_ids, tf_vectors)
        return scorer

    # Run each baseline
    for baseline_name in args.baselines:
        LOGGER.info("=" * 60)
        LOGGER.info("Running baseline: %s", baseline_name)
        LOGGER.info("=" * 60)

        scorer = make_scorer(baseline_name)
        summary = evaluate_baseline(
            baseline_name, scorer, query_df, doc_ids, doc_sequences,
            relevance_lookup, label_lookup, activity_lookup, top_k, cfg,
        )

        out_path = out_dir / f"eval_{baseline_name}.json"
        dump_json(summary, str(out_path))

        LOGGER.info("Results for %s:", baseline_name)
        for m in ["MRR", "nDCG", "Recall@1", "Recall@5", "Recall@10", "Recall@50",
                   "class_purity@10", "activity_jaccard@10"]:
            LOGGER.info("  %s: %.4f", m, summary.get(m, 0))
        LOGGER.info("Saved to %s", out_path)

    LOGGER.info("All baselines complete.")


if __name__ == "__main__":
    main()
