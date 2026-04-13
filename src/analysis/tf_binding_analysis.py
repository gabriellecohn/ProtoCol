"""Comprehensive TF binding vector analysis for the 50k cCRE subset.

Tasks 1-5: distribution stats, ubiquitous TF filtering, yield analysis,
k-mer sanity check, and class-conditioned profiling with UMAP.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

OUT_DIR = Path("outputs/tf_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(manifest_path: str, tf_path: str):
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.dropna(subset=["ccre_id", "ccre_class", "split"]).reset_index(drop=True)
    tf_data = np.load(tf_path, allow_pickle=True)
    tf_ccre_ids = tf_data["ccre_ids"]
    tf_matrix = tf_data["tf_matrix"]
    tf_names = tf_data["tf_names"]
    tf_biosamples = tf_data["tf_biosamples"]
    tf_experiments = tf_data["tf_experiments"]

    # Align TF matrix to manifest order
    tf_lookup = {cid: i for i, cid in enumerate(tf_ccre_ids)}
    n_tfs = tf_matrix.shape[1]
    aligned_matrix = np.zeros((len(manifest), n_tfs), dtype=np.uint8)
    matched = 0
    for i, cid in enumerate(manifest["ccre_id"]):
        if cid in tf_lookup:
            aligned_matrix[i] = tf_matrix[tf_lookup[cid]]
            matched += 1
    LOGGER.info("Aligned %d / %d cCREs to TF matrix", matched, len(manifest))
    return manifest, aligned_matrix, tf_names, tf_biosamples, tf_experiments


def binary_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def binary_jaccard_batch(pairs_a: np.ndarray, pairs_b: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(pairs_a, pairs_b).sum(axis=1)
    union = np.logical_or(pairs_a, pairs_b).sum(axis=1)
    mask = union > 0
    result = np.zeros(len(pairs_a), dtype=float)
    result[mask] = intersection[mask] / union[mask]
    return result


def kmer_vector(seq: str, k: int = 6) -> np.ndarray:
    counts = Counter()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if all(c in "ACGT" for c in kmer):
            counts[kmer] += 1
    vec = np.zeros(4**k, dtype=float)
    base_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    for kmer, count in counts.items():
        idx = sum(base_map[c] * (4 ** (k - 1 - j)) for j, c in enumerate(kmer))
        vec[idx] = count
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def kmer_cosine(seq1: str, seq2: str, k: int = 6) -> float:
    v1 = kmer_vector(seq1, k)
    v2 = kmer_vector(seq2, k)
    return float(np.dot(v1, v2))


# ============================================================
# TASK 1: TF binding vector statistics
# ============================================================
def task1(manifest, tf_matrix, tf_names, tf_biosamples, tf_experiments):
    LOGGER.info("=== TASK 1: TF binding vector statistics ===")
    stats = {}

    # 1a: Per-cCRE density
    row_sums = tf_matrix.sum(axis=1)
    percentiles = [0, 5, 25, 50, 75, 95, 100]
    pct_vals = np.percentile(row_sums, percentiles)
    stats["per_ccre"] = {
        f"p{p}": float(v) for p, v in zip(percentiles, pct_vals)
    }
    stats["per_ccre"]["mean"] = float(row_sums.mean())
    stats["per_ccre"]["zero_count"] = int((row_sums == 0).sum())
    stats["per_ccre"]["zero_fraction"] = float((row_sums == 0).mean())

    LOGGER.info("Per-cCRE TF density: min=%d, median=%.0f, max=%d, zeros=%d (%.1f%%)",
                row_sums.min(), np.median(row_sums), row_sums.max(),
                stats["per_ccre"]["zero_count"], stats["per_ccre"]["zero_fraction"] * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(row_sums, bins=100, edgecolor="black", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Number of TF experiments per cCRE")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of TF binding density per cCRE (50k subset)")
    ax.axvline(np.median(row_sums), color="red", linestyle="--", label=f"Median={np.median(row_sums):.0f}")
    ax.legend()
    fig.savefig(OUT_DIR / "task1_ccre_tf_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 1b: Per-experiment density
    col_sums = tf_matrix.sum(axis=0)
    stats["per_experiment"] = {
        "min": int(col_sums.min()),
        "median": float(np.median(col_sums)),
        "mean": float(col_sums.mean()),
        "max": int(col_sums.max()),
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(col_sums, bins=100, edgecolor="black", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Number of cCREs per TF experiment")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Distribution of cCRE count per TF experiment (50k subset)")
    ax.axvline(np.median(col_sums), color="red", linestyle="--", label=f"Median={np.median(col_sums):.0f}")
    ax.legend()
    fig.savefig(OUT_DIR / "task1_experiment_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Top 20 experiments
    top20_idx = np.argsort(col_sums)[::-1][:20]
    top20 = []
    for idx in top20_idx:
        top20.append({
            "rank": len(top20) + 1,
            "experiment": str(tf_experiments[idx]),
            "tf_name": str(tf_names[idx]),
            "biosample": str(tf_biosamples[idx]),
            "ccre_count": int(col_sums[idx]),
            "fraction": float(col_sums[idx] / len(tf_matrix)),
        })
    stats["top20_experiments"] = top20
    LOGGER.info("Top 5 TF experiments:")
    for entry in top20[:5]:
        LOGGER.info("  %s (%s in %s): %d cCREs (%.1f%%)",
                     entry["experiment"], entry["tf_name"], entry["biosample"],
                     entry["ccre_count"], entry["fraction"] * 100)

    # 1c: Pairwise Jaccard distributions
    rng = np.random.default_rng(42)
    n_pairs = 50000

    # Random pairs
    idx_a = rng.integers(0, len(tf_matrix), size=n_pairs)
    idx_b = rng.integers(0, len(tf_matrix), size=n_pairs)
    random_jaccards = binary_jaccard_batch(tf_matrix[idx_a], tf_matrix[idx_b])

    # Same-class pairs
    classes = manifest["ccre_class"].astype(str).values
    class_to_indices = defaultdict(list)
    for i, c in enumerate(classes):
        class_to_indices[c].append(i)

    same_class_a, same_class_b = [], []
    for _ in range(n_pairs):
        cls = rng.choice(list(class_to_indices.keys()))
        indices = class_to_indices[cls]
        if len(indices) < 2:
            continue
        a, b = rng.choice(indices, size=2, replace=False)
        same_class_a.append(a)
        same_class_b.append(b)
    same_class_a = np.array(same_class_a)
    same_class_b = np.array(same_class_b)
    same_class_jaccards = binary_jaccard_batch(tf_matrix[same_class_a], tf_matrix[same_class_b])

    # Filter out undefined (both zero)
    both_nonzero_rand = (tf_matrix[idx_a].sum(axis=1) > 0) & (tf_matrix[idx_b].sum(axis=1) > 0)
    both_nonzero_sc = (tf_matrix[same_class_a].sum(axis=1) > 0) & (tf_matrix[same_class_b].sum(axis=1) > 0)
    random_jaccards_valid = random_jaccards[both_nonzero_rand]
    same_class_jaccards_valid = same_class_jaccards[both_nonzero_sc]

    for name, arr in [("random_pairs", random_jaccards_valid), ("same_class_pairs", same_class_jaccards_valid)]:
        stats[f"jaccard_{name}"] = {
            "count": int(len(arr)),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    LOGGER.info("Jaccard — random pairs: median=%.4f, mean=%.4f, p95=%.4f",
                stats["jaccard_random_pairs"]["median"],
                stats["jaccard_random_pairs"]["mean"],
                stats["jaccard_random_pairs"]["p95"])
    LOGGER.info("Jaccard — same-class pairs: median=%.4f, mean=%.4f, p95=%.4f",
                stats["jaccard_same_class_pairs"]["median"],
                stats["jaccard_same_class_pairs"]["mean"],
                stats["jaccard_same_class_pairs"]["p95"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 0.6, 80)
    ax.hist(random_jaccards_valid, bins=bins, alpha=0.5, label=f"Random pairs (n={len(random_jaccards_valid)})", density=True)
    ax.hist(same_class_jaccards_valid, bins=bins, alpha=0.5, label=f"Same-class pairs (n={len(same_class_jaccards_valid)})", density=True)
    ax.set_xlabel("TF Binding Jaccard Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise TF Jaccard: Random vs Same-Class Pairs")
    ax.legend()
    fig.savefig(OUT_DIR / "task1_jaccard_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 1d: Overlap count distribution for same-class pairs
    overlap_counts = np.logical_and(tf_matrix[same_class_a], tf_matrix[same_class_b]).sum(axis=1)
    stats["overlap_same_class"] = {}
    for thresh in [2, 3, 4, 5]:
        frac = float((overlap_counts >= thresh).mean())
        stats["overlap_same_class"][f"frac_gte_{thresh}"] = frac
        LOGGER.info("Same-class pairs with overlap >= %d: %.1f%%", thresh, frac * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(overlap_counts, bins=range(0, int(overlap_counts.max()) + 2), edgecolor="black", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Number of shared TF experiments")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("TF Overlap Count Distribution (Same-Class Pairs)")
    fig.savefig(OUT_DIR / "task1_overlap_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # For overlap==2 pairs, what TFs are shared?
    overlap2_mask = overlap_counts == 2
    if overlap2_mask.sum() > 0:
        shared_tf_counter = Counter()
        overlap2_indices = np.where(overlap2_mask)[0][:500]  # sample up to 500
        for idx in overlap2_indices:
            shared = np.where(tf_matrix[same_class_a[idx]] & tf_matrix[same_class_b[idx]])[0]
            for tf_idx in shared:
                shared_tf_counter[str(tf_names[tf_idx])] += 1
        stats["overlap2_top_tfs"] = dict(shared_tf_counter.most_common(20))
        LOGGER.info("Top TFs in overlap==2 pairs: %s", dict(shared_tf_counter.most_common(5)))

    # Save intermediate data for other tasks
    return stats, {
        "random_jaccards": random_jaccards_valid,
        "same_class_jaccards": same_class_jaccards_valid,
        "same_class_a": same_class_a,
        "same_class_b": same_class_b,
        "class_to_indices": class_to_indices,
        "col_sums": col_sums,
    }


# ============================================================
# TASK 2: Ubiquitous TF filtering analysis
# ============================================================
def task2(manifest, tf_matrix, tf_names, col_sums, same_class_a, same_class_b):
    LOGGER.info("=== TASK 2: Ubiquitous TF filtering analysis ===")
    stats = {}
    n_ccres = len(tf_matrix)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    thresholds = [10, 20, 30]

    for ax, pct in zip(axes, thresholds):
        cutoff = n_ccres * pct / 100
        ubiq_mask = col_sums >= cutoff
        n_ubiq = int(ubiq_mask.sum())
        stats[f"ubiq_{pct}pct"] = {"n_experiments": n_ubiq, "threshold": int(cutoff)}
        LOGGER.info("Ubiquitous at %d%%: %d experiments (binding >= %d cCREs)",
                     pct, n_ubiq, int(cutoff))

        # Filter matrix
        filtered = tf_matrix.copy()
        filtered[:, ubiq_mask] = 0

        # Recompute Jaccard for same-class pairs
        sc_jaccards_filtered = binary_jaccard_batch(filtered[same_class_a], filtered[same_class_b])
        both_nonzero = (filtered[same_class_a].sum(axis=1) > 0) & (filtered[same_class_b].sum(axis=1) > 0)
        sc_valid = sc_jaccards_filtered[both_nonzero]

        stats[f"ubiq_{pct}pct"]["jaccard_median"] = float(np.median(sc_valid)) if len(sc_valid) > 0 else 0
        stats[f"ubiq_{pct}pct"]["jaccard_mean"] = float(sc_valid.mean()) if len(sc_valid) > 0 else 0
        stats[f"ubiq_{pct}pct"]["jaccard_p95"] = float(np.percentile(sc_valid, 95)) if len(sc_valid) > 0 else 0
        stats[f"ubiq_{pct}pct"]["n_valid_pairs"] = int(len(sc_valid))

        # Unfiltered for comparison
        sc_jaccards_orig = binary_jaccard_batch(tf_matrix[same_class_a], tf_matrix[same_class_b])
        both_nz_orig = (tf_matrix[same_class_a].sum(axis=1) > 0) & (tf_matrix[same_class_b].sum(axis=1) > 0)
        sc_orig_valid = sc_jaccards_orig[both_nz_orig]

        bins = np.linspace(0, 0.6, 80)
        ax.hist(sc_orig_valid, bins=bins, alpha=0.5, label="Unfiltered", density=True)
        ax.hist(sc_valid, bins=bins, alpha=0.5, label=f"Filtered (>{pct}%)", density=True)
        ax.set_xlabel("TF Jaccard")
        ax.set_title(f"Filter >{pct}% ({n_ubiq} TFs removed)")
        ax.legend()

    fig.suptitle("Effect of Ubiquitous TF Filtering on Same-Class Jaccard", fontsize=14)
    fig.savefig(OUT_DIR / "task2_ubiq_filtering.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return stats


# ============================================================
# TASK 3: Positive and negative yield analysis
# ============================================================
def task3(manifest, tf_matrix, class_to_indices, col_sums, recommended_filter_pct=None):
    LOGGER.info("=== TASK 3: Positive/negative yield analysis ===")
    rng = np.random.default_rng(42)

    # Optionally filter ubiquitous TFs
    working_matrix = tf_matrix
    if recommended_filter_pct:
        cutoff = len(tf_matrix) * recommended_filter_pct / 100
        ubiq_mask = col_sums >= cutoff
        working_matrix = tf_matrix.copy()
        working_matrix[:, ubiq_mask] = 0
        LOGGER.info("Filtering %d ubiquitous TF experiments (>%d%% binding)",
                     ubiq_mask.sum(), recommended_filter_pct)

    classes = manifest["ccre_class"].astype(str).values
    splits = manifest["split"].astype(str).values
    train_mask = splits == "train"

    # Sample 1000 queries from training split
    train_indices = np.where(train_mask)[0]
    query_indices = rng.choice(train_indices, size=min(1000, len(train_indices)), replace=False)

    pos_thresholds = [0.3, 0.4, 0.5, 0.6]
    overlap_thresholds = [2, 3, 5]
    neg_thresholds = [0.05, 0.1, 0.15]

    results_by_class = defaultdict(lambda: defaultdict(list))
    results_all = defaultdict(list)

    for qi in query_indices:
        q_class = classes[qi]
        q_vec = working_matrix[qi]

        # Same-class training candidates
        same_class = [idx for idx in class_to_indices[q_class]
                       if idx != qi and train_mask[idx]]
        if len(same_class) > 512:
            same_class = list(rng.choice(same_class, size=512, replace=False))

        if not same_class or q_vec.sum() == 0:
            continue

        cand_vecs = working_matrix[np.array(same_class)]
        jaccards = binary_jaccard_batch(
            np.tile(q_vec, (len(same_class), 1)),
            cand_vecs,
        )
        overlaps = np.logical_and(
            np.tile(q_vec, (len(same_class), 1)),
            cand_vecs,
        ).sum(axis=1)

        row = {"class": q_class}
        for t in pos_thresholds:
            row[f"pos_jaccard_{t}"] = int((jaccards >= t).sum())
        for t in overlap_thresholds:
            row[f"pos_overlap_{t}"] = int((overlaps >= t).sum())
        for t in neg_thresholds:
            row[f"neg_jaccard_{t}"] = int((jaccards <= t).sum())

        for key, val in row.items():
            if key != "class":
                results_all[key].append(val)
                results_by_class[q_class][key].append(val)

    # Build summary tables
    stats = {"all_classes": {}, "per_class": {}}

    def summarize(values, label):
        arr = np.array(values)
        return {
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "frac_zero": float((arr == 0).mean()),
            "frac_gte4": float((arr >= 4).mean()),
        }

    for key in results_all:
        stats["all_classes"][key] = summarize(results_all[key], key)

    for cls in sorted(results_by_class.keys()):
        stats["per_class"][cls] = {}
        for key in results_by_class[cls]:
            stats["per_class"][cls][key] = summarize(results_by_class[cls][key], key)

    # Log key findings
    for t in pos_thresholds:
        k = f"pos_jaccard_{t}"
        s = stats["all_classes"][k]
        LOGGER.info("Jaccard >= %.1f: median=%d positives, %.1f%% queries with 0, %.1f%% with >=4",
                     t, s["median"], s["frac_zero"] * 100, s["frac_gte4"] * 100)

    for t in overlap_thresholds:
        k = f"pos_overlap_{t}"
        s = stats["all_classes"][k]
        LOGGER.info("Overlap >= %d: median=%d positives, %.1f%% queries with 0, %.1f%% with >=4",
                     t, s["median"], s["frac_zero"] * 100, s["frac_gte4"] * 100)

    return stats


# ============================================================
# TASK 4: Sanity check — k-mer vs TF Jaccard
# ============================================================
def task4(manifest, tf_matrix, same_class_a, same_class_b, same_class_jaccards):
    LOGGER.info("=== TASK 4: k-mer vs TF Jaccard sanity check ===")
    seq_col = "sequence_512"
    sequences = manifest[seq_col].values

    # Recompute Jaccard for the full same-class pairs (aligned to tf_matrix)
    sc_jaccards = binary_jaccard_batch(tf_matrix[same_class_a], tf_matrix[same_class_b])

    # Filter to pairs where both have >= 5 TFs
    both_have_tfs = (tf_matrix[same_class_a].sum(axis=1) >= 5) & (tf_matrix[same_class_b].sum(axis=1) >= 5)
    valid_jaccards = sc_jaccards[both_have_tfs]
    valid_a = same_class_a[both_have_tfs]
    valid_b = same_class_b[both_have_tfs]

    # Top 100 by Jaccard
    top100_idx = np.argsort(valid_jaccards)[::-1][:100]
    # Bottom 100 (near 0)
    bot100_idx = np.argsort(valid_jaccards)[:100]

    selected_idx = np.concatenate([top100_idx, bot100_idx])

    kmer_sims = []
    tf_jacs = []
    labels = []
    for idx in selected_idx:
        a_idx, b_idx = valid_a[idx], valid_b[idx]
        seq_a, seq_b = str(sequences[a_idx]), str(sequences[b_idx])
        if not isinstance(seq_a, str) or not isinstance(seq_b, str) or len(seq_a) < 10 or len(seq_b) < 10:
            continue
        ks = kmer_cosine(seq_a, seq_b)
        kmer_sims.append(ks)
        tf_jacs.append(float(valid_jaccards[idx]))
        labels.append("High TF Jaccard" if idx in top100_idx else "Low TF Jaccard")

    kmer_sims = np.array(kmer_sims)
    tf_jacs = np.array(tf_jacs)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["red" if l == "High TF Jaccard" else "blue" for l in labels]
    ax.scatter(tf_jacs, kmer_sims, c=colors, alpha=0.5, s=20)
    ax.set_xlabel("TF Binding Jaccard Similarity")
    ax.set_ylabel("6-mer Cosine Similarity")
    ax.set_title("Sequence Similarity vs TF Binding Similarity")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="High TF Jaccard (top 100)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="Low TF Jaccard (bottom 100)"),
    ]
    ax.legend(handles=legend_elements)

    # Correlation
    corr = float(np.corrcoef(tf_jacs, kmer_sims)[0, 1]) if len(tf_jacs) > 2 else 0
    ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
    fig.savefig(OUT_DIR / "task4_kmer_vs_jaccard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "pearson_correlation": corr,
        "high_jaccard_mean_kmer": float(kmer_sims[np.array(labels) == "High TF Jaccard"].mean()),
        "low_jaccard_mean_kmer": float(kmer_sims[np.array(labels) == "Low TF Jaccard"].mean()),
    }
    LOGGER.info("Pearson r(kmer, TF Jaccard) = %.3f", corr)
    LOGGER.info("High TF Jaccard mean k-mer sim: %.4f", stats["high_jaccard_mean_kmer"])
    LOGGER.info("Low TF Jaccard mean k-mer sim: %.4f", stats["low_jaccard_mean_kmer"])

    # Top 20 pairs: shared TFs
    top20_shared = []
    for idx in top100_idx[:20]:
        a_idx, b_idx = valid_a[idx], valid_b[idx]
        shared = np.where(tf_matrix[a_idx] & tf_matrix[b_idx])[0]
        shared_names = [str(tf_names_global[s]) for s in shared]
        top20_shared.append({
            "jaccard": float(valid_jaccards[idx]),
            "n_shared": int(len(shared)),
            "shared_tfs": shared_names[:20],
        })
    stats["top20_pairs"] = top20_shared

    return stats


# ============================================================
# TASK 5: Class-conditioned TF profile + UMAP
# ============================================================
def task5(manifest, tf_matrix, tf_names, col_sums):
    LOGGER.info("=== TASK 5: Class-conditioned TF profile analysis ===")
    classes = manifest["ccre_class"].astype(str).values
    unique_classes = sorted(set(classes))
    n_tfs = tf_matrix.shape[1]

    # 5a: Mean TF binding vector per class
    class_means = {}
    for cls in unique_classes:
        mask = classes == cls
        class_means[cls] = tf_matrix[mask].mean(axis=0)

    # 5b: Between-class and within-class variance per TF
    overall_mean = tf_matrix.mean(axis=0)
    between_var = np.zeros(n_tfs)
    within_var = np.zeros(n_tfs)

    for cls in unique_classes:
        mask = classes == cls
        n_cls = mask.sum()
        cls_mean = class_means[cls]
        between_var += n_cls * (cls_mean - overall_mean) ** 2
        within_var += ((tf_matrix[mask].astype(float) - cls_mean[None, :]) ** 2).sum(axis=0)

    between_var /= len(tf_matrix)
    within_var /= len(tf_matrix)

    # Top 20 within-class variance TFs
    within_top20_idx = np.argsort(within_var)[::-1][:20]
    within_top20 = []
    for idx in within_top20_idx:
        within_top20.append({
            "tf_name": str(tf_names[idx]),
            "within_class_var": float(within_var[idx]),
            "between_class_var": float(between_var[idx]),
            "total_binding": int(col_sums[idx]),
        })

    between_top20_idx = np.argsort(between_var)[::-1][:20]
    between_top20 = []
    for idx in between_top20_idx:
        between_top20.append({
            "tf_name": str(tf_names[idx]),
            "between_class_var": float(between_var[idx]),
            "within_class_var": float(within_var[idx]),
            "total_binding": int(col_sums[idx]),
        })

    LOGGER.info("Top 5 within-class discriminative TFs:")
    for entry in within_top20[:5]:
        LOGGER.info("  %s: within_var=%.4f, between_var=%.4f, binding=%d",
                     entry["tf_name"], entry["within_class_var"],
                     entry["between_class_var"], entry["total_binding"])

    LOGGER.info("Top 5 between-class discriminative TFs:")
    for entry in between_top20[:5]:
        LOGGER.info("  %s: between_var=%.4f, within_var=%.4f, binding=%d",
                     entry["tf_name"], entry["between_class_var"],
                     entry["within_class_var"], entry["total_binding"])

    stats = {
        "within_class_top20": within_top20,
        "between_class_top20": between_top20,
    }

    # 5c: UMAP
    try:
        from umap import UMAP
    except ImportError:
        LOGGER.warning("umap-learn not installed, trying sklearn TSNE instead")
        from sklearn.manifold import TSNE
        LOGGER.info("Running t-SNE on 50k TF vectors (this may take a few minutes)...")
        # Subsample for speed
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(tf_matrix), size=min(10000, len(tf_matrix)), replace=False)
        sub_matrix = tf_matrix[sub_idx].astype(float)
        sub_classes = classes[sub_idx]
        embedding = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(sub_matrix)
        title = "t-SNE"
    else:
        LOGGER.info("Running UMAP on 50k TF vectors...")
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(tf_matrix), size=min(20000, len(tf_matrix)), replace=False)
        sub_matrix = tf_matrix[sub_idx].astype(float)
        sub_classes = classes[sub_idx]
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(sub_matrix)
        title = "UMAP"

    fig, ax = plt.subplots(figsize=(10, 8))
    color_map = {cls: plt.cm.tab10(i) for i, cls in enumerate(unique_classes)}
    for cls in unique_classes:
        mask = sub_classes == cls
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                  c=[color_map[cls]], label=cls, alpha=0.3, s=5)
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")
    ax.set_title(f"{title} of 50k cCRE TF Binding Vectors (colored by regulatory class)")
    ax.legend(markerscale=5)
    fig.savefig(OUT_DIR / f"task5_{title.lower()}_by_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return stats


# ============================================================
# Main
# ============================================================
tf_names_global = None  # set in main for task4 reference

def main():
    global tf_names_global

    manifest, tf_matrix, tf_names, tf_biosamples, tf_experiments = load_data(
        "data/interim/manifests/small.csv",
        "data/interim/tf_vectors/small_tf.npz",
    )
    tf_names_global = tf_names

    all_stats = {}

    # Task 1
    stats1, intermediates = task1(manifest, tf_matrix, tf_names, tf_biosamples, tf_experiments)
    all_stats["task1"] = stats1

    # Task 2
    stats2 = task2(manifest, tf_matrix, tf_names,
                   intermediates["col_sums"],
                   intermediates["same_class_a"],
                   intermediates["same_class_b"])
    all_stats["task2"] = stats2

    # Task 3
    stats3 = task3(manifest, tf_matrix,
                   intermediates["class_to_indices"],
                   intermediates["col_sums"],
                   recommended_filter_pct=None)  # will decide after seeing task2
    all_stats["task3"] = stats3

    # Task 4
    stats4 = task4(manifest, tf_matrix,
                   intermediates["same_class_a"],
                   intermediates["same_class_b"],
                   intermediates["same_class_jaccards"])
    all_stats["task4"] = stats4

    # Task 5
    stats5 = task5(manifest, tf_matrix, tf_names, intermediates["col_sums"])
    all_stats["task5"] = stats5

    # Save all stats
    with open(OUT_DIR / "tf_analysis_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    LOGGER.info("Saved all statistics to %s", OUT_DIR / "tf_analysis_stats.json")


if __name__ == "__main__":
    main()
