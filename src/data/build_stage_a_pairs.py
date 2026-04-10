from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.io import parse_json_list, write_table
from src.utils.logging import get_logger
from src.utils.metrics import binary_jaccard


LOGGER = get_logger(__name__)


def sample_candidates(indices: list[int], max_candidates: int, rng: np.random.Generator) -> list[int]:
    if len(indices) <= max_candidates:
        return indices
    chosen = rng.choice(indices, size=max_candidates, replace=False)
    return [int(idx) for idx in chosen]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weakly supervised Stage A cCRE retrieval pairs")
    parser.add_argument("--manifest", default="data/interim/manifests/screen_manifest.csv")
    parser.add_argument("--output", default="data/processed/stage_a/stage_a_pairs.csv")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--positive-jaccard", type=float, default=0.5)
    parser.add_argument("--positive-overlap-count", type=int, default=2)
    parser.add_argument("--negatives-per-query", type=int, default=15)
    parser.add_argument("--gc-bins", type=int, default=10)
    parser.add_argument("--tf-activity", default=None, help="Path to TF vectors .npz (overrides manifest activity_vector)")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest.dropna(subset=["ccre_id", "ccre_class", "split"]).reset_index(drop=True)
    gc_series = manifest["gc_content"]
    if gc_series.isna().all():
        manifest["gc_bin"] = 0
    else:
        manifest["gc_bin"] = pd.qcut(
            gc_series.fillna(gc_series.median()),
            q=min(args.gc_bins, max(2, gc_series.nunique())),
            labels=False,
            duplicates="drop",
        )
    # Load activity vectors: prefer TF vectors from .npz if provided
    if args.tf_activity:
        LOGGER.info("Loading TF activity vectors from %s", args.tf_activity)
        tf_data = np.load(args.tf_activity, allow_pickle=False)
        tf_ccre_ids = tf_data["ccre_ids"]
        tf_matrix = tf_data["tf_matrix"]
        tf_lookup = {cid: i for i, cid in enumerate(tf_ccre_ids)}
        activity_arrays = []
        matched, missed = 0, 0
        for cid in manifest["ccre_id"]:
            if cid in tf_lookup:
                activity_arrays.append(tf_matrix[tf_lookup[cid]].astype(float))
                matched += 1
            else:
                activity_arrays.append(np.zeros(tf_matrix.shape[1], dtype=float))
                missed += 1
        LOGGER.info("TF vectors: %d matched, %d missing (zero-filled), %d TF dims", matched, missed, tf_matrix.shape[1])
    else:
        activity_lists = manifest["activity_vector"].apply(parse_json_list).tolist()
        activity_arrays = [np.asarray(a, dtype=float) for a in activity_lists]

    ccre_ids = manifest["ccre_id"].tolist()
    ccre_classes = manifest["ccre_class"].astype(str).tolist()
    splits = manifest["split"].astype(str).tolist()
    gc_bins = manifest["gc_bin"].astype(int).tolist()
    biosample_groups = manifest.get("biosample_group", pd.Series(["unknown"] * len(manifest))).tolist()

    class_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    gc_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    split_to_indices: dict[str, list[int]] = defaultdict(list)
    group_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx in range(len(manifest)):
        s, c, g = splits[idx], ccre_classes[idx], gc_bins[idx]
        class_to_indices[(s, c)].append(idx)
        gc_to_indices[(s, g)].append(idx)
        split_to_indices[s].append(idx)
        bg = biosample_groups[idx]
        if bg != "unknown":
            group_to_indices[(s, bg)].append(idx)

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []
    for idx in range(len(manifest)):
        query_id = ccre_ids[idx]
        query_class = ccre_classes[idx]
        query_activity = activity_arrays[idx]
        query_gc_bin = gc_bins[idx]
        query_split = splits[idx]

        same_class_candidates = [
            other
            for other in sample_candidates(class_to_indices[(query_split, query_class)], 512, rng)
            if other != idx
        ]
        positive_indices: list[int] = []
        hard_negative_indices: list[int] = []

        has_activity = len(query_activity) > 0 and query_activity.any()
        for other_idx in same_class_candidates:
            other_activity = activity_arrays[other_idx]
            if has_activity and len(other_activity) > 0:
                q_bin = query_activity > 0
                o_bin = other_activity > 0
                union = np.logical_or(q_bin, o_bin).sum()
                intersection = np.logical_and(q_bin, o_bin).sum()
                similarity = float(intersection / union) if union > 0 else 0.0
                overlap = int(intersection)
                if similarity >= args.positive_jaccard or overlap >= args.positive_overlap_count:
                    positive_indices.append(other_idx)
                elif similarity <= 0.1:
                    hard_negative_indices.append(other_idx)
            else:
                positive_indices.append(other_idx)

        bg = biosample_groups[idx]
        if bg != "unknown":
            gm = group_to_indices.get((query_split, bg), [])
            group_matches = [g for g in gm if g != idx]
            positive_indices.extend(sample_candidates(group_matches, 16, rng))

        positive_indices = list(dict.fromkeys(positive_indices))[:4]
        if not positive_indices:
            continue

        negative_indices: list[int] = []
        same_gc = [other for other in gc_to_indices[(query_split, query_gc_bin)] if other != idx]
        negative_indices.extend(sample_candidates(same_gc, args.negatives_per_query, rng))
        negative_indices.extend(sample_candidates(hard_negative_indices, args.negatives_per_query, rng))

        diff_class_gc = [
            o for o in gc_to_indices[(query_split, query_gc_bin)]
            if ccre_classes[o] != query_class
        ]
        negative_indices.extend(sample_candidates(diff_class_gc, args.negatives_per_query, rng))
        random_pool = [o for o in split_to_indices[query_split] if o != idx]
        negative_indices.extend(sample_candidates(random_pool, args.negatives_per_query, rng))
        negative_indices = [other for other in dict.fromkeys(negative_indices) if other not in positive_indices]
        negative_indices = negative_indices[: max(args.negatives_per_query, 15)]

        for other_idx in positive_indices:
            records.append(
                {
                    "query_id": query_id,
                    "doc_id": manifest.iloc[other_idx]["ccre_id"],
                    "label": 1,
                    "pair_type": "positive",
                    "split": query_split,
                }
            )
        for other_idx in negative_indices:
            records.append(
                {
                    "query_id": query_id,
                    "doc_id": manifest.iloc[other_idx]["ccre_id"],
                    "label": 0,
                    "pair_type": "negative",
                    "split": query_split,
                }
            )

    pair_df = pd.DataFrame.from_records(records)
    write_table(pair_df, args.output)
    LOGGER.info("Wrote %d Stage A pairs to %s", len(pair_df), args.output)


if __name__ == "__main__":
    main()
