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
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest = manifest.dropna(subset=["ccre_id", "ccre_class", "split"]).reset_index(drop=True)
    manifest["gc_bin"] = pd.qcut(
        manifest["gc_content"].fillna(manifest["gc_content"].median()),
        q=min(args.gc_bins, max(2, manifest["gc_content"].nunique())),
        labels=False,
        duplicates="drop",
    )
    manifest["activity_list"] = manifest["activity_vector"].apply(parse_json_list)

    class_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    gc_to_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    for idx, row in manifest.iterrows():
        split = str(row["split"])
        class_to_indices[(split, str(row["ccre_class"]))].append(idx)
        gc_to_indices[(split, int(row["gc_bin"]))].append(idx)

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []
    for idx, row in manifest.iterrows():
        query_id = row["ccre_id"]
        query_class = str(row["ccre_class"])
        query_activity = row["activity_list"]
        query_gc_bin = int(row["gc_bin"])
        query_split = row["split"]

        same_class_candidates = [
            other
            for other in sample_candidates(class_to_indices[(query_split, query_class)], 512, rng)
            if other != idx
        ]
        positive_indices: list[int] = []
        hard_negative_indices: list[int] = []

        for other_idx in same_class_candidates:
            other = manifest.iloc[other_idx]
            similarity = binary_jaccard(query_activity, other["activity_list"])
            overlap = np.logical_and(
                np.asarray(query_activity) > 0,
                np.asarray(other["activity_list"]) > 0,
            ).sum() if query_activity and other["activity_list"] else 0
            if similarity >= args.positive_jaccard or overlap >= args.positive_overlap_count:
                positive_indices.append(other_idx)
            elif similarity <= 0.1:
                hard_negative_indices.append(other_idx)

        if row.get("biosample_group", "unknown") != "unknown":
            group_matches = manifest.index[
                (manifest["split"] == query_split)
                & (manifest["biosample_group"] == row["biosample_group"])
                & (manifest.index != idx)
            ].tolist()
            positive_indices.extend(sample_candidates(group_matches, 16, rng))

        positive_indices = list(dict.fromkeys(positive_indices))[:4]
        if not positive_indices:
            continue

        negative_indices: list[int] = []
        same_gc = [other for other in gc_to_indices[(query_split, query_gc_bin)] if other != idx]
        negative_indices.extend(sample_candidates(same_gc, args.negatives_per_query, rng))
        negative_indices.extend(sample_candidates(hard_negative_indices, args.negatives_per_query, rng))

        different_class = manifest.index[
            (manifest["split"] == query_split)
            & (manifest["ccre_class"] != query_class)
            & (manifest["gc_bin"] == query_gc_bin)
        ].tolist()
        negative_indices.extend(sample_candidates(different_class, args.negatives_per_query, rng))
        random_pool = manifest.index[(manifest["split"] == query_split) & (manifest.index != idx)].tolist()
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
