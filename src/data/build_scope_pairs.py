"""Build retrieval pairs from SCOPe manifest.

Positives: same superfamily (structural homologs)
Hard negatives: same fold but different superfamily (structural analogs — similar fold, no evolutionary relationship)
Easy negatives: different fold (structurally unrelated)
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from src.utils.io import write_table
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def sample_up_to(items: list[int], n: int, rng: np.random.Generator) -> list[int]:
    if len(items) <= n:
        return items
    return list(rng.choice(items, size=n, replace=False))


def main():
    parser = argparse.ArgumentParser(description="Build SCOPe retrieval pairs")
    parser.add_argument("--manifest", default="data/interim/manifests/scope_manifest.csv")
    parser.add_argument("--output", default="data/processed/scope/scope_pairs.csv")
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument("--negatives-per-query", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    LOGGER.info("Loaded manifest with %d domains", len(manifest))

    domain_ids = manifest["domain_id"].tolist()
    superfamily_ids = manifest["superfamily_id"].tolist()
    fold_ids = manifest["fold_id"].tolist()
    splits = manifest["split"].tolist()

    # Build lookup indices
    sf_to_idx: dict[tuple[str, str], list[int]] = defaultdict(list)  # (split, sf) -> indices
    fold_to_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
    split_to_idx: dict[str, list[int]] = defaultdict(list)

    for idx in range(len(manifest)):
        s = splits[idx]
        sf = superfamily_ids[idx]
        f = fold_ids[idx]
        sf_to_idx[(s, sf)].append(idx)
        fold_to_idx[(s, f)].append(idx)
        split_to_idx[s].append(idx)

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    skipped = 0

    for idx in range(len(manifest)):
        query_id = domain_ids[idx]
        query_sf = superfamily_ids[idx]
        query_fold = fold_ids[idx]
        query_split = splits[idx]

        # Positives: same superfamily (excluding self)
        same_sf = [i for i in sf_to_idx[(query_split, query_sf)] if i != idx]
        positives = sample_up_to(same_sf, args.positives_per_query, rng)

        if not positives:
            skipped += 1
            continue

        # Hard negatives: same fold, different superfamily
        same_fold = fold_to_idx[(query_split, query_fold)]
        hard_negatives = [i for i in same_fold if superfamily_ids[i] != query_sf]
        hard_negatives = sample_up_to(hard_negatives, args.negatives_per_query // 3, rng)

        # Medium negatives: different fold, same class
        query_class = query_fold.split(".")[0]
        diff_fold_same_class = [
            i for i in split_to_idx[query_split]
            if fold_ids[i] != query_fold and fold_ids[i].split(".")[0] == query_class
        ]
        medium_negatives = sample_up_to(diff_fold_same_class, args.negatives_per_query // 3, rng)

        # Easy negatives: random from different superfamily
        random_pool = [i for i in split_to_idx[query_split] if superfamily_ids[i] != query_sf]
        easy_negatives = sample_up_to(random_pool, args.negatives_per_query // 3, rng)

        all_negatives = list(dict.fromkeys(hard_negatives + medium_negatives + easy_negatives))
        all_negatives = all_negatives[:args.negatives_per_query]

        for pos_idx in positives:
            records.append({
                "query_id": query_id,
                "doc_id": domain_ids[pos_idx],
                "label": 1,
                "pair_type": "positive",
                "split": query_split,
            })
        for neg_idx in all_negatives:
            records.append({
                "query_id": query_id,
                "doc_id": domain_ids[neg_idx],
                "label": 0,
                "pair_type": "negative",
                "split": query_split,
            })

    pair_df = pd.DataFrame.from_records(records)
    out_path = args.output
    from pathlib import Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_table(pair_df, out_path)

    n_pos = (pair_df["label"] == 1).sum()
    n_neg = (pair_df["label"] == 0).sum()
    n_queries = pair_df["query_id"].nunique()
    LOGGER.info("Built %d pairs (%d pos, %d neg) for %d queries (skipped %d singletons)",
                len(pair_df), n_pos, n_neg, n_queries, skipped)
    LOGGER.info("Split distribution:\n%s", pair_df.groupby("split")["label"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()