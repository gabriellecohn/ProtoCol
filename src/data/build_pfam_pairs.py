"""Build retrieval pairs from Pfam manifest.

Positives: same family (homologs)
Hard negatives: same clan, different family (functionally related but distinct)
Easy negatives: different clan (unrelated)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Build Pfam retrieval pairs")
    parser.add_argument("--manifest", default="data/interim/manifests/pfam_manifest.csv")
    parser.add_argument("--output", default="data/processed/pfam/pfam_pairs.csv")
    parser.add_argument("--positives-per-query", type=int, default=4)
    parser.add_argument("--negatives-per-query", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna({"clan_acc": "", "clan_name": ""})
    LOGGER.info("Loaded manifest with %d sequences", len(manifest))

    seq_ids = manifest["sequence_id"].tolist()
    families = manifest["family_acc"].tolist()
    clans = manifest["clan_acc"].tolist()
    splits = manifest["split"].tolist()

    # Lookup indices
    fam_to_idx: dict[tuple[str, str], list[int]] = defaultdict(list)  # (split, family) -> indices
    clan_to_idx: dict[tuple[str, str], list[int]] = defaultdict(list)  # (split, clan) -> indices
    split_to_idx: dict[str, list[int]] = defaultdict(list)

    for idx in range(len(manifest)):
        s = splits[idx]
        fam_to_idx[(s, families[idx])].append(idx)
        if clans[idx]:
            clan_to_idx[(s, clans[idx])].append(idx)
        split_to_idx[s].append(idx)

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    skipped = 0

    for idx in range(len(manifest)):
        query_id = seq_ids[idx]
        query_fam = families[idx]
        query_clan = clans[idx]
        query_split = splits[idx]

        # Positives: same family, different sequence
        same_fam = [i for i in fam_to_idx[(query_split, query_fam)] if i != idx]
        positives = sample_up_to(same_fam, args.positives_per_query, rng)
        if not positives:
            skipped += 1
            continue

        # Hard negatives: same clan, different family
        hard_negs = []
        if query_clan:
            clan_members = clan_to_idx[(query_split, query_clan)]
            hard_negs = [i for i in clan_members if families[i] != query_fam]
            hard_negs = sample_up_to(hard_negs, args.negatives_per_query // 2, rng)

        # Easy negatives: rejection-sample from split pool
        # (probability of same-family collision is ~20/140k = negligible, but we filter anyway)
        remaining = max(args.negatives_per_query - len(hard_negs), 0)
        split_pool = split_to_idx[query_split]
        easy_negs: list[int] = []
        if remaining > 0 and split_pool:
            # Oversample 2x to give room for dedup + same-family filter
            n_sample = min(remaining * 2, len(split_pool))
            candidate_idx = rng.choice(len(split_pool), size=n_sample, replace=False)
            for ci in candidate_idx:
                i = split_pool[ci]
                if families[i] != query_fam:
                    easy_negs.append(int(i))
                    if len(easy_negs) >= remaining:
                        break

        all_negs = list(dict.fromkeys(hard_negs + easy_negs))[:args.negatives_per_query]

        for p in positives:
            records.append({
                "query_id": query_id, "doc_id": seq_ids[p],
                "label": 1, "pair_type": "positive", "split": query_split,
            })
        for n in all_negs:
            records.append({
                "query_id": query_id, "doc_id": seq_ids[n],
                "label": 0, "pair_type": "negative", "split": query_split,
            })

    pair_df = pd.DataFrame.from_records(records)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(pair_df, str(out_path))

    n_pos = (pair_df["label"] == 1).sum()
    n_neg = (pair_df["label"] == 0).sum()
    n_queries = pair_df["query_id"].nunique()
    LOGGER.info("Built %d pairs (%d pos, %d neg) for %d queries (skipped %d)",
                len(pair_df), n_pos, n_neg, n_queries, skipped)
    LOGGER.info("Split distribution:\n%s",
                pair_df.groupby("split")["label"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
