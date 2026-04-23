"""Per-query subsampler for Pfam pair files.

The original ``pfam_pairs_sub.csv`` was produced by uniform row sampling, which
destroyed per-query structure — many queries ended up with 1–2 negatives. This
script instead subsamples *queries*, keeping all positives plus up to
``--max-negatives-per-query`` negatives for each kept query, preserving the
retrieval structure the loss expects.

Usage:
    python -m scripts.subsample_pfam_pairs \\
        --input data/processed/pfam/pfam_pairs.csv \\
        --output data/processed/pfam/pfam_pairs_sub_v2.csv \\
        --queries-per-split 84000 22000 22000 \\
        --max-negatives-per-query 15 \\
        --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def subsample(
    df: pd.DataFrame,
    queries_per_split: dict[str, int],
    max_negs_per_query: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kept = []
    for split, group in df.groupby("split"):
        query_ids = group["query_id"].unique()
        target = queries_per_split.get(split, len(query_ids))
        target = min(target, len(query_ids))
        chosen = rng.choice(query_ids, size=target, replace=False)
        sub = group[group["query_id"].isin(chosen)]

        # For each query: keep all positives, cap negatives
        per_query_parts = []
        for qid, qrows in sub.groupby("query_id"):
            pos = qrows[qrows["label"] == 1]
            neg = qrows[qrows["label"] == 0]
            if len(neg) > max_negs_per_query:
                neg = neg.sample(n=max_negs_per_query, random_state=rng.integers(2**31))
            per_query_parts.append(pd.concat([pos, neg], ignore_index=True))
        if per_query_parts:
            kept.append(pd.concat(per_query_parts, ignore_index=True))
    return pd.concat(kept, ignore_index=True) if kept else df.iloc[:0]


def _report(df: pd.DataFrame) -> None:
    for split, group in df.groupby("split"):
        n_queries = group["query_id"].nunique()
        n_pos = (group["label"] == 1).sum()
        n_neg = (group["label"] == 0).sum()
        negs_per_q = (
            group[group["label"] == 0].groupby("query_id").size().describe()
            if n_neg else None
        )
        print(f"[{split}] queries={n_queries} pos={n_pos} neg={n_neg}")
        if negs_per_q is not None:
            print(
                f"    negatives/query: min={int(negs_per_q['min'])} "
                f"median={int(negs_per_q['50%'])} max={int(negs_per_q['max'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/pfam/pfam_pairs.csv")
    parser.add_argument("--output", default="data/processed/pfam/pfam_pairs_sub_v2.csv")
    parser.add_argument(
        "--queries-per-split",
        nargs=3,
        type=int,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[84000, 22000, 22000],
    )
    parser.add_argument("--max-negatives-per-query", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    q_per_split = dict(zip(["train", "val", "test"], args.queries_per_split))

    print(f"Input: {args.input}")
    _report(df)

    out = subsample(df, q_per_split, args.max_negatives_per_query, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nOutput: {args.output}")
    _report(out)


if __name__ == "__main__":
    main()
