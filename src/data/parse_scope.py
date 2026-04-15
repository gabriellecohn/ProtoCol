"""Parse SCOPe ASTRAL FASTA + classification into a protein retrieval manifest.

Outputs a CSV with columns:
  domain_id, pdb_id, chain, sccs, class_id, fold_id, superfamily_id, family_id,
  sequence, seq_length, split
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def parse_fasta(fasta_path: str) -> dict[str, str]:
    """Parse FASTA file into {domain_id: sequence} dict."""
    sequences: dict[str, str] = {}
    current_id = None
    current_seq: list[str] = []
    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]  # e.g., "d1ux8a_"
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(current_seq)
    return sequences


def parse_classification(cla_path: str) -> pd.DataFrame:
    """Parse SCOPe dir.cla file into a DataFrame."""
    records = []
    with open(cla_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            domain_id = parts[0]
            pdb_id = parts[1]
            chain = parts[2]
            sccs = parts[3]  # e.g., "a.1.1.1"
            # Parse SCCS into hierarchy
            sccs_parts = sccs.split(".")
            if len(sccs_parts) < 4:
                continue
            # Parse hierarchy IDs from the last column
            hier = {}
            for kv in parts[5].split(","):
                k, v = kv.split("=")
                hier[k] = int(v)
            records.append({
                "domain_id": domain_id,
                "pdb_id": pdb_id,
                "chain": chain,
                "sccs": sccs,
                "class_name": sccs_parts[0],
                "fold_id": f"{sccs_parts[0]}.{sccs_parts[1]}",
                "superfamily_id": f"{sccs_parts[0]}.{sccs_parts[1]}.{sccs_parts[2]}",
                "family_id": sccs,
                "cl": hier.get("cl", 0),
                "cf": hier.get("cf", 0),
                "sf": hier.get("sf", 0),
                "fa": hier.get("fa", 0),
            })
    return pd.DataFrame(records)


def assign_splits(df: pd.DataFrame, test_folds: list[str], val_folds: list[str]) -> pd.Series:
    """Assign train/val/test splits by held-out folds."""
    splits = pd.Series("train", index=df.index)
    splits[df["fold_id"].isin(test_folds)] = "test"
    splits[df["fold_id"].isin(val_folds)] = "val"
    return splits


def main():
    parser = argparse.ArgumentParser(description="Parse SCOPe into protein retrieval manifest")
    parser.add_argument("--fasta", default="data/raw/scope/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa")
    parser.add_argument("--classification", default="data/raw/scope/dir.cla.scope.2.08-stable.txt")
    parser.add_argument("--output", default="data/interim/manifests/scope_manifest.csv")
    parser.add_argument("--max-length", type=int, default=1022, help="Max sequence length (ESM-2 limit)")
    parser.add_argument("--min-length", type=int, default=30, help="Min sequence length")
    args = parser.parse_args()

    # Parse sequences and classification
    LOGGER.info("Parsing FASTA from %s", args.fasta)
    sequences = parse_fasta(args.fasta)
    LOGGER.info("Parsed %d sequences", len(sequences))

    LOGGER.info("Parsing classification from %s", args.classification)
    cla_df = parse_classification(args.classification)
    LOGGER.info("Parsed %d classification entries", len(cla_df))

    # Join sequences with classification
    cla_df = cla_df[cla_df["domain_id"].isin(sequences)]
    cla_df["sequence"] = cla_df["domain_id"].map(sequences)
    cla_df["seq_length"] = cla_df["sequence"].str.len()

    # Filter by length
    before = len(cla_df)
    cla_df = cla_df[(cla_df["seq_length"] >= args.min_length) & (cla_df["seq_length"] <= args.max_length)]
    LOGGER.info("Filtered %d → %d domains (length %d–%d)",
                before, len(cla_df), args.min_length, args.max_length)

    # Report hierarchy stats
    n_classes = cla_df["class_name"].nunique()
    n_folds = cla_df["fold_id"].nunique()
    n_superfamilies = cla_df["superfamily_id"].nunique()
    n_families = cla_df["family_id"].nunique()
    LOGGER.info("Hierarchy: %d classes, %d folds, %d superfamilies, %d families",
                n_classes, n_folds, n_superfamilies, n_families)

    # Assign splits: hold out ~10% of folds for test, ~10% for val
    fold_counts = cla_df["fold_id"].value_counts()
    rng = np.random.default_rng(42)
    all_folds = fold_counts.index.tolist()
    rng.shuffle(all_folds)

    # Accumulate folds until we reach ~15% of domains for test, ~10% for val
    total = len(cla_df)
    test_folds, val_folds = [], []
    test_count, val_count = 0, 0
    for fold in all_folds:
        count = fold_counts[fold]
        if test_count < total * 0.15:
            test_folds.append(fold)
            test_count += count
        elif val_count < total * 0.10:
            val_folds.append(fold)
            val_count += count

    cla_df["split"] = assign_splits(cla_df, test_folds, val_folds)
    split_counts = cla_df["split"].value_counts()
    LOGGER.info("Splits: %s", dict(split_counts))
    LOGGER.info("Test folds (%d): %s...", len(test_folds), test_folds[:5])
    LOGGER.info("Val folds (%d): %s...", len(val_folds), val_folds[:5])

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cla_df.to_csv(out_path, index=False)
    LOGGER.info("Saved manifest with %d domains to %s", len(cla_df), out_path)


if __name__ == "__main__":
    main()
