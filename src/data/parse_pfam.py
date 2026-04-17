"""Parse Pfam-A seed alignment (Stockholm format) into a protein retrieval manifest.

Outputs CSV with: sequence_id, family_acc, family_name, clan_acc, clan_name,
                  family_type, sequence, seq_length, split
"""
from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_clans(clans_path: str) -> dict[str, tuple[str, str]]:
    """Load family -> (clan_acc, clan_name) mapping from Pfam-A.clans.tsv."""
    family_to_clan: dict[str, tuple[str, str]] = {}
    opener = gzip.open if str(clans_path).endswith(".gz") else open
    with opener(clans_path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            family_acc = parts[0]
            clan_acc = parts[1] if parts[1] else ""
            clan_name = parts[2] if len(parts) > 2 and parts[2] else ""
            if clan_acc:
                family_to_clan[family_acc] = (clan_acc, clan_name)
    return family_to_clan


def parse_stockholm(seed_path: str, family_to_clan: dict[str, tuple[str, str]], max_per_family: int = 50):
    """Parse Pfam-A.seed Stockholm format. Yields dicts per sequence."""
    opener = gzip.open if str(seed_path).endswith(".gz") else open
    current: dict = {}
    sequences: list[tuple[str, str]] = []  # (id, aligned_seq)

    def _reset():
        return {"family_acc": None, "family_name": None, "family_type": None}

    current = _reset()

    with opener(seed_path, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# STOCKHOLM"):
                continue
            if line.startswith("#=GF AC"):
                # Accession like "PF00001.23" — strip version
                current["family_acc"] = line.split(None, 2)[2].split(".")[0]
            elif line.startswith("#=GF ID"):
                current["family_name"] = line.split(None, 2)[2]
            elif line.startswith("#=GF TP"):
                current["family_type"] = line.split(None, 2)[2]
            elif line.startswith("#") or not line.strip():
                continue
            elif line.startswith("//"):
                # End of record — emit sequences
                fam = current.get("family_acc")
                if fam:
                    clan_acc, clan_name = family_to_clan.get(fam, ("", ""))
                    # Subsample to cap per-family sequence count
                    if len(sequences) > max_per_family:
                        sequences = sequences[:max_per_family]
                    for seq_id, aligned in sequences:
                        # Strip gaps (- and .) and uppercase
                        seq = aligned.replace("-", "").replace(".", "").upper()
                        yield {
                            "sequence_id": f"{fam}/{seq_id}",
                            "family_acc": fam,
                            "family_name": current.get("family_name", ""),
                            "family_type": current.get("family_type", ""),
                            "clan_acc": clan_acc,
                            "clan_name": clan_name,
                            "sequence": seq,
                            "seq_length": len(seq),
                        }
                current = _reset()
                sequences = []
            else:
                # Sequence line: "name/range   ALIGNED_SEQUENCE"
                parts = line.split(None, 1)
                if len(parts) == 2:
                    seq_id, aligned = parts
                    sequences.append((seq_id, aligned))


def main():
    parser = argparse.ArgumentParser(description="Parse Pfam-A seed into retrieval manifest")
    parser.add_argument("--seed", default="data/raw/pfam/Pfam-A.seed.gz")
    parser.add_argument("--clans", default="data/raw/pfam/Pfam-A.clans.tsv.gz")
    parser.add_argument("--output", default="data/interim/manifests/pfam_manifest.csv")
    parser.add_argument("--max-per-family", type=int, default=30, help="Cap sequences per family")
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-length", type=int, default=1022)
    parser.add_argument("--require-clan", action="store_true",
                        help="Keep only families assigned to a clan (enables hard negatives)")
    args = parser.parse_args()

    LOGGER.info("Loading clan mapping from %s", args.clans)
    family_to_clan = load_clans(args.clans)
    LOGGER.info("Loaded %d family-to-clan mappings (%d unique clans)",
                len(family_to_clan), len({c[0] for c in family_to_clan.values()}))

    LOGGER.info("Parsing Pfam seed from %s (max %d seqs per family)", args.seed, args.max_per_family)
    records = list(parse_stockholm(args.seed, family_to_clan, args.max_per_family))
    LOGGER.info("Parsed %d total sequences", len(records))

    df = pd.DataFrame.from_records(records)

    # Length filter
    before = len(df)
    df = df[(df["seq_length"] >= args.min_length) & (df["seq_length"] <= args.max_length)]
    LOGGER.info("Filtered by length: %d -> %d", before, len(df))

    if args.require_clan:
        before = len(df)
        df = df[df["clan_acc"] != ""]
        LOGGER.info("Filtered to clan-assigned families: %d -> %d", before, len(df))

    # Stats
    n_families = df["family_acc"].nunique()
    n_clans = df["clan_acc"].nunique()
    LOGGER.info("%d families, %d clans", n_families, n_clans)
    LOGGER.info("Family-type counts:\n%s", df["family_type"].value_counts().to_string())

    # Clan-held-out splits (to prevent leakage between train/val/test)
    # Assign entire clans to test/val so families within a clan don't leak across splits
    rng = np.random.default_rng(42)
    all_clans = df["clan_acc"].dropna().unique().tolist()
    rng.shuffle(all_clans)

    # Also handle unclan'd families: assign by family ID hash
    total = len(df)
    clan_counts = df.groupby("clan_acc").size()

    test_clans, val_clans = [], []
    test_count, val_count = 0, 0
    for clan in all_clans:
        if clan == "":
            continue
        count = clan_counts[clan]
        if test_count < total * 0.15:
            test_clans.append(clan)
            test_count += count
        elif val_count < total * 0.10:
            val_clans.append(clan)
            val_count += count

    # Assign splits
    def assign_split(row):
        if row["clan_acc"] in test_clans:
            return "test"
        if row["clan_acc"] in val_clans:
            return "val"
        # Unclan'd families: split by family accession hash for reproducibility
        if row["clan_acc"] == "":
            h = hash(row["family_acc"]) % 100
            if h < 15:
                return "test"
            if h < 25:
                return "val"
        return "train"

    df["split"] = df.apply(assign_split, axis=1)
    LOGGER.info("Split counts: %s", df["split"].value_counts().to_dict())
    LOGGER.info("Test clans: %d, Val clans: %d", len(test_clans), len(val_clans))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    LOGGER.info("Saved manifest with %d sequences to %s", len(df), out_path)


if __name__ == "__main__":
    main()
