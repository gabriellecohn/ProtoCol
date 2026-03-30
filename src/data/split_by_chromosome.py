from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import dump_json, load_yaml, write_table
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def assign_split(chrom: str, val_chroms: set[str], test_chroms: set[str]) -> str:
    if chrom in test_chroms:
        return "test"
    if chrom in val_chroms:
        return "val"
    return "train"


def sample_subset(df: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if len(df) <= size:
        return df.copy()
    fractions = df["split"].value_counts(normalize=True).to_dict()
    sampled = []
    remaining = size
    for split, fraction in fractions.items():
        split_df = df[df["split"] == split]
        split_size = min(len(split_df), max(1, int(round(size * fraction))))
        remaining -= split_size
        sampled.append(split_df.sample(n=split_size, random_state=seed))
    result = pd.concat(sampled, ignore_index=True).drop_duplicates(subset=["ccre_id"])
    if len(result) < size:
        extra = df[~df["ccre_id"].isin(result["ccre_id"])]
        if not extra.empty:
            fill = extra.sample(n=min(size - len(result), len(extra)), random_state=seed)
            result = pd.concat([result, fill], ignore_index=True)
    return result.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign chromosome-held-out splits and create subset manifests")
    parser.add_argument("--manifest", default="data/interim/manifests/screen_manifest.csv")
    parser.add_argument("--config", default="configs/data/screen.yaml")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    manifest = pd.read_csv(args.manifest)
    val_chroms = set(cfg["chrom_splits"]["val"])
    test_chroms = set(cfg["chrom_splits"]["test"])
    manifest["split"] = [
        assign_split(chrom, val_chroms=val_chroms, test_chroms=test_chroms)
        for chrom in manifest["chrom"].astype(str)
    ]

    write_table(manifest, args.manifest)
    manifest_dir = Path(args.manifest).parent
    subset_sizes = cfg["subset_sizes"]
    subset_outputs: dict[str, str] = {}
    for subset_name, subset_size in subset_sizes.items():
        subset = sample_subset(manifest, int(subset_size), seed=args.seed)
        subset_path = manifest_dir / f"{subset_name}.csv"
        write_table(subset, subset_path)
        subset_outputs[subset_name] = str(subset_path)
        LOGGER.info("Wrote %s with %d rows", subset_name, len(subset))

    qc_path = manifest_dir.parent / "qc" / "split_summary.json"
    split_counts = manifest["split"].value_counts().to_dict()
    dump_json({"split_counts": split_counts, "subset_outputs": subset_outputs}, qc_path)
    LOGGER.info("Updated split assignments in %s", args.manifest)


if __name__ == "__main__":
    main()
