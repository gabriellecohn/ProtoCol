from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.genome import compute_midpoint
from src.utils.io import read_table, write_table
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)
BED_COLUMNS = ["chrom", "start", "end", "name", "score", "strand"]


def resolve_input_file(raw_dir: str | Path) -> Path:
    candidates: list[Path] = []
    for pattern in ("*.bed", "*.bed.gz", "*.tsv", "*.tsv.gz", "*.csv", "*.csv.gz"):
        candidates.extend(sorted(Path(raw_dir).glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"No SCREEN files found under {raw_dir}")
    return candidates[0]


def load_screen_table(path: Path) -> pd.DataFrame:
    first_line = path.open("rt", encoding="utf-8").readline()
    if first_line.startswith("chr") and len(first_line.strip().split("\t")) >= 3 and "start" not in first_line:
        df = read_table(path, header=None)
        df.columns = BED_COLUMNS[: len(df.columns)]
        return df
    return read_table(path)


def find_first_column(columns: list[str], choices: list[str]) -> str | None:
    normalized = {column.lower(): column for column in columns}
    for choice in choices:
        if choice.lower() in normalized:
            return normalized[choice.lower()]
    for column in columns:
        lower = column.lower()
        if any(choice.lower() in lower for choice in choices):
            return column
    return None


def infer_activity_columns(df: pd.DataFrame, reserved: set[str]) -> list[str]:
    candidates: list[str] = []
    for column in df.columns:
        if column in reserved:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            nonzero_rate = float((df[column].fillna(0) > 0).mean())
            if 0.0 < nonzero_rate < 1.0:
                candidates.append(column)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a SCREEN cCRE export into a normalized manifest")
    parser.add_argument("--input", default=None, help="Optional explicit SCREEN input file path")
    parser.add_argument("--raw-dir", default="data/raw/screen", help="Directory containing SCREEN source files")
    parser.add_argument(
        "--output",
        default="data/interim/manifests/screen_manifest.csv",
        help="Normalized manifest output path",
    )
    parser.add_argument(
        "--assembly",
        default="hg38",
        help="Assembly label to assign to parsed cCRE rows",
    )
    parser.add_argument(
        "--activity-columns",
        nargs="*",
        default=None,
        help="Optional explicit activity columns to encode as the activity_vector",
    )
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else resolve_input_file(args.raw_dir)
    LOGGER.info("Parsing SCREEN file %s", input_path)
    raw_df = load_screen_table(input_path)

    chrom_col = find_first_column(list(raw_df.columns), ["chrom", "#chrom", "chr"])
    start_col = find_first_column(list(raw_df.columns), ["start", "chromstart"])
    end_col = find_first_column(list(raw_df.columns), ["end", "chromend"])
    id_col = find_first_column(list(raw_df.columns), ["ccre_id", "accession", "name", "id"])
    class_col = find_first_column(list(raw_df.columns), ["ccre_class", "class", "group"])

    if not chrom_col or not start_col or not end_col:
        raise ValueError("Could not identify chromosome/start/end columns in SCREEN table")

    manifest = pd.DataFrame(
        {
            "ccre_id": raw_df[id_col].astype(str) if id_col else [f"ccre_{idx}" for idx in range(len(raw_df))],
            "chrom": raw_df[chrom_col].astype(str),
            "start": raw_df[start_col].astype(int),
            "end": raw_df[end_col].astype(int),
            "assembly": args.assembly,
            "ccre_class": raw_df[class_col].astype(str) if class_col else "unknown",
        }
    )
    manifest["midpoint"] = [compute_midpoint(start, end) for start, end in zip(manifest["start"], manifest["end"])]
    manifest["length"] = manifest["end"] - manifest["start"]

    reserved = set(manifest.columns)
    activity_columns = args.activity_columns or infer_activity_columns(raw_df, reserved)
    LOGGER.info("Using %d activity columns", len(activity_columns))

    if activity_columns:
        activity_matrix = raw_df[activity_columns].fillna(0.0).astype(float).to_numpy()
    else:
        activity_matrix = np.zeros((len(raw_df), 0), dtype=float)

    manifest["activity_vector"] = [json.dumps(row.tolist()) for row in activity_matrix]
    manifest["activity_count"] = (activity_matrix > 0).sum(axis=1) if activity_matrix.size else 0
    if activity_matrix.shape[1] > 0:
        dominant = np.argmax(activity_matrix, axis=1)
        manifest["biosample_group"] = [activity_columns[idx] for idx in dominant]
    else:
        manifest["biosample_group"] = "unknown"

    for window in (512, 1024, 2048):
        manifest[f"sequence_{window}"] = ""
    manifest["gc_content"] = np.nan
    manifest["split"] = ""

    write_table(manifest, args.output)
    LOGGER.info("Wrote normalized manifest with %d rows to %s", len(manifest), args.output)


if __name__ == "__main__":
    main()
