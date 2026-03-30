from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.genome import GenomeReference, compute_gc_content
from src.utils.io import write_table
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract centered cCRE sequences from hg38")
    parser.add_argument("--manifest", default="data/interim/manifests/screen_manifest.csv")
    parser.add_argument("--reference-fasta", default="data/raw/refs/hg38/hg38.fa")
    parser.add_argument("--window", type=int, default=1024, choices=[512, 1024, 2048])
    parser.add_argument("--output", default=None, help="Optional manifest output path")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    sequence_column = f"sequence_{args.window}"
    if sequence_column not in manifest.columns:
        manifest[sequence_column] = ""

    genome = GenomeReference(args.reference_fasta)
    sequences = []
    for row in manifest.itertuples(index=False):
        sequence = genome.fetch_centered(row.chrom, int(row.midpoint), args.window)
        sequences.append(sequence)

    manifest[sequence_column] = sequences
    if args.window == 1024:
        manifest["gc_content"] = [compute_gc_content(sequence) for sequence in sequences]

    output_path = args.output or str(manifest_path)
    write_table(manifest, output_path)
    LOGGER.info("Updated %s for %d rows", sequence_column, len(manifest))


if __name__ == "__main__":
    main()
