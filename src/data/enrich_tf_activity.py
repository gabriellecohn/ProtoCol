"""Enrich cCRE manifests with TF binding activity vectors from ENCODE TF peaks matrix.

The TF peaks matrix (ENCFF257UKO) uses rDHS accessions (EH38D...) while our
manifests use cCRE accessions (EH38E...). The GRCh38-cCREs.bed file provides
the mapping between the two ID namespaces.

Output: a .npz file containing:
  - ccre_ids: (N,) array of cCRE accession strings
  - tf_matrix: (N, T) binary uint8 matrix of TF binding
  - tf_names: (T,) array of TF target names
  - tf_biosamples: (T,) array of biosample names
  - tf_experiments: (T,) array of experiment accessions
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_rdhs_to_ccre_map(bed_path: str | Path) -> dict[str, str]:
    """Build rDHS -> cCRE ID mapping from the GRCh38-cCREs.bed file."""
    df = pd.read_csv(
        bed_path, sep="\t", header=None, usecols=[3, 4],
        names=["rdhs_id", "ccre_id"], dtype=str,
    )
    return dict(zip(df["rdhs_id"], df["ccre_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich manifest with TF binding vectors")
    parser.add_argument(
        "--tf-matrix", default="data/raw/screen/ENCFF257UKO.txt.gz",
        help="Path to ENCODE TF peaks matrix (gzipped TSV)",
    )
    parser.add_argument(
        "--bed", default="data/raw/screen/GRCh38-cCREs.bed",
        help="GRCh38-cCREs.bed with rDHS-to-cCRE mapping",
    )
    parser.add_argument(
        "--manifest", default="data/interim/manifests/small.csv",
        help="Manifest CSV to filter cCREs",
    )
    parser.add_argument(
        "--output", default="data/interim/tf_vectors/small_tf.npz",
        help="Output .npz file with TF activity matrix",
    )
    args = parser.parse_args()

    # Step 1: Build rDHS -> cCRE mapping
    LOGGER.info("Loading rDHS-to-cCRE mapping from %s", args.bed)
    rdhs_to_ccre = load_rdhs_to_ccre_map(args.bed)
    LOGGER.info("Loaded %d rDHS-to-cCRE mappings", len(rdhs_to_ccre))

    # Step 2: Get target cCRE IDs from manifest
    manifest = pd.read_csv(args.manifest, usecols=["ccre_id"])
    target_ccres = set(manifest["ccre_id"].astype(str))
    LOGGER.info("Manifest has %d cCREs", len(target_ccres))

    # Build reverse map: cCRE -> rDHS for fast lookup
    ccre_to_rdhs: dict[str, str] = {}
    for rdhs_id, ccre_id in rdhs_to_ccre.items():
        if ccre_id in target_ccres:
            ccre_to_rdhs[ccre_id] = rdhs_id
    LOGGER.info("Matched %d / %d cCREs to rDHS IDs", len(ccre_to_rdhs), len(target_ccres))

    target_rdhs = set(ccre_to_rdhs.values())

    # Step 3: Parse TF matrix headers (first 3 rows)
    LOGGER.info("Reading TF peaks matrix headers from %s", args.tf_matrix)
    header_df = pd.read_csv(args.tf_matrix, sep="\t", nrows=3, header=None, dtype=str)
    tf_experiments = header_df.iloc[0, 1:].values  # row 0: experiment accessions
    tf_biosamples = header_df.iloc[1, 1:].values   # row 1: biosample names
    tf_names = header_df.iloc[2, 1:].values         # row 2: TF target names
    n_tfs = len(tf_names)
    LOGGER.info("TF matrix has %d TF experiments", n_tfs)

    # Step 4: Stream through the matrix line-by-line (low memory)
    LOGGER.info("Streaming TF matrix (filtering to %d target rDHS)...", len(target_rdhs))
    matched_ccre_ids: list[str] = []
    matched_rows: list[np.ndarray] = []

    import gzip
    tf_path = str(args.tf_matrix)
    # Prefer decompressed version if available
    if tf_path.endswith(".gz"):
        plain_path = tf_path[:-3]
        from pathlib import Path as _P
        if _P(plain_path).exists():
            tf_path = plain_path
    opener = gzip.open if tf_path.endswith(".gz") else open
    with opener(tf_path, "rt") as fh:
        # Skip 3 header rows
        for _ in range(3):
            next(fh)
        rows_read = 0
        for line in fh:
            rows_read += 1
            parts = line.rstrip("\n").split("\t")
            rdhs_id = parts[0]
            if rdhs_id in target_rdhs:
                ccre_id = rdhs_to_ccre[rdhs_id]
                matched_ccre_ids.append(ccre_id)
                matched_rows.append(np.array([int(x) for x in parts[1:]], dtype=np.uint8))
                if len(matched_ccre_ids) == len(target_rdhs):
                    break  # Found all targets
            if rows_read % 500_000 == 0:
                LOGGER.info("  processed %d / ~2.35M rows, matched %d", rows_read, len(matched_ccre_ids))

    LOGGER.info("Matched %d cCREs with TF vectors", len(matched_ccre_ids))

    # Step 5: Compute summary stats
    tf_matrix = np.stack(matched_rows) if matched_rows else np.zeros((0, n_tfs), dtype=np.uint8)
    tfs_per_ccre = tf_matrix.sum(axis=1)
    ccres_per_tf = tf_matrix.sum(axis=0)
    LOGGER.info(
        "TF binding density: %.1f TFs/cCRE (median), %.1f cCREs/TF (median)",
        float(np.median(tfs_per_ccre)) if len(tfs_per_ccre) else 0,
        float(np.median(ccres_per_tf)) if len(ccres_per_tf) else 0,
    )

    # Step 6: Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ccre_ids=np.array(matched_ccre_ids, dtype=str),
        tf_matrix=tf_matrix,
        tf_names=tf_names,
        tf_biosamples=tf_biosamples,
        tf_experiments=tf_experiments,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    LOGGER.info("Saved TF vectors to %s (%.1f MB)", out_path, size_mb)


if __name__ == "__main__":
    main()
