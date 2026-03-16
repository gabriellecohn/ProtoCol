import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_gencode_gtf(gtf_path: str) -> pd.DataFrame:
    """
    Parse a Gencode GTF file and return a DataFrame of protein-coding gene TSS positions.
    Columns: gene_id, gene_name, chrom, tss, strand
    Only includes genes on chr1-chr22 (autosomes).
    """
    records = []
    autosomes = {f"chr{i}" for i in range(1, 23)}

    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start, end, _, strand, _, attrs = fields
            if feature != "gene":
                continue
            if chrom not in autosomes:
                continue
            # Parse attributes
            attr_dict = {}
            for attr in attrs.strip().split(";"):
                attr = attr.strip()
                if not attr:
                    continue
                parts = attr.split(" ", 1)
                if len(parts) == 2:
                    attr_dict[parts[0]] = parts[1].strip('"')
            gene_type = attr_dict.get("gene_type", "")
            if gene_type != "protein_coding":
                continue
            gene_id = attr_dict.get("gene_id", "").split(".")[0]
            gene_name = attr_dict.get("gene_name", gene_id)
            start = int(start) - 1  # GTF is 1-based
            end = int(end)
            tss = start if strand == "+" else end - 1
            records.append({
                "gene_id": gene_id,
                "gene_name": gene_name,
                "chrom": chrom,
                "tss": tss,
                "strand": strand,
            })

    df = pd.DataFrame(records).drop_duplicates(subset=["gene_id"])
    logger.info(f"Loaded {len(df)} protein-coding genes from {gtf_path}")
    return df


def get_window_bins(tss: int, window_size: int = 500_000, bin_size: int = 2500) -> np.ndarray:
    """
    Return (num_bins, 2) array of [start, end) positions for bins centered on TSS.
    num_bins = window_size // bin_size = 200
    """
    half = window_size // 2
    window_start = tss - half
    num_bins = window_size // bin_size
    starts = window_start + np.arange(num_bins) * bin_size
    ends = starts + bin_size
    return np.stack([starts, ends], axis=1)


def extract_sequence(fasta_path: str, chrom: str, start: int, end: int) -> str:
    """
    Extract DNA sequence from a FASTA file using pyfaidx.
    Pads with N if the window extends beyond chromosome boundaries.
    """
    try:
        from pyfaidx import Fasta
        fasta = Fasta(fasta_path)
        chrom_len = len(fasta[chrom])
        start_clip = max(0, start)
        end_clip = min(chrom_len, end)
        seq = str(fasta[chrom][start_clip:end_clip])
        # Pad if necessary
        left_pad = "N" * max(0, -start)
        right_pad = "N" * max(0, end - chrom_len)
        return left_pad + seq + right_pad
    except ImportError:
        raise ImportError("pyfaidx is required for sequence extraction. pip install pyfaidx")
    except Exception as e:
        logger.warning(f"Failed to extract sequence for {chrom}:{start}-{end}: {e}")
        return "N" * (end - start)


def one_hot_encode(sequence: str) -> np.ndarray:
    """
    One-hot encode a DNA sequence.
    Returns array of shape (4, seq_len) with channels [A, C, G, T].
    N's are encoded as all zeros.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq_len = len(sequence)
    encoding = np.zeros((4, seq_len), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        idx = mapping.get(base)
        if idx is not None:
            encoding[idx, i] = 1.0
    return encoding
