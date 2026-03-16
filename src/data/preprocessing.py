import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def map_cn_to_bins(cn_df: pd.DataFrame, chrom: str, bin_coords: np.ndarray,
                   cell_line: str) -> np.ndarray:
    """
    Map segmented CN data to fixed bins using piecewise constant assignment.

    cn_df: DataFrame with columns ModelID, Chromosome, Start, End, AbsoluteCN
    chrom: chromosome name (e.g. "chr1")
    bin_coords: (num_bins, 2) array of [start, end) positions
    cell_line: ModelID to look up

    Returns: (num_bins,) array of CN values. Default = 2.0 (diploid) for missing bins.
    """
    num_bins = len(bin_coords)
    cn_values = np.full(num_bins, 2.0, dtype=np.float32)

    # Filter to this cell line and chromosome
    mask = (cn_df["ModelID"] == cell_line) & (cn_df["Chromosome"] == chrom)
    segs = cn_df[mask].sort_values("Start")

    if segs.empty:
        return cn_values

    for _, seg in segs.iterrows():
        seg_start = int(seg["Start"])
        seg_end = int(seg["End"])
        cn = float(seg["AbsoluteCN"])
        # Find bins that overlap this segment
        bin_starts = bin_coords[:, 0]
        bin_ends = bin_coords[:, 1]
        overlap = (bin_starts < seg_end) & (bin_ends > seg_start)
        cn_values[overlap] = cn

    return cn_values


def get_mean_cn_at_gene(cn_df: pd.DataFrame, chrom: str, tss: int,
                        cell_lines: List[str], window: int = 10_000) -> np.ndarray:
    """
    Get mean CN value in a ±window bp region around the TSS for each cell line.
    Returns array of shape (num_cell_lines,).
    """
    cn_vals = []
    for cl in cell_lines:
        mask = (
            (cn_df["ModelID"] == cl) &
            (cn_df["Chromosome"] == chrom) &
            (cn_df["Start"] <= tss + window) &
            (cn_df["End"] >= tss - window)
        )
        segs = cn_df[mask]
        if segs.empty:
            cn_vals.append(2.0)
        else:
            # weighted average by overlap length
            weights = []
            vals = []
            for _, seg in segs.iterrows():
                overlap = min(tss + window, seg["End"]) - max(tss - window, seg["Start"])
                if overlap > 0:
                    weights.append(overlap)
                    vals.append(seg["AbsoluteCN"])
            if weights:
                cn_vals.append(np.average(vals, weights=weights))
            else:
                cn_vals.append(2.0)
    return np.array(cn_vals, dtype=np.float32)


def compute_expression_residuals(
    cn_per_gene: Dict[str, np.ndarray],
    expr_per_gene: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    For each gene, fit linear regression CN → expression across cell lines.
    Return the residuals as training targets.

    cn_per_gene: {gene_id: array of shape (num_cell_lines,)} — mean CN at gene locus
    expr_per_gene: {gene_id: array of shape (num_cell_lines,)}
    Returns: {gene_id: residual array of shape (num_cell_lines,)}
    """
    residuals = {}
    for gene in cn_per_gene:
        if gene not in expr_per_gene:
            continue
        cn = cn_per_gene[gene].reshape(-1, 1)
        expr = expr_per_gene[gene]
        valid = np.isfinite(cn.ravel()) & np.isfinite(expr)
        if valid.sum() < 3:
            residuals[gene] = np.zeros_like(expr)
            continue
        model = LinearRegression()
        model.fit(cn[valid], expr[valid])
        pred = model.predict(cn)
        res = expr - pred
        res[~valid] = 0.0
        residuals[gene] = res.astype(np.float32)
    logger.info(f"Computed residuals for {len(residuals)} genes")
    return residuals


def normalize_residuals(residuals: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Standardize residuals per gene to zero mean and unit variance."""
    normalized = {}
    for gene, res in residuals.items():
        std = res.std()
        if std > 1e-6:
            normalized[gene] = (res - res.mean()) / std
        else:
            normalized[gene] = res - res.mean()
    return normalized
