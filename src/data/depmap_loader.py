import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEPMAP_FILES = {
    "cn_segments": "OmicsCNSegmentsProfile.csv",
    "expression": "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
}


class DepMapLoader:
    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cn_data = None
        self._expr_data = None

    def load_cn_segments(self) -> pd.DataFrame:
        """
        Load OmicsCNSegmentsProfile.csv.
        Expected columns: ModelID, Chromosome, Start, End, SegmentMean (log2 CN ratio)
        Returns DataFrame with these columns plus an absolute CN column computed as:
            CN = 2 * 2^SegmentMean  (convert log2 ratio to absolute copy number)
        """
        path = self.data_dir / DEPMAP_FILES["cn_segments"]
        if not path.exists():
            raise FileNotFoundError(
                f"CN segments file not found at {path}. "
                "Please download OmicsCNSegmentsProfile.csv from https://depmap.org/portal/download/all/ "
                "(DepMap Public release, section: Copy Number)"
            )
        df = pd.read_csv(path)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        # Compute absolute CN from log2 ratio if needed
        if "SegmentMean" in df.columns and "AbsoluteCN" not in df.columns:
            df["AbsoluteCN"] = 2 * (2 ** df["SegmentMean"])
        return df

    def load_expression(self) -> pd.DataFrame:
        """
        Load OmicsExpressionProteinCodingGenesTPMLogp1.csv.
        Rows = cell lines (ModelID), Columns = gene symbols (SYMBOL (ENTREZ_ID))
        Returns a DataFrame with index=ModelID and columns=gene symbols.
        """
        path = self.data_dir / DEPMAP_FILES["expression"]
        if not path.exists():
            raise FileNotFoundError(
                f"Expression file not found at {path}. "
                "Please download OmicsExpressionProteinCodingGenesTPMLogp1.csv from https://depmap.org/portal/download/all/ "
                "(DepMap Public release, section: Omics)"
            )
        df = pd.read_csv(path, index_col=0)
        return df

    def get_common_cell_lines(self, cn_df: pd.DataFrame, expr_df: pd.DataFrame) -> list:
        """Return sorted list of cell line IDs present in both datasets."""
        cn_lines = set(cn_df["ModelID"].unique())
        expr_lines = set(expr_df.index)
        common = sorted(cn_lines & expr_lines)
        logger.info(f"Found {len(common)} cell lines with both CN and expression data")
        return common

    def load_all(self):
        """Load both datasets and return (cn_df, expr_df, common_cell_lines)."""
        cn_df = self.load_cn_segments()
        expr_df = self.load_expression()
        common = self.get_common_cell_lines(cn_df, expr_df)
        return cn_df, expr_df, common
