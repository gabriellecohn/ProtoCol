from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_similarity_heatmap(
    similarity_matrix: np.ndarray,
    output_path: str | Path,
    title: str,
    xlabel: str = "Document tokens",
    ylabel: str = "Query tokens",
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.imshow(similarity_matrix, aspect="auto", cmap="viridis")
    plt.colorbar(label="Similarity")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output
