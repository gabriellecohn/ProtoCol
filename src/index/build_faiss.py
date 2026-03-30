from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.io import dump_json, load_json
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index or numpy fallback from saved corpus embeddings")
    parser.add_argument("--embeddings", default="outputs/indexes/corpus.npy")
    parser.add_argument("--ids", default="outputs/indexes/corpus.ids.json")
    parser.add_argument("--output", default="outputs/indexes/corpus.index")
    args = parser.parse_args()

    matrix = np.load(args.embeddings)
    metadata = load_json(args.ids)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import faiss

        matrix = matrix.astype("float32")
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(output_path))
        dump_json({"backend": "faiss", "ids": metadata["ids"]}, output_path.with_suffix(".json"))
    except ImportError:
        np.save(output_path.with_suffix(".npy"), matrix)
        dump_json({"backend": "numpy", "ids": metadata["ids"]}, output_path.with_suffix(".json"))

    LOGGER.info("Index artifact written to %s", output_path)


if __name__ == "__main__":
    main()
