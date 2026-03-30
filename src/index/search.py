from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a saved retrieval index with a dense query embedding")
    parser.add_argument("--index", default="outputs/indexes/corpus.index")
    parser.add_argument("--query", required=True, help="Path to a .npy query embedding vector")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    query = np.load(args.query).astype("float32").reshape(1, -1)
    meta = load_json(Path(args.index).with_suffix(".json"))
    if meta["backend"] == "faiss":
        import faiss

        index = faiss.read_index(args.index)
        scores, indices = index.search(query, args.top_k)
        results = [(meta["ids"][idx], float(score)) for idx, score in zip(indices[0], scores[0])]
    else:
        matrix = np.load(Path(args.index).with_suffix(".npy")).astype("float32")
        scores = (matrix @ query.T).squeeze(-1)
        order = np.argsort(scores)[::-1][: args.top_k]
        results = [(meta["ids"][idx], float(scores[idx])) for idx in order]

    for doc_id, score in results:
        print(f"{doc_id}\t{score:.6f}")


if __name__ == "__main__":
    main()
