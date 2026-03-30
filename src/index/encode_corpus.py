from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.train.trainer_utils import build_retriever
from src.utils.io import dump_json
from src.utils.logging import get_logger


LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a cCRE corpus into dense embeddings")
    parser.add_argument("--manifest", default="data/interim/manifests/screen_manifest.csv")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sequence-column", default="sequence_1024")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-prefix", default="outputs/indexes/corpus")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = build_retriever(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    manifest = pd.read_csv(args.manifest)
    subset = manifest[manifest["split"] == args.split].dropna(subset=[args.sequence_column]).copy()
    sequences = subset[args.sequence_column].tolist()
    ids = subset["ccre_id"].tolist()

    embeddings = []
    with torch.no_grad():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            if hasattr(model, "encode"):
                batch_embeddings = model.encode(batch)
                if isinstance(batch_embeddings, tuple):
                    batch_embeddings = batch_embeddings[0].mean(dim=1)
            else:
                batch_embeddings = model.score_sequences(batch, batch)
            embeddings.append(batch_embeddings.detach().cpu().numpy())

    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, 1), dtype=float)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_prefix.with_suffix(".npy"), matrix)
    dump_json({"ids": ids}, output_prefix.with_suffix(".ids.json"))
    LOGGER.info("Saved corpus embeddings for %d sequences", len(ids))


if __name__ == "__main__":
    main()
