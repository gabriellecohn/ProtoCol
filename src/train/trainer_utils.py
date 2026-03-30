from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.models.backbone import BackboneConfig, build_backbone
from src.models.colbert import ColBERTRetriever
from src.models.meanpool import MeanPoolRetriever
from src.utils.io import load_yaml


@dataclass
class QueryGroup:
    query_id: str
    query_sequence: str
    positive_ids: list[str]
    positive_sequences: list[str]
    negative_ids: list[str]
    negative_sequences: list[str]


class GroupedPairDataset(Dataset):
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        pair_df: pd.DataFrame,
        split: str,
        sequence_column: str,
        negatives_per_query: int,
        seed: int,
    ) -> None:
        self.sequence_column = sequence_column
        self.negatives_per_query = negatives_per_query
        self.seed = seed
        sequence_lookup = manifest_df.set_index("ccre_id")[sequence_column].dropna().to_dict()
        groups: list[QueryGroup] = []

        split_pairs = pair_df[pair_df["split"] == split]
        for query_id, query_rows in split_pairs.groupby("query_id"):
            if query_id not in sequence_lookup:
                continue
            positive_ids = [
                doc_id for doc_id in query_rows[query_rows["label"] == 1]["doc_id"].tolist()
                if doc_id in sequence_lookup
            ]
            negative_ids = [
                doc_id for doc_id in query_rows[query_rows["label"] == 0]["doc_id"].tolist()
                if doc_id in sequence_lookup
            ]
            if not positive_ids or not negative_ids:
                continue
            groups.append(
                QueryGroup(
                    query_id=query_id,
                    query_sequence=sequence_lookup[query_id],
                    positive_ids=positive_ids,
                    positive_sequences=[sequence_lookup[doc_id] for doc_id in positive_ids],
                    negative_ids=negative_ids,
                    negative_sequences=[sequence_lookup[doc_id] for doc_id in negative_ids],
                )
            )
        self.groups = groups

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        group = self.groups[idx]
        rng = np.random.default_rng(self.seed + idx)
        pos_idx = int(rng.integers(len(group.positive_sequences)))
        negative_indices = list(range(len(group.negative_sequences)))
        if len(negative_indices) > self.negatives_per_query:
            negative_indices = rng.choice(
                negative_indices,
                size=self.negatives_per_query,
                replace=False,
            ).tolist()
        return {
            "query_id": group.query_id,
            "query_sequence": group.query_sequence,
            "positive_id": group.positive_ids[pos_idx],
            "positive_sequence": group.positive_sequences[pos_idx],
            "negative_ids": [group.negative_ids[i] for i in negative_indices],
            "negative_sequences": [group.negative_sequences[i] for i in negative_indices],
        }


def collate_grouped_pairs(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "query_ids": [item["query_id"] for item in batch],
        "query_sequences": [item["query_sequence"] for item in batch],
        "positive_ids": [item["positive_id"] for item in batch],
        "positive_sequences": [item["positive_sequence"] for item in batch],
        "negative_ids": [item["negative_ids"] for item in batch],
        "negative_sequences": [item["negative_sequences"] for item in batch],
    }


def build_retriever(model_cfg: dict[str, Any]) -> torch.nn.Module:
    model_type = model_cfg["model_type"]
    backbone_keys = {field.name for field in fields(BackboneConfig)}
    backbone_cfg = {key: value for key, value in model_cfg.items() if key in backbone_keys}
    backbone = build_backbone(backbone_cfg)
    if model_type == "meanpool":
        return MeanPoolRetriever(backbone)
    if model_type == "colbert":
        return ColBERTRetriever(backbone, projection_dim=int(model_cfg.get("projection_dim", 64)))
    raise ValueError(f"Unsupported model_type={model_type}")


def load_training_bundle(config_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    train_cfg = load_yaml(config_path)
    model_cfg = load_yaml(train_cfg["model"]["config_path"])
    return train_cfg, model_cfg
