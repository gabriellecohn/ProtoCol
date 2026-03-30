from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(doc_id in relevant_ids for doc_id in top_k)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(rank + 1.0)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def binary_jaccard(left: Iterable[float], right: Iterable[float], threshold: float = 0.0) -> float:
    left_arr = np.asarray(list(left)) > threshold
    right_arr = np.asarray(list(right)) > threshold
    union = np.logical_or(left_arr, right_arr).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(left_arr, right_arr).sum()
    return float(intersection / union)


def safe_pearson(left: Iterable[float], right: Iterable[float]) -> float:
    left_arr = np.asarray(list(left), dtype=float)
    right_arr = np.asarray(list(right), dtype=float)
    if left_arr.size == 0 or right_arr.size == 0:
        return 0.0
    if np.std(left_arr) == 0 or np.std(right_arr) == 0:
        return 0.0
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def top_k_class_purity(retrieved_ids: list[str], label_lookup: dict[str, str], query_label: str, k: int) -> float:
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    matches = sum(label_lookup.get(doc_id) == query_label for doc_id in top_k)
    return matches / len(top_k)

