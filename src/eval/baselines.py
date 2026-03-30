from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def random_scores(num_docs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(num_docs)


def kmer_counts(sequence: str, k: int) -> Counter[str]:
    sequence = sequence.upper()
    if len(sequence) < k:
        return Counter()
    return Counter(sequence[idx : idx + k] for idx in range(len(sequence) - k + 1))


def cosine_from_counters(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in intersection)
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def kmer_similarity_scores(query_sequence: str, doc_sequences: Iterable[str], k: int) -> np.ndarray:
    query_counter = kmer_counts(query_sequence, k)
    return np.asarray(
        [cosine_from_counters(query_counter, kmer_counts(doc_sequence, k)) for doc_sequence in doc_sequences],
        dtype=float,
    )
