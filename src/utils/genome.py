from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


try:
    from pyfaidx import Fasta
except ImportError:  # pragma: no cover
    Fasta = None


def compute_midpoint(start: int, end: int) -> int:
    return int((int(start) + int(end)) // 2)


def compute_gc_content(sequence: str) -> float:
    seq = sequence.upper()
    informative = [base for base in seq if base in {"A", "C", "G", "T"}]
    if not informative:
        return 0.0
    gc = sum(base in {"G", "C"} for base in informative)
    return gc / len(informative)


def pad_sequence(sequence: str, target_length: int, pad_char: str = "N") -> str:
    if len(sequence) >= target_length:
        return sequence[:target_length]
    return sequence + (pad_char * (target_length - len(sequence)))


@dataclass
class GenomeReference:
    fasta_path: str | Path

    def __post_init__(self) -> None:
        if Fasta is None:
            raise ImportError("pyfaidx is required for FASTA-backed sequence extraction")
        self.fasta = Fasta(str(self.fasta_path), rebuild=False)

    def fetch(self, chrom: str, start: int, end: int) -> str:
        start = max(int(start), 0)
        end = max(int(end), start)
        if chrom not in self.fasta:
            return "N" * max(end - start, 0)
        return self.fasta[chrom][start:end].seq.upper()

    def fetch_centered(self, chrom: str, midpoint: int, window: int) -> str:
        half = window // 2
        start = max(midpoint - half, 0)
        end = start + window
        sequence = self.fetch(chrom, start, end)
        return pad_sequence(sequence, window)

