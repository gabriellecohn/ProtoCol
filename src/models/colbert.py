from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.backbone import DNAEncoderBackbone


class ColBERTRetriever(nn.Module):
    def __init__(self, backbone: DNAEncoderBackbone, projection_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(backbone.hidden_size, projection_dim, bias=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, attention_mask = self.backbone.forward_sequences(sequences, self.device)
        projected = F.normalize(self.projection(hidden), dim=-1)
        return projected, attention_mask

    def encode_tokens(self, tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenized inputs (skips CPU tokenization)."""
        hidden, attention_mask = self.backbone.forward_tokens(tokens, self.device)
        projected = F.normalize(self.projection(hidden), dim=-1)
        return projected, attention_mask

    @staticmethod
    def _score_batched(
        q_proj: torch.Tensor,
        q_mask: torch.Tensor,
        d_proj: torch.Tensor,
        d_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized MaxSim scoring: (Q, Tq, F) x (D, Td, F) -> (Q, D)."""
        # (Q, Tq, D, Td) — all pairwise token similarities
        sim = torch.einsum("qif,djf->qidj", q_proj, d_proj)
        # Mask out padding tokens in docs
        sim = sim.masked_fill(~d_mask[None, None, :, :], -1e4)
        # Max over doc tokens per query token, then mask query padding and sum
        max_sim = sim.max(dim=-1).values          # (Q, Tq, D)
        max_sim = max_sim * q_mask[:, :, None].float()
        return max_sim.sum(dim=1)                  # (Q, D)

    def score_token_batches(
        self,
        query_tokens: dict[str, torch.Tensor],
        doc_tokens: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score pre-tokenized query and doc batches."""
        q_proj, q_mask = self.encode_tokens(query_tokens)
        d_proj, d_mask = self.encode_tokens(doc_tokens)
        return self._score_batched(q_proj, q_mask, d_proj, d_mask)

    @staticmethod
    def _score_pair(
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        doc_tokens: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("qf,df->qd", query_tokens, doc_tokens)
        scores = scores.masked_fill(~doc_mask.unsqueeze(0), -1e4)
        max_scores = scores.max(dim=-1).values
        max_scores = max_scores * query_mask.float()
        return max_scores.sum()

    def score_sequences(self, query_sequences: list[str], doc_sequences: list[str]) -> torch.Tensor:
        q_proj, q_mask = self.encode(query_sequences)
        d_proj, d_mask = self.encode(doc_sequences)
        return self._score_batched(q_proj, q_mask, d_proj, d_mask)
