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
        query_tokens, query_mask = self.encode(query_sequences)
        doc_tokens, doc_mask = self.encode(doc_sequences)
        rows = []
        for q_idx in range(query_tokens.size(0)):
            row_scores = []
            for d_idx in range(doc_tokens.size(0)):
                row_scores.append(
                    self._score_pair(
                        query_tokens[q_idx],
                        query_mask[q_idx],
                        doc_tokens[d_idx],
                        doc_mask[d_idx],
                    )
                )
            rows.append(torch.stack(row_scores))
        return torch.stack(rows, dim=0)
