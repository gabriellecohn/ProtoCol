from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.backbone import DNAEncoderBackbone


class MeanPoolRetriever(nn.Module):
    def __init__(self, backbone: DNAEncoderBackbone, projection_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = nn.Linear(backbone.hidden_size, projection_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, sequences: list[str]) -> torch.Tensor:
        hidden, attention_mask = self.backbone.forward_sequences(sequences, self.device)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)

    def encode_tokens(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode pre-tokenized inputs into mean-pooled projected embeddings."""
        hidden, mask = self.backbone.forward_tokens(tokens, self.device)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)

    def score_sequences(self, query_sequences: list[str], doc_sequences: list[str]) -> torch.Tensor:
        query_embeddings = self.encode(query_sequences)
        doc_embeddings = self.encode(doc_sequences)
        return query_embeddings @ doc_embeddings.T

    def score_token_batches(
        self, query_tokens: dict[str, torch.Tensor], doc_tokens: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Score pre-tokenized queries against pre-tokenized docs."""
        q_emb = self.encode_tokens(query_tokens)
        d_emb = self.encode_tokens(doc_tokens)
        return q_emb @ d_emb.T
