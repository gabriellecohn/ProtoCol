from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.backbone import DNAEncoderBackbone


class MeanPoolRetriever(nn.Module):
    def __init__(self, backbone: DNAEncoderBackbone) -> None:
        super().__init__()
        self.backbone = backbone

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, sequences: list[str]) -> torch.Tensor:
        hidden, attention_mask = self.backbone.forward_sequences(sequences, self.device)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)

    def score_sequences(self, query_sequences: list[str], doc_sequences: list[str]) -> torch.Tensor:
        query_embeddings = self.encode(query_sequences)
        doc_embeddings = self.encode(doc_sequences)
        return query_embeddings @ doc_embeddings.T
