from __future__ import annotations

import torch


def multi_positive_softmax_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    scaled = scores / temperature
    scaled = scaled - scaled.max(dim=1, keepdim=True).values
    log_denom = torch.logsumexp(scaled, dim=1)
    positive_scores = scaled.masked_fill(~positive_mask, float("-inf"))
    log_positive = torch.logsumexp(positive_scores, dim=1)
    valid_rows = positive_mask.any(dim=1)
    return -(log_positive[valid_rows] - log_denom[valid_rows]).mean()
