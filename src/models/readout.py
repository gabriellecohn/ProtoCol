import torch
import torch.nn as nn


class ExpressionReadout(nn.Module):
    """
    Converts attention weights + sequence embeddings → scalar expression prediction.

    The attention weights are softmax-normalized and used to compute a weighted
    sum of Borzoi sequence features, which is then passed through an MLP.

    Critically, the attention maps are NEVER directly supervised — they emerge
    as latent structure the diffusion model discovers to explain expression.
    """

    def __init__(self, seq_embed_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(seq_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        attn_weights: torch.Tensor,    # (batch, num_bins)
        seq_embeddings: torch.Tensor,  # (batch, num_bins, seq_embed_dim)
    ) -> torch.Tensor:
        """Returns (batch, 1) expression prediction."""
        attn = torch.softmax(attn_weights, dim=-1)                        # (batch, num_bins)
        weighted = (attn.unsqueeze(-1) * seq_embeddings).sum(dim=1)       # (batch, seq_embed_dim)
        return self.head(weighted)                                         # (batch, 1)
