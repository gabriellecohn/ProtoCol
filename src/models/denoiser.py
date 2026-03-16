import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed)[0]
        x = x + self.ff(self.norm2(x))
        return x


class CNAwareDenoiser(nn.Module):
    """
    Denoises a regulatory attention weight vector (num_bins dims) conditioned on
    Borzoi sequence embeddings, CN dosage, and (optionally) expression value.

    Expression conditioning uses classifier-free guidance (CFG):
    - During training, expression is randomly zeroed with probability cfg_drop_prob,
      so the model learns both the conditional and unconditional distributions.
    - During inference, pass expression=None for unconditional sampling,
      or pass expression + use CFG interpolation for guided sampling.
    """

    def __init__(
        self,
        num_bins: int = 200,
        seq_embed_dim: int = 512,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        num_transformer_blocks: int = 4,
        num_heads: int = 4,
        cfg_drop_prob: float = 0.1,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.cfg_drop_prob = cfg_drop_prob

        # Sinusoidal time embedding → hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Expression scalar → hidden_dim
        self.expr_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-bin condition: concat(seq_embed, CN) → hidden_dim
        self.condition_encoder = nn.Sequential(
            nn.Linear(seq_embed_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection: noisy attn weights (scalar per bin) → hidden_dim
        self.input_proj = nn.Linear(1, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_transformer_blocks)
        ])

        # Output projection: hidden_dim → predicted noise (scalar per bin)
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        noisy_attn: torch.Tensor,       # (batch, num_bins)
        t: torch.Tensor,                 # (batch,) float timestep
        seq_embeddings: torch.Tensor,    # (batch, num_bins, seq_embed_dim)
        cn_dosage: torch.Tensor,         # (batch, num_bins)
        expression: Optional[torch.Tensor] = None,  # (batch, 1) or None
    ) -> torch.Tensor:
        batch_size = noisy_attn.shape[0]

        # Time embedding: (batch, hidden_dim)
        t_emb = self.time_mlp(t)

        # Expression embedding with CFG dropout
        if expression is not None and self.training:
            drop_mask = (torch.rand(batch_size, 1, device=expression.device) < self.cfg_drop_prob)
            expr_input = expression * (~drop_mask).float()
            expr_emb = self.expr_mlp(expr_input)
        elif expression is not None:
            expr_emb = self.expr_mlp(expression)
        else:
            # Unconditional: zero expression embedding
            expr_emb = torch.zeros(batch_size, self.expr_mlp[-1].out_features,
                                   device=noisy_attn.device)

        # Per-bin condition from sequence embeddings and CN
        cn_expanded = cn_dosage.unsqueeze(-1)                               # (batch, num_bins, 1)
        condition_input = torch.cat([seq_embeddings, cn_expanded], dim=-1)  # (batch, num_bins, seq_embed_dim+1)
        condition = self.condition_encoder(condition_input)                  # (batch, num_bins, hidden_dim)

        # Project noisy attention weights
        x = self.input_proj(noisy_attn.unsqueeze(-1))  # (batch, num_bins, hidden_dim)

        # Combine all conditioning signals
        x = x + condition + t_emb.unsqueeze(1) + expr_emb.unsqueeze(1)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Predict noise
        noise_pred = self.output_proj(x).squeeze(-1)  # (batch, num_bins)
        return noise_pred
