import torch
import torch.nn as nn
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DiffusionSchedule:
    """Linear beta schedule for DDPM."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.device = device

    def to(self, device: str):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        self.device = device
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None):
        """
        Forward diffusion: q(x_t | x_0).

        x0: (batch, num_bins)
        t:  (batch,) long integer timestep indices
        Returns: (noisy_x, noise) both of shape (batch, num_bins)

        alpha_cumprod[t] has shape (batch,); we unsqueeze to (batch, 1) for
        broadcasting against the (batch, num_bins) tensors.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_t = self.alpha_cumprod[t].unsqueeze(-1)  # (batch, 1)
        noisy = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return noisy, noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor,
                   noise_pred: torch.Tensor) -> torch.Tensor:
        """
        Estimate x_0 from noisy x_t and predicted noise (one-step denoising).

        x_t, noise_pred: (batch, num_bins)
        t: (batch,) long integer timestep indices
        """
        alpha_t = self.alpha_cumprod[t].unsqueeze(-1)  # (batch, 1)
        return (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / (torch.sqrt(alpha_t) + 1e-8)


class DiffusionTrainer:
    """
    End-to-end trainer for the CN-aware diffusion regulatory model.

    Jointly trains the CNAwareDenoiser and ExpressionReadout:
    - Diffusion loss: MSE between predicted and actual noise (standard DDPM objective)
    - Expression loss: MSE between readout prediction and target expression residual

    CFG dropout is handled internally by the denoiser during training.
    """

    def __init__(self, denoiser: nn.Module, readout: nn.Module,
                 schedule: DiffusionSchedule, device: str = "cpu",
                 lr: float = 1e-4, weight_decay: float = 1e-5,
                 expr_loss_weight: float = 0.1, max_grad_norm: float = 1.0):
        self.denoiser = denoiser.to(device)
        self.readout = readout.to(device)
        self.schedule = schedule.to(device)
        self.device = device
        self.expr_loss_weight = expr_loss_weight
        self.max_grad_norm = max_grad_norm
        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            list(denoiser.parameters()) + list(readout.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
        self.all_params = list(denoiser.parameters()) + list(readout.parameters())

    def train_step(
        self,
        seq_embeddings: torch.Tensor,  # (batch, num_bins, embed_dim)
        cn_dosage: torch.Tensor,       # (batch, num_bins)
        expr_target: torch.Tensor,     # (batch, 1)
    ) -> Dict[str, float]:
        self.denoiser.train()
        self.readout.train()

        seq_embeddings = seq_embeddings.to(self.device)
        cn_dosage = cn_dosage.to(self.device)
        expr_target = expr_target.to(self.device)

        batch_size = seq_embeddings.shape[0]
        num_bins = seq_embeddings.shape[1]

        # Sample random "clean" attention maps as x_0 (prior: N(0,1))
        attn_clean = torch.randn(batch_size, num_bins, device=self.device)

        # Sample random timesteps (long integers for indexing alpha_cumprod)
        t = torch.randint(0, self.schedule.num_timesteps, (batch_size,),
                          device=self.device)

        # Forward diffusion
        noisy_attn, noise = self.schedule.q_sample(attn_clean, t)

        # Predict noise conditioned on expression (CFG dropout handled internally)
        # Pass t as float for sinusoidal embeddings
        noise_pred = self.denoiser(
            noisy_attn, t.float(), seq_embeddings, cn_dosage, expression=expr_target
        )
        diff_loss = self.loss_fn(noise_pred, noise)

        # One-step denoising to get differentiable clean map estimate
        pred_clean = self.schedule.predict_x0(noisy_attn, t, noise_pred)

        # Expression loss through readout head
        expr_pred = self.readout(pred_clean, seq_embeddings)
        expr_loss = self.loss_fn(expr_pred, expr_target)

        loss = diff_loss + self.expr_loss_weight * expr_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.all_params, self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_loss": loss.item(),
            "diff_loss": diff_loss.item(),
            "expr_loss": expr_loss.item(),
        }

    def run_epoch(self, dataloader) -> Dict[str, float]:
        """Run one full training epoch. Returns mean losses."""
        totals = {"total_loss": 0.0, "diff_loss": 0.0, "expr_loss": 0.0}
        n = 0
        for batch in dataloader:
            seq_emb = batch["seq_embeddings"]
            cn = batch["cn_dosage"]
            expr = batch["expression"]
            if expr.dim() == 1:
                expr = expr.unsqueeze(-1)
            metrics = self.train_step(seq_emb, cn, expr)
            for k in totals:
                totals[k] += metrics[k]
            n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    def set_expr_loss_weight(self, weight: float):
        """Allows gradually increasing the expression loss weight during training."""
        self.expr_loss_weight = weight
