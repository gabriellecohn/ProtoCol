import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class WarmupTrainer:
    """
    Warms up the ExpressionReadout head before end-to-end diffusion training.

    Feeds random attention maps through the readout head and trains it to predict
    expression. This gets the head's weights into a reasonable regime so that
    gradients flowing back through attention maps will be meaningful when the
    diffusion model starts producing structured maps.

    Note: prediction accuracy won't be great — that's expected. The goal is
    parameter initialization, not task completion.
    """

    def __init__(self, readout: nn.Module, device: str = "cpu", lr: float = 1e-3):
        self.readout = readout.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(readout.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(
        self,
        seq_embeddings: torch.Tensor,  # (batch, num_bins, embed_dim)
        expr_target: torch.Tensor,     # (batch, 1)
    ) -> float:
        seq_embeddings = seq_embeddings.to(self.device)
        expr_target = expr_target.to(self.device)
        batch_size, num_bins, _ = seq_embeddings.shape

        # Random attention maps — readout learns what a "useful" weighting looks like
        random_attn = torch.randn(batch_size, num_bins, device=self.device)

        expr_pred = self.readout(random_attn, seq_embeddings)
        loss = self.loss_fn(expr_pred, expr_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self, dataloader, num_epochs: int = 10) -> list:
        """Run warmup training for num_epochs. Returns list of per-step losses."""
        self.readout.train()
        losses = []
        for epoch in range(num_epochs):
            epoch_losses = []
            for batch in dataloader:
                seq_emb = batch["seq_embeddings"]
                expr = batch["expression"]
                if expr.dim() == 1:
                    expr = expr.unsqueeze(-1)
                loss = self.train_step(seq_emb, expr)
                epoch_losses.append(loss)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            losses.extend(epoch_losses)
            logger.info(f"Warmup epoch {epoch+1}/{num_epochs} | loss={mean_loss:.4f}")
        return losses
