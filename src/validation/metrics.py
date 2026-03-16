import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    """Pearson correlation coefficient, handling edge cases."""
    if len(pred) < 2:
        return 0.0
    r, _ = stats.pearsonr(pred, target)
    return float(r) if np.isfinite(r) else 0.0


def evaluate_expression_prediction(
    denoiser: nn.Module,
    readout: nn.Module,
    dataloader,
    schedule,
    num_inference_samples: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate expression prediction accuracy by:
    1. Sampling attention maps unconditionally
    2. Predicting expression via readout head
    3. Comparing to actual expression residuals

    Returns mean Pearson r across genes in the dataloader.
    """
    from src.inference.sampling import sample_attention_maps

    denoiser.eval()
    readout.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            seq_emb = batch["seq_embeddings"].to(device)
            cn = batch["cn_dosage"].to(device)
            expr = batch["expression"].to(device)
            if expr.dim() == 1:
                expr = expr.unsqueeze(-1)

            batch_size = seq_emb.shape[0]
            for i in range(batch_size):
                maps = sample_attention_maps(
                    denoiser, seq_emb[i:i+1], cn[i:i+1],
                    betas=schedule.betas, alphas=schedule.alphas,
                    alpha_cumprod=schedule.alpha_cumprod,
                    num_samples=num_inference_samples,
                    num_timesteps=schedule.num_timesteps,
                    device=device,
                )
                # Average prediction over samples
                seq_expand = seq_emb[i:i+1].expand(num_inference_samples, -1, -1)
                expr_pred = readout(maps, seq_expand).mean().item()
                all_preds.append(expr_pred)
                all_targets.append(expr[i].item())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    r = pearson_r(all_preds, all_targets)
    mse = float(np.mean((all_preds - all_targets) ** 2))

    return {"pearson_r": r, "mse": mse}


def linear_baseline_r(cn_vals: np.ndarray, expr_vals: np.ndarray) -> float:
    """Pearson r of a simple linear CN→expression model (null baseline)."""
    from sklearn.linear_model import LinearRegression
    if len(cn_vals) < 3:
        return 0.0
    model = LinearRegression().fit(cn_vals.reshape(-1, 1), expr_vals)
    pred = model.predict(cn_vals.reshape(-1, 1))
    return pearson_r(pred, expr_vals)


def compute_attention_entropy(attn_maps: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of attention distributions.
    Lower entropy = more focused attention = clearer regulatory target.
    attn_maps: (num_samples, num_bins)
    Returns: (num_samples,) entropy values
    """
    probs = torch.softmax(attn_maps, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    return -(probs * log_probs).sum(dim=-1)
