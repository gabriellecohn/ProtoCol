import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


@torch.no_grad()
def sample_attention_maps(
    denoiser: nn.Module,
    seq_embeddings: torch.Tensor,    # (1, num_bins, embed_dim) or (num_bins, embed_dim)
    cn_dosage: torch.Tensor,         # (1, num_bins) or (num_bins,)
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    expression_guide: Optional[torch.Tensor] = None,  # (1, 1)
    guidance_scale: float = 1.0,
    num_samples: int = 50,
    num_timesteps: int = 1000,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample regulatory attention maps using DDPM reverse process.

    Modes:
    - Unconditional (expression_guide=None): samples from p(attn | DNA, CN)
    - Guided (expression_guide + guidance_scale > 1): CFG-steered sampling

    Returns: (num_samples, num_bins) tensor of sampled attention maps.
    """
    denoiser.eval()

    # Ensure correct shapes
    if seq_embeddings.dim() == 2:
        seq_embeddings = seq_embeddings.unsqueeze(0)  # (1, num_bins, embed_dim)
    if cn_dosage.dim() == 1:
        cn_dosage = cn_dosage.unsqueeze(0)  # (1, num_bins)

    seq_embeddings = seq_embeddings.to(device)
    cn_dosage = cn_dosage.to(device)
    if expression_guide is not None:
        expression_guide = expression_guide.to(device)

    num_bins = seq_embeddings.shape[1]
    samples = []

    for _ in range(num_samples):
        x = torch.randn(1, num_bins, device=device)

        for t_idx in reversed(range(num_timesteps)):
            t_tensor = torch.tensor([float(t_idx)], device=device)

            if expression_guide is not None and guidance_scale != 1.0:
                # Classifier-free guidance
                noise_uncond = denoiser(x, t_tensor, seq_embeddings, cn_dosage, expression=None)
                noise_cond = denoiser(x, t_tensor, seq_embeddings, cn_dosage,
                                      expression=expression_guide)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = denoiser(x, t_tensor, seq_embeddings, cn_dosage, expression=None)

            # DDPM reverse step
            beta_t = betas[t_idx]
            alpha_t = alphas[t_idx]
            alpha_cumprod_t = alpha_cumprod[t_idx]

            x = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * noise_pred
            )
            if t_idx > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)

        samples.append(x)

    return torch.cat(samples, dim=0)  # (num_samples, num_bins)


@torch.no_grad()
def ddim_sample_attention_maps(
    denoiser: nn.Module,
    seq_embeddings: torch.Tensor,
    cn_dosage: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    expression_guide: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.0,
    num_samples: int = 50,
    num_inference_steps: int = 50,
    device: str = "cpu",
) -> torch.Tensor:
    """
    DDIM sampling (~20x faster than DDPM). Drop-in replacement for sample_attention_maps.
    Uses deterministic sampling with ~50 steps instead of 1000.
    """
    denoiser.eval()

    if seq_embeddings.dim() == 2:
        seq_embeddings = seq_embeddings.unsqueeze(0)
    if cn_dosage.dim() == 1:
        cn_dosage = cn_dosage.unsqueeze(0)

    seq_embeddings = seq_embeddings.to(device)
    cn_dosage = cn_dosage.to(device)
    if expression_guide is not None:
        expression_guide = expression_guide.to(device)

    num_timesteps = len(alpha_cumprod)
    timesteps = torch.linspace(num_timesteps - 1, 0, num_inference_steps, dtype=torch.long)
    num_bins = seq_embeddings.shape[1]
    samples = []

    for _ in range(num_samples):
        x = torch.randn(1, num_bins, device=device)

        for i, t_idx in enumerate(timesteps):
            t_idx = t_idx.item()
            t_tensor = torch.tensor([float(t_idx)], device=device)

            if expression_guide is not None and guidance_scale != 1.0:
                noise_uncond = denoiser(x, t_tensor, seq_embeddings, cn_dosage, expression=None)
                noise_cond = denoiser(x, t_tensor, seq_embeddings, cn_dosage,
                                      expression=expression_guide)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = denoiser(x, t_tensor, seq_embeddings, cn_dosage, expression=None)

            alpha_t = alpha_cumprod[t_idx]
            alpha_prev = (
                alpha_cumprod[timesteps[i + 1].item()]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0, device=device)
            )

            # DDIM update
            x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            x0_pred = x0_pred.clamp(-3, 3)
            x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * noise_pred

        samples.append(x)

    return torch.cat(samples, dim=0)


def counterfactual_analysis(
    denoiser: nn.Module,
    readout: nn.Module,
    seq_embeddings: torch.Tensor,
    cn_original: torch.Tensor,
    cn_modified: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    num_samples: int = 50,
    device: str = "cpu",
) -> Dict:
    """
    Compare regulatory attention maps between two CN profiles.
    Returns a dict with attention_shift, expression_change, and raw maps.
    """
    maps_original = sample_attention_maps(
        denoiser, seq_embeddings, cn_original,
        betas=betas, alphas=alphas, alpha_cumprod=alpha_cumprod,
        num_samples=num_samples, num_timesteps=len(betas), device=device,
    )
    maps_modified = sample_attention_maps(
        denoiser, seq_embeddings, cn_modified,
        betas=betas, alphas=alphas, alpha_cumprod=alpha_cumprod,
        num_samples=num_samples, num_timesteps=len(betas), device=device,
    )

    if seq_embeddings.dim() == 2:
        seq_embeddings_expanded = seq_embeddings.unsqueeze(0).expand(num_samples, -1, -1).to(device)
    else:
        seq_embeddings_expanded = seq_embeddings.expand(num_samples, -1, -1).to(device)

    with torch.no_grad():
        expr_original = readout(maps_original.to(device), seq_embeddings_expanded)
        expr_modified = readout(maps_modified.to(device), seq_embeddings_expanded)

    return {
        "attention_shift": (maps_modified.mean(0) - maps_original.mean(0)).cpu(),
        "expression_change": (expr_modified.mean() - expr_original.mean()).item(),
        "expression_variance_original": expr_original.var().item(),
        "expression_variance_modified": expr_modified.var().item(),
        "maps_original": maps_original.cpu(),
        "maps_modified": maps_modified.cpu(),
    }


def expression_guided_discovery(
    denoiser: nn.Module,
    readout: nn.Module,
    seq_embeddings: torch.Tensor,
    cn_dosage: torch.Tensor,
    target_expression: float,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_cumprod: torch.Tensor,
    guidance_scale: float = 3.0,
    num_samples: int = 50,
    device: str = "cpu",
) -> Dict:
    """
    Discover regulatory configurations that explain a specific expression level.
    Returns guided vs unguided attention maps and their difference.
    """
    expr_tensor = torch.tensor([[target_expression]], dtype=torch.float32, device=device)

    guided_maps = sample_attention_maps(
        denoiser, seq_embeddings, cn_dosage,
        betas=betas, alphas=alphas, alpha_cumprod=alpha_cumprod,
        expression_guide=expr_tensor, guidance_scale=guidance_scale,
        num_samples=num_samples, num_timesteps=len(betas), device=device,
    )
    unguided_maps = sample_attention_maps(
        denoiser, seq_embeddings, cn_dosage,
        betas=betas, alphas=alphas, alpha_cumprod=alpha_cumprod,
        num_samples=num_samples, num_timesteps=len(betas), device=device,
    )

    return {
        "guided_maps": guided_maps.cpu(),
        "unguided_maps": unguided_maps.cpu(),
        "attention_difference": (guided_maps.mean(0) - unguided_maps.mean(0)).cpu(),
    }
