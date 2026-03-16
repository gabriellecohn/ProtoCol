#!/usr/bin/env python3
"""
Inference script for the CN-Aware Diffusion Regulatory Model.

Usage:
    # Unconditional sampling (what regulatory configs exist for this CN?)
    python infer.py --gene BRCA1 --cell-line ACH-000001 --mode unconditional

    # Expression-guided discovery (what configs explain high expression?)
    python infer.py --gene MYC --cell-line ACH-000001 --mode guided --target-expr 2.5

    # Counterfactual analysis (how does amplification rewire regulation?)
    python infer.py --gene EGFR --mode counterfactual --cn-amplified 8

    # DDIM fast sampling
    python infer.py --gene TP53 --cell-line ACH-000001 --mode unconditional --fast
"""

import argparse
import logging
import numpy as np
import torch
import json
from pathlib import Path

from config import Config
from src.models.denoiser import CNAwareDenoiser
from src.models.readout import ExpressionReadout
from src.training.diffusion_trainer import DiffusionSchedule
from src.inference.sampling import (
    sample_attention_maps,
    ddim_sample_attention_maps,
    counterfactual_analysis,
    expression_guided_discovery,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_models(cfg: Config, checkpoint_tag: str = "best"):
    denoiser = CNAwareDenoiser(
        num_bins=cfg.num_bins,
        seq_embed_dim=cfg.seq_embed_dim,
        hidden_dim=cfg.hidden_dim,
        time_embed_dim=cfg.time_embed_dim,
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_heads=cfg.num_heads,
        cfg_drop_prob=cfg.cfg_drop_prob,
    )
    readout = ExpressionReadout(seq_embed_dim=cfg.seq_embed_dim)

    ckpt_path = Path(cfg.checkpoint_dir) / f"checkpoint_{checkpoint_tag}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        denoiser.load_state_dict(ckpt["denoiser_state"])
        readout.load_state_dict(ckpt["readout_state"])
        logger.info(f"Loaded checkpoint from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_path} — using random weights")

    denoiser = denoiser.to(cfg.device)
    readout = readout.to(cfg.device)
    denoiser.eval()
    readout.eval()
    return denoiser, readout


def load_seq_features(cfg: Config, gene_name: str) -> torch.Tensor:
    """
    Load cached Borzoi sequence features for a gene.
    Falls back to random features if not found.
    """
    import h5py
    cache_dir = Path(cfg.cache_dir)
    features_path = cache_dir / "borzoi_features.h5"

    if features_path.exists():
        with h5py.File(features_path, "r") as f:
            # Try exact match first, then substring
            if gene_name in f:
                seq_emb = torch.from_numpy(f[gene_name][:]).float()
                logger.info(f"Loaded sequence features for {gene_name}")
                return seq_emb
            for key in f.keys():
                if gene_name in key:
                    seq_emb = torch.from_numpy(f[key][:]).float()
                    logger.info(f"Loaded sequence features for {key} (matched {gene_name})")
                    return seq_emb

    logger.warning(f"Gene {gene_name} not in cache — using random features")
    return torch.randn(cfg.num_bins, cfg.seq_embed_dim)


def main():
    parser = argparse.ArgumentParser(description="CN-Aware Diffusion Model Inference")
    parser.add_argument("--gene", required=True,
                        help="Gene name or ID (e.g. BRCA1, MYC, EGFR)")
    parser.add_argument("--cell-line", default=None,
                        help="Cell line ModelID (e.g. ACH-000001)")
    parser.add_argument("--mode",
                        choices=["unconditional", "guided", "counterfactual"],
                        default="unconditional",
                        help="Inference mode")
    parser.add_argument("--target-expr", type=float, default=2.0,
                        help="Target expression residual for guided sampling")
    parser.add_argument("--guidance-scale", type=float, default=3.0,
                        help="CFG guidance scale (>1 = stronger guidance)")
    parser.add_argument("--cn-amplified", type=float, default=8.0,
                        help="Amplified CN level for counterfactual analysis")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to draw from the diffusion model")
    parser.add_argument("--fast", action="store_true",
                        help="Use DDIM (50-step) instead of full DDPM (1000-step) sampling")
    parser.add_argument("--checkpoint", default="best",
                        help="Checkpoint tag to load (default: best)")
    parser.add_argument("--output", default="results/",
                        help="Directory to write output JSON files")
    parser.add_argument("--device", default=None,
                        help="Override device (cpu/cuda)")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device

    Path(args.output).mkdir(parents=True, exist_ok=True)

    denoiser, readout = load_models(cfg, args.checkpoint)
    schedule = DiffusionSchedule(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=cfg.device,
    )

    seq_emb = load_seq_features(cfg, args.gene)

    # Default: diploid CN (2 copies per bin) vs amplified
    cn_diploid = torch.full((cfg.num_bins,), 2.0, dtype=torch.float32)
    cn_amplified = torch.full((cfg.num_bins,), args.cn_amplified, dtype=torch.float32)

    result: dict

    if args.mode == "unconditional":
        logger.info(f"Unconditional sampling for gene={args.gene} ...")
        if args.fast:
            maps = ddim_sample_attention_maps(
                denoiser, seq_emb, cn_diploid,
                alpha_cumprod=schedule.alpha_cumprod,
                num_samples=args.num_samples,
                num_inference_steps=cfg.num_ddim_steps,
                device=cfg.device,
            )
        else:
            maps = sample_attention_maps(
                denoiser, seq_emb, cn_diploid,
                betas=schedule.betas, alphas=schedule.alphas,
                alpha_cumprod=schedule.alpha_cumprod,
                num_samples=args.num_samples,
                num_timesteps=cfg.num_timesteps,
                device=cfg.device,
            )
        result = {
            "gene": args.gene,
            "mode": "unconditional",
            "fast": args.fast,
            "num_samples": args.num_samples,
            "maps_mean": maps.mean(0).cpu().numpy().tolist(),
            "maps_std": maps.std(0).cpu().numpy().tolist(),
        }

    elif args.mode == "guided":
        logger.info(
            f"Expression-guided sampling for gene={args.gene}, "
            f"target_expr={args.target_expr}, guidance_scale={args.guidance_scale} ..."
        )
        raw = expression_guided_discovery(
            denoiser, readout, seq_emb, cn_diploid,
            target_expression=args.target_expr,
            betas=schedule.betas, alphas=schedule.alphas,
            alpha_cumprod=schedule.alpha_cumprod,
            guidance_scale=args.guidance_scale,
            num_samples=args.num_samples,
            device=cfg.device,
        )
        result = {
            "gene": args.gene,
            "mode": "guided",
            "target_expr": args.target_expr,
            "guidance_scale": args.guidance_scale,
            "num_samples": args.num_samples,
            "attention_difference": raw["attention_difference"].tolist(),
            "guided_maps_mean": raw["guided_maps"].mean(0).tolist(),
            "unguided_maps_mean": raw["unguided_maps"].mean(0).tolist(),
        }

    elif args.mode == "counterfactual":
        logger.info(
            f"Counterfactual analysis for gene={args.gene}: "
            f"CN=2 (diploid) vs CN={args.cn_amplified} (amplified) ..."
        )
        raw = counterfactual_analysis(
            denoiser, readout, seq_emb,
            cn_original=cn_diploid,
            cn_modified=cn_amplified,
            betas=schedule.betas, alphas=schedule.alphas,
            alpha_cumprod=schedule.alpha_cumprod,
            num_samples=args.num_samples,
            device=cfg.device,
        )
        result = {
            "gene": args.gene,
            "mode": "counterfactual",
            "cn_diploid": 2.0,
            "cn_amplified": args.cn_amplified,
            "num_samples": args.num_samples,
            "attention_shift": raw["attention_shift"].tolist(),
            "expression_change": raw["expression_change"],
            "expression_variance_original": raw["expression_variance_original"],
            "expression_variance_modified": raw["expression_variance_modified"],
        }
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    out_path = Path(args.output) / f"{args.gene}_{args.mode}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
