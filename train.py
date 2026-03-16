#!/usr/bin/env python3
"""
Main training script for the CN-Aware Diffusion Regulatory Model.

Usage:
    python train.py [--debug] [--data-dir DATA_DIR] [--cache-dir CACHE_DIR]
                    [--checkpoint-dir CHECKPOINT_DIR] [--device DEVICE]
                    [--warmup-epochs N] [--max-epochs N] [--batch-size N]

The --debug flag runs on 100 genes / 50 cell lines for rapid iteration.
"""

import argparse
import logging
import os
import json
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from config import Config
from src.models.denoiser import CNAwareDenoiser
from src.models.readout import ExpressionReadout
from src.training.warmup import WarmupTrainer
from src.training.diffusion_trainer import DiffusionTrainer, DiffusionSchedule
from src.data.depmap_loader import DepMapLoader
from src.data.genome_utils import load_gencode_gtf, get_window_bins, extract_sequence, one_hot_encode
from src.data.preprocessing import (
    map_cn_to_bins,
    get_mean_cn_at_gene,
    compute_expression_residuals,
    normalize_residuals,
)
from src.models.borzoi_encoder import BorzoiEncoder, RandomEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class RegulatoryDataset(Dataset):
    """
    PyTorch Dataset for the CN-aware diffusion model.
    Each item is a (gene, cell_line) pair with:
      - seq_embeddings: (num_bins, embed_dim) Borzoi features
      - cn_dosage: (num_bins,) CN per bin
      - expression: scalar expression residual
    """

    def __init__(self, items: list, seq_features: dict, cn_data: dict, residuals: dict):
        self.items = items          # list of (gene_id, cell_line) tuples
        self.seq_features = seq_features  # gene_id → (num_bins, embed_dim)
        self.cn_data = cn_data     # (gene_id, cell_line) → (num_bins,)
        self.residuals = residuals  # (gene_id, cell_line) → float

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        gene_id, cell_line = self.items[idx]
        seq_emb = torch.from_numpy(self.seq_features[gene_id])
        cn = torch.from_numpy(self.cn_data[(gene_id, cell_line)])
        expr = torch.tensor(self.residuals[(gene_id, cell_line)], dtype=torch.float32)
        return {"seq_embeddings": seq_emb, "cn_dosage": cn, "expression": expr}


def build_datasets(cfg: Config, debug: bool = False):
    """
    Build train/val/test datasets from DepMap data and gene annotations.

    In debug mode, uses only 100 genes and 50 cell lines.
    """
    logger.info("Loading DepMap data...")
    loader = DepMapLoader(cfg.data_dir)
    cn_df, expr_df, common_cell_lines = loader.load_all()

    if debug:
        common_cell_lines = common_cell_lines[:50]
        logger.info(f"Debug mode: using {len(common_cell_lines)} cell lines")

    logger.info("Loading gene annotations...")
    gtf_path = Path(cfg.data_dir) / "gencode.v41.annotation.gtf"
    if not gtf_path.exists():
        raise FileNotFoundError(
            f"Gencode GTF not found at {gtf_path}. "
            "Download from https://www.gencodegenes.org/human/release_41.html"
        )
    gene_df = load_gencode_gtf(str(gtf_path))

    if debug:
        gene_df = gene_df.head(100)
        logger.info(f"Debug mode: using {len(gene_df)} genes")

    logger.info("Setting up sequence encoder...")
    if cfg.borzoi_model_path and Path(cfg.borzoi_model_path).exists():
        encoder = BorzoiEncoder(
            model_path=cfg.borzoi_model_path,
            cache_dir=cfg.cache_dir,
            embed_dim=cfg.seq_embed_dim,
            num_bins=cfg.num_bins,
            device=cfg.device,
        )
    else:
        logger.warning("Borzoi model not found — using RandomEncoder for debugging")
        encoder = RandomEncoder(embed_dim=cfg.seq_embed_dim, num_bins=cfg.num_bins)

    fasta_path = Path(cfg.data_dir) / "hg38.fa"

    logger.info("Extracting sequence features and building CN/expression matrices...")
    seq_features = {}
    cn_gene_means = {}   # gene_id → array over cell lines
    expr_gene = {}       # gene_id → array over cell lines

    train_genes = gene_df[gene_df["chrom"].isin(cfg.train_chroms)]
    val_genes = gene_df[gene_df["chrom"].isin(cfg.val_chroms)]
    test_genes = gene_df[gene_df["chrom"].isin(cfg.test_chroms)]

    all_gene_sets = {
        "train": train_genes,
        "val": val_genes,
        "test": test_genes,
    }

    cn_data_dict = {}   # (gene_id, cell_line) → (num_bins,)
    residual_dict = {}  # (gene_id, cell_line) → float
    split_items = {"train": [], "val": [], "test": []}

    for split_name, genes in all_gene_sets.items():
        logger.info(f"Processing {len(genes)} {split_name} genes...")
        for _, gene_row in genes.iterrows():
            gene_id = gene_row["gene_id"]
            gene_name = gene_row["gene_name"]
            chrom = gene_row["chrom"]
            tss = int(gene_row["tss"])

            # Compute window bins
            bin_coords = get_window_bins(tss, cfg.window_size, cfg.bin_size)

            # Get sequence features (cached)
            if gene_id not in seq_features:
                if fasta_path.exists():
                    window_start = tss - cfg.window_size // 2
                    window_end = tss + cfg.window_size // 2
                    seq = extract_sequence(str(fasta_path), chrom, window_start, window_end)
                    seq_oh = one_hot_encode(seq)
                else:
                    # No FASTA available — use zeros (debug)
                    seq_oh = np.zeros((4, cfg.window_size), dtype=np.float32)
                seq_features[gene_id] = encoder.get_or_compute_features(gene_id, seq_oh)

            # Get expression values for this gene across cell lines
            expr_col = None
            for col in expr_df.columns:
                if gene_name in col or gene_id in col:
                    expr_col = col
                    break
            if expr_col is None:
                continue

            valid_cell_lines = [cl for cl in common_cell_lines if cl in expr_df.index]
            expr_vals = expr_df.loc[valid_cell_lines, expr_col].values.astype(np.float32)

            if len(valid_cell_lines) == 0:
                continue

            # Get mean CN at gene locus for residual computation
            gene_cn_means = get_mean_cn_at_gene(
                cn_df, chrom, tss, valid_cell_lines
            )
            cn_gene_means[gene_id] = gene_cn_means
            expr_gene[gene_id] = expr_vals

            # Map CN to bins for each cell line
            for cl_idx, cell_line in enumerate(valid_cell_lines):
                cn_bins = map_cn_to_bins(cn_df, chrom, bin_coords, cell_line)
                cn_data_dict[(gene_id, cell_line)] = cn_bins

    # Compute expression residuals
    logger.info("Computing expression residuals...")
    residuals = compute_expression_residuals(cn_gene_means, expr_gene)
    residuals = normalize_residuals(residuals)

    # Build final datasets
    for split_name, genes in all_gene_sets.items():
        for _, gene_row in genes.iterrows():
            gene_id = gene_row["gene_id"]
            if gene_id not in residuals:
                continue
            # Reconstruct the list of valid cell lines for this gene
            gene_cell_lines = [
                cl for cl in common_cell_lines
                if (gene_id, cl) in cn_data_dict
            ]
            for i, cell_line in enumerate(gene_cell_lines):
                if i < len(residuals[gene_id]):
                    residual_dict[(gene_id, cell_line)] = float(residuals[gene_id][i])
                    split_items[split_name].append((gene_id, cell_line))

    def make_dataset(items):
        return RegulatoryDataset(items, seq_features, cn_data_dict, residual_dict)

    train_ds = make_dataset(split_items["train"])
    val_ds = make_dataset(split_items["val"])
    test_ds = make_dataset(split_items["test"])

    logger.info(
        f"Dataset sizes — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}"
    )
    return train_ds, val_ds, test_ds


def save_checkpoint(cfg, denoiser, readout, optimizer, epoch, loss, tag="latest"):
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{tag}.pt"
    torch.save({
        "epoch": epoch,
        "loss": loss,
        "denoiser_state": denoiser.state_dict(),
        "readout_state": readout.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg.__dict__,
    }, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(cfg, denoiser, readout, optimizer=None, tag="latest"):
    path = Path(cfg.checkpoint_dir) / f"checkpoint_{tag}.pt"
    if not path.exists():
        return 0, float("inf")
    ckpt = torch.load(path, map_location=cfg.device)
    denoiser.load_state_dict(ckpt["denoiser_state"])
    readout.load_state_dict(ckpt["readout_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    logger.info(f"Loaded checkpoint from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["loss"]


def main():
    parser = argparse.ArgumentParser(description="Train CN-Aware Diffusion Regulatory Model")
    parser.add_argument("--debug", action="store_true",
                        help="Run on 100 genes / 50 cell lines for rapid iteration")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--cache-dir", default="cache/")
    parser.add_argument("--checkpoint-dir", default="checkpoints/")
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    args = parser.parse_args()

    cfg = Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.cache_dir:
        cfg.cache_dir = args.cache_dir
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.device:
        cfg.device = args.device
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    logger.info(f"Using device: {cfg.device}")

    # Build datasets
    train_ds, val_ds, test_ds = build_datasets(cfg, debug=args.debug)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2,
        pin_memory=(cfg.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2,
        pin_memory=(cfg.device == "cuda"),
    )

    # Build models
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
    schedule = DiffusionSchedule(
        num_timesteps=cfg.num_timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        device=cfg.device,
    )

    # ---- Stage 1: Warmup ----
    logger.info("=== Stage 1: Readout Head Warmup ===")
    warmup = WarmupTrainer(readout, device=cfg.device, lr=cfg.lr_warmup)
    warmup_losses = warmup.run(train_loader, num_epochs=cfg.warmup_epochs)
    if warmup_losses:
        logger.info(f"Warmup complete. Final loss: {warmup_losses[-1]:.4f}")
    else:
        logger.info("Warmup complete (no batches processed).")

    # ---- Stage 2: Diffusion Training ----
    logger.info("=== Stage 2: End-to-End Diffusion Training ===")
    trainer = DiffusionTrainer(
        denoiser=denoiser,
        readout=readout,
        schedule=schedule,
        device=cfg.device,
        lr=cfg.lr_diffusion,
        weight_decay=cfg.weight_decay,
        expr_loss_weight=cfg.expr_loss_weight,
        max_grad_norm=cfg.max_grad_norm,
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            cfg, denoiser, readout, trainer.optimizer
        )

    training_log = []
    for epoch in range(start_epoch, cfg.max_epochs):
        # Curriculum: gradually increase expression loss weight from 0.1 → 1.0
        progress = epoch / max(cfg.max_epochs - 1, 1)
        expr_weight = cfg.expr_loss_weight * (1 + 9 * progress)
        trainer.set_expr_loss_weight(expr_weight)

        train_metrics = trainer.run_epoch(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{cfg.max_epochs} | "
            f"total={train_metrics['total_loss']:.4f} | "
            f"diff={train_metrics['diff_loss']:.4f} | "
            f"expr={train_metrics['expr_loss']:.4f} | "
            f"expr_weight={expr_weight:.3f}"
        )

        training_log.append({"epoch": epoch + 1, **train_metrics})

        # Save checkpoints
        if train_metrics["total_loss"] < best_val_loss:
            best_val_loss = train_metrics["total_loss"]
            save_checkpoint(
                cfg, denoiser, readout, trainer.optimizer,
                epoch, best_val_loss, tag="best"
            )
        save_checkpoint(
            cfg, denoiser, readout, trainer.optimizer,
            epoch, train_metrics["total_loss"], tag="latest"
        )

    # Save training log
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    logger.info(f"Training complete. Log saved to {log_path}")


if __name__ == "__main__":
    main()
