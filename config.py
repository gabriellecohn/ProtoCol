import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # Architecture
    num_bins: int = 200
    seq_embed_dim: int = 512
    hidden_dim: int = 256
    time_embed_dim: int = 64
    num_transformer_blocks: int = 4
    num_heads: int = 4
    cfg_drop_prob: float = 0.1

    # Diffusion schedule
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    warmup_epochs: int = 10
    max_epochs: int = 100
    batch_size: int = 32
    lr_warmup: float = 1e-3
    lr_diffusion: float = 1e-4
    weight_decay: float = 1e-5
    expr_loss_weight: float = 0.1
    max_grad_norm: float = 1.0

    # Inference
    guidance_scale: float = 3.0
    num_inference_samples: int = 50
    num_ddim_steps: int = 50

    # Genomics
    window_size: int = 500_000   # ±250kb around TSS
    bin_size: int = 2_500        # 2.5kb bins
    train_chroms: List[str] = field(default_factory=lambda: [f"chr{i}" for i in range(1, 18)])
    val_chroms: List[str] = field(default_factory=lambda: ["chr18", "chr19", "chr20"])
    test_chroms: List[str] = field(default_factory=lambda: ["chr21", "chr22"])

    # Paths
    data_dir: str = "data/"
    cache_dir: str = "cache/"
    checkpoint_dir: str = "checkpoints/"
    borzoi_model_path: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
