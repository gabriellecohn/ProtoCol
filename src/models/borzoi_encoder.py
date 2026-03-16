import torch
import torch.nn as nn
import numpy as np
import h5py
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class BorzoiEncoder:
    """
    Wrapper around pretrained Borzoi for extracting intermediate sequence features.
    Features are cached to disk as they only depend on reference genome sequence.

    Since Borzoi requires a specific installation (https://github.com/calico/borzoi),
    this class gracefully falls back to a RandomEncoder for testing/debugging
    when Borzoi is not available.
    """

    def __init__(self, model_path: Optional[str] = None, cache_dir: str = "cache/",
                 embed_dim: int = 512, num_bins: int = 200, device: str = "cpu"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.device = device
        self.model = None
        self._hooks = []

        if model_path is not None:
            self._load_borzoi(model_path)

    def _load_borzoi(self, model_path: str):
        """Load pretrained Borzoi model and freeze all parameters."""
        try:
            # Try to import borzoi
            import sys
            sys.path.insert(0, str(Path(model_path).parent))
            # Borzoi uses keras/tensorflow or pytorch depending on version
            # This is a placeholder — adjust based on actual Borzoi API
            logger.info(f"Loading Borzoi from {model_path}")
            # TODO: Replace with actual Borzoi loading code once installed
            # from borzoi_pytorch import BorzoiModel
            # self.model = BorzoiModel.from_pretrained(model_path)
            # self.model.eval()
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # self.model = self.model.to(self.device)
            logger.warning("Borzoi loading is a placeholder — using RandomEncoder fallback")
        except Exception as e:
            logger.warning(f"Could not load Borzoi: {e}. Using RandomEncoder.")

    def _register_hook(self, layer):
        """Register a forward hook to capture intermediate features."""
        features = []

        def hook_fn(module, input, output):
            features.append(output.detach())

        handle = layer.register_forward_hook(hook_fn)
        return features, handle

    @torch.no_grad()
    def extract_features(self, sequence_one_hot: np.ndarray) -> np.ndarray:
        """
        Extract intermediate Borzoi features for a one-hot encoded sequence.

        sequence_one_hot: (4, seq_len) numpy array
        Returns: (num_bins, embed_dim) numpy array

        If Borzoi is not loaded, returns random features for debugging.
        """
        if self.model is None:
            # Fallback: random features for debugging pipeline
            logger.debug("Using random features (Borzoi not loaded)")
            return np.random.randn(self.num_bins, self.embed_dim).astype(np.float32)

        x = torch.from_numpy(sequence_one_hot).unsqueeze(0).to(self.device)  # (1, 4, seq_len)

        features = []
        # Hook into the last convolutional block before the expression head
        # Adjust layer name based on actual Borzoi architecture
        target_layer = self.model.conv_tower[-1]
        features_list, handle = self._register_hook(target_layer)

        _ = self.model(x)
        handle.remove()

        raw = features_list[0].squeeze(0)  # (embed_dim, num_raw_bins) or (num_raw_bins, embed_dim)
        # Permute if needed and interpolate to num_bins
        if raw.dim() == 2 and raw.shape[0] == self.embed_dim:
            raw = raw.T  # → (num_raw_bins, embed_dim)

        # Interpolate to fixed num_bins
        raw = raw.unsqueeze(0).permute(0, 2, 1)  # (1, embed_dim, num_raw_bins)
        raw = torch.nn.functional.interpolate(raw, size=self.num_bins, mode='linear', align_corners=False)
        raw = raw.squeeze(0).T  # (num_bins, embed_dim)

        return raw.cpu().numpy()

    def get_cached_features(self, gene_id: str) -> Optional[np.ndarray]:
        """Load cached features for a gene if available."""
        cache_file = self.cache_dir / "borzoi_features.h5"
        if not cache_file.exists():
            return None
        try:
            with h5py.File(cache_file, "r") as f:
                if gene_id in f:
                    return f[gene_id][:]
        except Exception as e:
            logger.warning(f"Could not read cache for {gene_id}: {e}")
        return None

    def cache_features(self, gene_id: str, features: np.ndarray):
        """Save features for a gene to the HDF5 cache."""
        cache_file = self.cache_dir / "borzoi_features.h5"
        try:
            with h5py.File(cache_file, "a") as f:
                if gene_id not in f:
                    f.create_dataset(gene_id, data=features, compression="gzip")
        except Exception as e:
            logger.warning(f"Could not cache features for {gene_id}: {e}")

    def get_or_compute_features(self, gene_id: str, sequence_one_hot: np.ndarray) -> np.ndarray:
        """Return cached features if available, otherwise compute and cache."""
        cached = self.get_cached_features(gene_id)
        if cached is not None:
            return cached
        features = self.extract_features(sequence_one_hot)
        self.cache_features(gene_id, features)
        return features


class RandomEncoder:
    """
    Drop-in replacement for BorzoiEncoder that returns random features.
    Useful for rapid pipeline testing without Borzoi installed.
    """

    def __init__(self, embed_dim: int = 512, num_bins: int = 200, seed: int = 42):
        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.rng = np.random.RandomState(seed)
        self._cache: Dict[str, np.ndarray] = {}

    def get_or_compute_features(self, gene_id: str, sequence_one_hot: np.ndarray) -> np.ndarray:
        if gene_id not in self._cache:
            # Use gene_id as seed for reproducibility
            seed = hash(gene_id) % (2 ** 31)
            rng = np.random.RandomState(seed)
            self._cache[gene_id] = rng.randn(self.num_bins, self.embed_dim).astype(np.float32)
        return self._cache[gene_id]
