from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModel = None
    AutoTokenizer = None

try:
    import torch.distributed.tensor  # noqa: F401 — peft 0.18 references this as an attribute
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None


@dataclass
class BackboneConfig:
    model_name: str = "zhihan1996/DNABERT-2-117M"
    max_length: int = 1024
    gradient_checkpointing: bool = True
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    trust_remote_code: bool = True
    lora_target_modules: list[str] | None = None  # auto-detect or explicit
    # Alternative to LoRA: freeze backbone except last N transformer layers
    num_unfrozen_layers: int | None = None  # None = no freezing; int = unfreeze last N


class SimpleDNABackbone(nn.Module):
    def __init__(self, hidden_size: int = 256, max_length: int = 1024) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = nn.Embedding(6, hidden_size)

    def _tokenize(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 5}
        batch = []
        masks = []
        for sequence in sequences:
            ids = [mapping.get(base.upper(), 5) for base in sequence[: self.max_length]]
            mask = [1] * len(ids)
            if len(ids) < self.max_length:
                pad = self.max_length - len(ids)
                ids.extend([0] * pad)
                mask.extend([0] * pad)
            batch.append(ids)
            masks.append(mask)
        return torch.tensor(batch, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)

    def forward_sequences(self, sequences: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask = self._tokenize(sequences)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        hidden = self.embedding(input_ids)
        return hidden, attention_mask


class DNAEncoderBackbone(nn.Module):
    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.is_fallback = False

        if config.model_name == "hash-dna-debug":
            self.model = SimpleDNABackbone(max_length=config.max_length)
            self.hidden_size = self.model.hidden_size
            self.tokenizer = None
            self.is_fallback = True
            return

        if AutoModel is None or AutoTokenizer is None:
            raise ImportError("transformers is required for pretrained backbones")

        from transformers import AutoConfig
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )

        # DNABERT-2 requires special loading (custom model class + Triton fix)
        auto_map = getattr(hf_config, "auto_map", {})
        model_class_ref = auto_map.get("AutoModel")
        if model_class_ref and config.trust_remote_code:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            model_class = get_class_from_dynamic_module(
                model_class_ref,
                config.model_name,
            )
            import sys
            for _mod in sys.modules.values():
                if hasattr(_mod, "flash_attn_qkvpacked_func"):
                    _mod.flash_attn_qkvpacked_func = None
            model_class.config_class = type(hf_config)
            self.model = model_class.from_pretrained(
                config.model_name,
                config=hf_config,
            )
        else:
            # Standard HuggingFace model (ESM-2, BERT, etc.)
            self.model = AutoModel.from_pretrained(
                config.model_name,
                config=hf_config,
                trust_remote_code=config.trust_remote_code,
            )
        self.hidden_size = int(self.model.config.hidden_size)

        # Fine-tuning strategy: LoRA or freeze-all-but-last-N
        if config.use_lora and get_peft_model is not None and LoraConfig is not None:
            # Use explicit targets if provided, otherwise auto-detect
            target_modules = config.lora_target_modules
            if target_modules is None:
                if "DNABERT" in config.model_name or "dnabert" in config.model_name.lower():
                    target_modules = ["Wqkv", "dense"]
                elif "esm" in config.model_name.lower():
                    target_modules = ["query", "value"]
                else:
                    target_modules = ["query", "value"]
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_cfg)
        elif config.num_unfrozen_layers is not None:
            self._freeze_except_last_n(config.num_unfrozen_layers)

        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            # Required when some params are frozen: embeddings need a hook to
            # force their outputs to have requires_grad=True so checkpointed
            # layers get a proper gradient path.
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

    def _freeze_except_last_n(self, n: int) -> None:
        """Freeze all backbone parameters except the last n transformer layers."""
        # Freeze everything first
        for param in self.model.parameters():
            param.requires_grad = False

        # Find the transformer layer list (ESM-2 / BERT / DNABERT-2 all use encoder.layer)
        layers = None
        for candidate in (
            getattr(getattr(self.model, "encoder", None), "layer", None),
            getattr(getattr(getattr(self.model, "esm", None), "encoder", None), "layer", None),
            getattr(getattr(getattr(self.model, "bert", None), "encoder", None), "layer", None),
        ):
            if candidate is not None:
                layers = candidate
                break
        if layers is None:
            raise ValueError(
                f"Could not find transformer layers in model {type(self.model).__name__}; "
                "unfreezing requires a standard encoder.layer list."
            )

        n_layers = len(layers)
        start = max(0, n_layers - n)
        n_trainable = 0
        for i in range(start, n_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
                n_trainable += param.numel()

        total = sum(p.numel() for p in self.model.parameters())
        import logging
        logging.getLogger(__name__).info(
            "Unfroze last %d of %d transformer layers: %.1fM / %.1fM params trainable (%.1f%%)",
            n, n_layers, n_trainable / 1e6, total / 1e6, 100 * n_trainable / total,
        )

    def tokenize(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize sequences on CPU (no GPU transfer)."""
        if self.is_fallback:
            input_ids, attention_mask = self.model._tokenize(sequences)
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return dict(self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ))

    def forward_tokens(self, tokens: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass on pre-tokenized inputs."""
        tokens_dev = {k: v.to(device) for k, v in tokens.items()}
        if self.is_fallback:
            hidden = self.model.embedding(tokens_dev["input_ids"])
            return hidden, tokens_dev["attention_mask"]
        outputs = self.model(**tokens_dev)
        hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        return hidden, tokens_dev["attention_mask"].bool()

    def forward_sequences(self, sequences: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_tokens(self.tokenize(sequences), device)


def build_backbone(config_dict: dict[str, Any]) -> DNAEncoderBackbone:
    return DNAEncoderBackbone(BackboneConfig(**config_dict))
