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
            raise ImportError("transformers is required for DNABERT-2 backbones")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self.hidden_size = int(self.model.config.hidden_size)
        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        if config.use_lora and get_peft_model is not None and LoraConfig is not None:
            lora_cfg = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)

    def forward_sequences(self, sequences: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_fallback:
            return self.model.forward_sequences(sequences, device)

        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"].bool()
        return hidden, attention_mask


def build_backbone(config_dict: dict[str, Any]) -> DNAEncoderBackbone:
    return DNAEncoderBackbone(BackboneConfig(**config_dict))
