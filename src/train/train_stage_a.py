from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.losses import multi_positive_softmax_loss
from src.train.trainer_utils import (
    GroupedPairDataset,
    build_retriever,
    collate_grouped_pairs,
    load_training_bundle,
)
from src.utils.io import dump_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed


LOGGER = get_logger(__name__)


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def build_docs(batch: dict[str, list[object]]) -> tuple[list[str], torch.Tensor]:
    positives = batch["positive_sequences"]
    negatives = [sequence for sequences in batch["negative_sequences"] for sequence in sequences]
    docs = positives + negatives
    positive_mask = torch.zeros((len(batch["query_sequences"]), len(docs)), dtype=torch.bool)
    for idx in range(len(positives)):
        positive_mask[idx, idx] = True
    return docs, positive_mask


def evaluate_model(model: torch.nn.Module, loader: DataLoader, temperature: float) -> dict[str, float]:
    model.eval()
    losses = []
    reciprocal_ranks = []
    with torch.no_grad():
        for batch in loader:
            docs, positive_mask = build_docs(batch)
            scores = model.score_sequences(batch["query_sequences"], docs)
            loss = multi_positive_softmax_loss(scores, positive_mask.to(scores.device), temperature=temperature)
            losses.append(float(loss.item()))
            for row_idx in range(scores.size(0)):
                target_score = scores[row_idx, row_idx].item()
                rank = int((scores[row_idx] > target_score).sum().item()) + 1
                reciprocal_ranks.append(1.0 / rank)
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "mrr": float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage A retrieval model")
    parser.add_argument("--config", default="configs/train/stage_a.yaml")
    args = parser.parse_args()

    train_cfg, model_cfg = load_training_bundle(args.config)
    set_global_seed(int(train_cfg["seed"]))

    device = resolve_device(str(train_cfg.get("device", "cpu")))
    manifest_df = pd.read_csv(train_cfg["data"]["manifest_path"])
    pair_df = pd.read_csv(train_cfg["data"]["pair_path"])
    sequence_column = train_cfg["data"]["sequence_column"]
    negatives_per_query = int(train_cfg["data"]["negatives_per_query"])

    train_dataset = GroupedPairDataset(
        manifest_df=manifest_df,
        pair_df=pair_df,
        split="train",
        sequence_column=sequence_column,
        negatives_per_query=negatives_per_query,
        seed=int(train_cfg["seed"]),
    )
    val_dataset = GroupedPairDataset(
        manifest_df=manifest_df,
        pair_df=pair_df,
        split="val",
        sequence_column=sequence_column,
        negatives_per_query=negatives_per_query,
        seed=int(train_cfg["seed"]) + 17,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["training"]["num_workers"]),
        collate_fn=collate_grouped_pairs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["training"]["num_workers"]),
        collate_fn=collate_grouped_pairs,
    )

    model = build_retriever(model_cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["training"]["learning_rate"]),
        weight_decay=float(train_cfg["training"]["weight_decay"]),
    )
    scaler_enabled = bool(train_cfg["training"].get("bf16", True)) and device == "cuda"
    autocast_dtype = torch.bfloat16 if scaler_enabled else torch.float32
    temperature = float(model_cfg.get("score_temperature", 0.02))
    best_mrr = float("-inf")
    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    checkpoint_path = output_dir / "checkpoints" / train_cfg["training"]["save_name"]
    metrics_path = output_dir / "metrics" / "train_stage_a_summary.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    global_step = 0
    grad_accum_steps = int(train_cfg["training"]["grad_accum_steps"])
    for epoch in range(int(train_cfg["training"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for batch in train_loader:
            docs, positive_mask = build_docs(batch)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=scaler_enabled):
                scores = model.score_sequences(batch["query_sequences"], docs)
                loss = multi_positive_softmax_loss(
                    scores,
                    positive_mask.to(scores.device),
                    temperature=temperature,
                ) / grad_accum_steps
            loss.backward()
            running_loss += float(loss.item())
            global_step += 1

            if global_step % grad_accum_steps == 0:
                clip_grad_norm_(model.parameters(), float(train_cfg["training"]["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        val_metrics = evaluate_model(model, val_loader, temperature=temperature)
        LOGGER.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_mrr=%.4f",
            epoch + 1,
            running_loss / max(len(train_loader), 1),
            val_metrics["loss"],
            val_metrics["mrr"],
        )
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_cfg,
                    "train_config": train_cfg,
                    "best_val_mrr": best_mrr,
                },
                checkpoint_path,
            )

    dump_json({"best_val_mrr": best_mrr, "checkpoint": str(checkpoint_path)}, metrics_path)
    LOGGER.info("Saved best checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
