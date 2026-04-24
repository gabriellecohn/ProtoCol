from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.models.losses import multi_positive_softmax_loss
from src.train.trainer_utils import (
    GroupedPairDataset,
    TokenizedCollator,
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
    was_training = model.training
    model.eval()
    losses = []
    reciprocal_ranks = []
    with torch.no_grad():
        for batch in loader:
            scores = model.score_token_batches(batch["query_tokens"], batch["doc_tokens"])
            positive_mask = batch["positive_mask"]
            loss = multi_positive_softmax_loss(scores, positive_mask.to(scores.device), temperature=temperature)
            losses.append(float(loss.item()))
            for row_idx in range(scores.size(0)):
                target_score = scores[row_idx, row_idx].item()
                rank = int((scores[row_idx] > target_score).sum().item()) + 1
                reciprocal_ranks.append(1.0 / rank)
    if was_training:
        model.train()
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "mrr": float(sum(reciprocal_ranks) / max(len(reciprocal_ranks), 1)),
    }


def _resolve_run_id(is_ddp: bool, is_main: bool) -> str:
    if is_ddp:
        obj = [time.strftime("%Y%m%d_%H%M%S") if is_main else None]
        dist.broadcast_object_list(obj, src=0)
        return obj[0]
    return time.strftime("%Y%m%d_%H%M%S")


def _log_event(log_path: Path, event: dict[str, object]) -> None:
    with log_path.open("a") as fh:
        fh.write(json.dumps(event) + "\n")


def _should_stop(stop: bool, is_ddp: bool) -> bool:
    if not is_ddp:
        return stop
    flag = torch.tensor([1 if stop else 0], dtype=torch.int32, device="cuda")
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage A retrieval model")
    parser.add_argument("--config", default="configs/train/stage_a.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # DDP detection — if LOCAL_RANK is set, we're under torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank >= 0
    if is_ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    is_main = (not is_ddp) or dist.get_rank() == 0

    train_cfg, model_cfg = load_training_bundle(args.config)
    set_global_seed(int(train_cfg["seed"]))

    if is_ddp:
        device = f"cuda:{local_rank}"
    else:
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

    model = build_retriever(model_cfg).to(device)

    # Use pre-tokenizing collator so DataLoader workers handle tokenization
    # on CPU while the GPU processes the current batch.
    collator = TokenizedCollator(model.backbone.tokenize)
    num_workers = int(train_cfg["training"].get("num_workers", 0))
    if num_workers == 0:
        num_workers = 4  # default to overlapping workers
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        persistent_workers=True,
        prefetch_factor=2,
    )
    # Fast val loader: random subset for mid-epoch evals so they don't dominate wall time.
    # Full val_loader is still used at epoch end.
    eval_subsample = int(train_cfg["training"].get("eval_subsample_queries", 0))
    if eval_subsample > 0 and eval_subsample < len(val_dataset):
        rng_sub = torch.Generator().manual_seed(int(train_cfg["seed"]) + 99)
        sub_indices = torch.randperm(len(val_dataset), generator=rng_sub)[:eval_subsample].tolist()
        val_loader_fast = DataLoader(
            torch.utils.data.Subset(val_dataset, sub_indices),
            batch_size=int(train_cfg["training"]["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
            persistent_workers=True,
            prefetch_factor=2,
        )
        if is_main:
            LOGGER.info(
                "Mid-epoch eval will use %d/%d val queries (full set used at epoch end)",
                eval_subsample, len(val_dataset),
            )
    else:
        val_loader_fast = val_loader
    gpu_ids = train_cfg["training"].get("gpu_ids")
    if not is_ddp and gpu_ids and len(gpu_ids) > 1 and hasattr(model, "backbone"):
        LOGGER.info("Wrapping backbone encoder in DataParallel across GPUs %s", gpu_ids)
        model.backbone.model = torch.nn.DataParallel(model.backbone.model, device_ids=gpu_ids)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg["training"]["learning_rate"]),
        weight_decay=float(train_cfg["training"]["weight_decay"]),
    )
    device_type = device.split(":")[0]
    scaler_enabled = bool(train_cfg["training"].get("bf16", True)) and device_type == "cuda"
    autocast_dtype = torch.bfloat16 if scaler_enabled else torch.float32
    temperature = float(model_cfg.get("score_temperature", 0.02))
    best_mrr = float("-inf")
    start_epoch = 0

    # Timestamped per-run output dir: rank 0 picks the stamp, all ranks use it.
    run_id = _resolve_run_id(is_ddp, is_main)
    base_output_dir = Path(train_cfg.get("output_dir", "outputs"))
    output_dir = base_output_dir / run_id
    checkpoint_path = output_dir / "checkpoints" / train_cfg["training"]["save_name"]
    latest_path = output_dir / "checkpoints" / "latest.pt"
    metrics_dir = output_dir / "metrics"
    metrics_path = metrics_dir / "train_stage_a_summary.json"
    jsonl_log_path = metrics_dir / "train_log.jsonl"
    if is_main:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Run output directory: %s", output_dir)

    # Resume from explicit --resume only. Auto-resume from latest.pt is removed
    # because each run writes to a fresh timestamped subdir.
    resume_path = args.resume
    if resume_path and Path(resume_path).exists():
        LOGGER.info("Resuming from checkpoint %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0)
        best_mrr = ckpt.get("best_val_mrr", float("-inf"))
        LOGGER.info("Resumed at epoch %d, best_val_mrr=%.4f", start_epoch, best_mrr)

    global_step = 0
    grad_accum_steps = int(train_cfg["training"]["grad_accum_steps"])
    log_interval = int(train_cfg["training"].get("log_interval", 100))
    eval_interval_steps = int(train_cfg["training"].get("eval_interval_steps", 0))
    early_stop_patience = int(train_cfg["training"].get("early_stop_patience", 0))
    patience_counter = 0
    stop_training = False
    wall_start = time.time()
    last_log_time = wall_start

    def _run_validation(epoch_1: int, epoch_step: int, loader: DataLoader) -> None:
        nonlocal best_mrr, patience_counter, stop_training
        val_metrics = evaluate_model(model, loader, temperature=temperature)
        if is_main:
            LOGGER.info(
                "[val] epoch=%d step=%d val_loss=%.4f val_mrr=%.4f",
                epoch_1, epoch_step, val_metrics["loss"], val_metrics["mrr"],
            )
            _log_event(jsonl_log_path, {
                "type": "val",
                "epoch": epoch_1,
                "global_step": global_step,
                "epoch_step": epoch_step,
                "val_loss": val_metrics["loss"],
                "val_mrr": val_metrics["mrr"],
                "wall_time": time.time() - wall_start,
            })
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            patience_counter = 0
            if is_main:
                save_state = {
                    k.replace("backbone.model.module.", "backbone.model."): v
                    for k, v in model.state_dict().items()
                }
                torch.save(
                    {
                        "model_state": save_state,
                        "model_config": model_cfg,
                        "train_config": train_cfg,
                        "best_val_mrr": best_mrr,
                    },
                    checkpoint_path,
                )
                LOGGER.info("New best MRR=%.4f, saved to %s", best_mrr, checkpoint_path)
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                stop_training = True
                if is_main:
                    LOGGER.info(
                        "Early stopping: val MRR did not improve for %d consecutive evals",
                        early_stop_patience,
                    )

    for epoch in range(start_epoch, int(train_cfg["training"]["epochs"])):
        epoch_1 = epoch + 1
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_step = 0
        num_batches = len(train_loader)
        for batch in train_loader:
            with torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=scaler_enabled):
                scores = model.score_token_batches(batch["query_tokens"], batch["doc_tokens"])
                loss = multi_positive_softmax_loss(
                    scores,
                    batch["positive_mask"].to(scores.device),
                    temperature=temperature,
                ) / grad_accum_steps
            loss.backward()
            running_loss += float(loss.item())
            global_step += 1
            epoch_step += 1

            did_opt_step = False
            if global_step % grad_accum_steps == 0:
                clip_grad_norm_(model.parameters(), float(train_cfg["training"]["max_grad_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                did_opt_step = True

            if epoch_step % log_interval == 0:
                now = time.time()
                secs_per_step = (now - last_log_time) / max(log_interval, 1)
                last_log_time = now
                avg_loss = running_loss / epoch_step
                if is_main:
                    LOGGER.info(
                        "epoch=%d step=%d/%d loss=%.4f sec/step=%.3f",
                        epoch_1, epoch_step, num_batches, avg_loss, secs_per_step,
                    )
                    _log_event(jsonl_log_path, {
                        "type": "train",
                        "epoch": epoch_1,
                        "global_step": global_step,
                        "epoch_step": epoch_step,
                        "loss": avg_loss,
                        "secs_per_step": secs_per_step,
                        "wall_time": now - wall_start,
                    })

            # Mid-epoch validation on optimizer-step boundaries
            if did_opt_step and eval_interval_steps > 0:
                opt_steps = global_step // grad_accum_steps
                if opt_steps % eval_interval_steps == 0:
                    _run_validation(epoch_1, epoch_step, val_loader_fast)
                    if _should_stop(stop_training, is_ddp):
                        break

        if _should_stop(stop_training, is_ddp):
            break

        # End-of-epoch validation
        val_metrics = evaluate_model(model, val_loader, temperature=temperature)
        train_loss = running_loss / max(len(train_loader), 1)
        epoch_mins = (time.time() - epoch_start) / 60.0
        if is_main:
            LOGGER.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f val_mrr=%.4f (%.1f min)",
                epoch_1, train_loss, val_metrics["loss"], val_metrics["mrr"], epoch_mins,
            )
            _log_event(jsonl_log_path, {
                "type": "epoch_end",
                "epoch": epoch_1,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_mrr": val_metrics["mrr"],
                "epoch_mins": epoch_mins,
                "wall_time": time.time() - wall_start,
            })
        # Save latest.pt (for explicit resume)
        if is_main:
            save_state = {
                k.replace("backbone.model.module.", "backbone.model."): v
                for k, v in model.state_dict().items()
            }
            torch.save(
                {
                    "model_state": save_state,
                    "model_config": model_cfg,
                    "train_config": train_cfg,
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch_1,
                    "best_val_mrr": best_mrr,
                },
                latest_path,
            )
        # Update best/patience based on end-of-epoch eval too
        if val_metrics["mrr"] > best_mrr:
            best_mrr = val_metrics["mrr"]
            patience_counter = 0
            if is_main:
                torch.save(
                    {
                        "model_state": save_state,
                        "model_config": model_cfg,
                        "train_config": train_cfg,
                        "best_val_mrr": best_mrr,
                    },
                    checkpoint_path,
                )
                LOGGER.info("New best MRR=%.4f, saved to %s", best_mrr, checkpoint_path)
        else:
            patience_counter += 1
            if early_stop_patience > 0 and patience_counter >= early_stop_patience:
                stop_training = True

        if _should_stop(stop_training, is_ddp):
            break

    if is_main:
        dump_json({"best_val_mrr": best_mrr, "checkpoint": str(checkpoint_path)}, metrics_path)
        LOGGER.info("Saved best checkpoint to %s", checkpoint_path)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
