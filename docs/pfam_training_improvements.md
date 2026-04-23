# Pfam training improvements — 2026-04-23

Notes covering four linked changes to the ESM-2 ColBERT + MeanPool Pfam training recipe: memory efficiency, overfitting regularization, per-run output directories, and structured train/val metric logging.

## 1. Memory efficiency

### Problem
The initial run OOM'd on 48 GB GPUs (L40S / A6000) at `batch_size=8` with `max_length=1022`. A probe revealed per-step memory was saturating ~46 GiB even at `batch_size=4` — implying gradient checkpointing was not engaging despite `gradient_checkpointing: true` in the config.

### Root cause
Two compounding issues in [src/models/backbone.py](../src/models/backbone.py):

1. **PEFT + gradient checkpointing silently no-ops unless input grads are forced.** When LoRA freezes base-model parameters, the embedding output has `requires_grad=False`, so `torch.utils.checkpoint.checkpoint(...)` has no grad-requiring inputs and short-circuits. The standard fix is `enable_input_require_grads()`.
2. **HF gates gradient checkpointing on `self.training`.** Without `model.train()`, the flag is ignored — relevant to the probe script but not the training loop itself (which does set train mode).

### Fixes
In [src/models/backbone.py:142-151](../src/models/backbone.py):
```python
if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
    if hasattr(self.model, "enable_input_require_grads"):
        self.model.enable_input_require_grads()
    self.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
```

In [configs/model/esm2_colbert.yaml](../configs/model/esm2_colbert.yaml) and [configs/model/esm2_meanpool.yaml](../configs/model/esm2_meanpool.yaml):
- `max_length: 1022 → 512` (Pfam is extremely short-tailed: median 122 AA, 98% ≤ 512)
- `gradient_checkpointing: true` (already set, now actually works)

### Result
Peak memory at matching per-step work dropped ~4×. Subprocess probe in [scripts/probe_max_batch.py](../scripts/probe_max_batch.py):

| batch_size | docs/batch | peak memory | status |
|---|---|---|---|
| 8  | 128 | 20.6 GiB | fits easily |
| 16 | 256 | 38.6 GiB | **fits, used** |
| 18 | 288 | —        | OOM |

Config updated to `batch_size: 16`, `grad_accum_steps: 2` — same effective batch (64) as before, but 2× larger per-forward for better GPU utilization.

## 2. Reducing overfitting

### Problem
A 5-epoch ColBERT run produced:
```
epoch=5 train_loss=0.0020 val_loss=8.6507 val_mrr=0.4945
```
Train collapsed to ~0 while val loss was catastrophic and val MRR (0.49) was only ~30% above random for 7-way ranking. Classic overfitting.

### Root cause analysis
Five candidate causes, prioritized:

1. **Train/val split leakage** — ruled out: the Pfam manifest uses clan-disjoint splits (whole clans held out). Verified 0 families and 0 clans span multiple splits.
2. **Too many epochs** — train loss already at 0.002, further epochs only deepened memorization.
3. **Overconfident logits** — `score_temperature: 0.1` scales logits by 10×, causing the loss to explode on confidently-wrong predictions.
4. **Insufficient regularization** — `lora_rank: 16`, `lora_dropout: 0.1`, `weight_decay: 0.01` gave the adapter too much unregularized capacity.
5. **Easy / too few negatives** — the `pfam_pairs_sub.csv` was uniformly row-sampled, destroying per-query structure (median 2 negatives/query instead of the intended 15).

### Fixes

**Model config** ([configs/model/esm2_colbert.yaml](../configs/model/esm2_colbert.yaml), [configs/model/esm2_meanpool.yaml](../configs/model/esm2_meanpool.yaml)):
| param | before | after |
|---|---|---|
| `score_temperature` | 0.1 | 0.3 |
| `lora_rank`         | 16  | 8   |
| `lora_alpha`        | 32  | 16  |
| `lora_dropout`      | 0.1 | 0.2 |

**Training config** ([configs/train/stage_a_pfam.yaml](../configs/train/stage_a_pfam.yaml), [configs/train/stage_a_pfam_meanpool.yaml](../configs/train/stage_a_pfam_meanpool.yaml)):
| param | before | after |
|---|---|---|
| `epochs`                | 5    | 3 (with early stop) |
| `weight_decay`          | 0.01 | 0.05 |
| `negatives_per_query`   | 6    | 15   |
| `pair_path`             | `pfam_pairs_sub.csv` | `pfam_pairs_sub_v2.csv` |

**New pair file**: [scripts/subsample_pfam_pairs.py](../scripts/subsample_pfam_pairs.py) resamples by *query* (not rows), preserving all 4 positives + 15 negatives per query. Written to `data/processed/pfam/pfam_pairs_sub_v2.csv`. Stats: 84k train / 22k val / 22k test queries, 15 negs/query (was 1–9, median 2).

## 3. Timestamped output directories

### Problem
Each run wrote to `outputs/pfam_colbert/checkpoints/pfam_colbert_best.pt`, silently overwriting the previous run's checkpoint and metrics.

### Fix
In [src/train/train_stage_a.py](../src/train/train_stage_a.py), rank 0 generates a timestamp and broadcasts to all DDP ranks:

```python
if is_ddp:
    obj = [time.strftime("%Y%m%d_%H%M%S") if is_main else None]
    dist.broadcast_object_list(obj, src=0)
    run_id = obj[0]
else:
    run_id = time.strftime("%Y%m%d_%H%M%S")
output_dir = output_dir / run_id
```

Resulting layout:
```
outputs/pfam_colbert/
├── 20260423_112700/
│   ├── checkpoints/{pfam_colbert_best.pt, latest.pt}
│   └── metrics/{train_log.jsonl, train_stage_a_summary.json}
├── 20260424_091500/
│   └── ...
```

**Resume note:** auto-resume via `latest.pt` in the base output dir no longer fires (each run starts in a fresh subdir). Explicit `--resume <path>` still works.

## 4. Train/val metric logging + mid-epoch validation + early stopping

### Problem
With only one validation pass per epoch, we couldn't tell *when* during an epoch overfitting set in — and all training/validation losses were only visible as terminal scrollback.

### Changes
In [src/train/train_stage_a.py](../src/train/train_stage_a.py):

**Mid-epoch validation.** Controlled by `eval_interval_steps` (counted in optimizer steps, so `grad_accum_steps` is correctly accounted for):
```python
opt_step_boundary = (
    eval_interval_steps > 0
    and global_step % grad_accum_steps == 0
    and (global_step // grad_accum_steps) % eval_interval_steps == 0
)
```
A full val pass runs at each boundary. `model.train()` is restored afterward.

**Early stopping.** New `early_stop_patience` config: stops training when val MRR hasn't improved for N consecutive evaluations (either mid-epoch or epoch-end). The stop flag is DDP-synced via `dist.broadcast`.

**JSONL metrics log** at `outputs/<run>/<timestamp>/metrics/train_log.jsonl`, one record per event:
- `type: "train"` — from the periodic `log_interval` tick. Fields: `epoch`, `global_step`, `epoch_step`, `loss`, `secs_per_step`, `wall_time`.
- `type: "val"` — from every mid-epoch evaluation. Fields: `epoch`, `global_step`, `epoch_step`, `val_loss`, `val_mrr`, `wall_time`.
- `type: "epoch_end"` — full epoch summary. Fields: `epoch`, `global_step`, `train_loss`, `val_loss`, `val_mrr`, `epoch_mins`, `wall_time`.

### New config knobs (added to both Pfam training configs)
```yaml
eval_interval_steps: 500    # eval every 500 optimizer steps (~3× per epoch)
early_stop_patience: 3      # stop if val MRR doesn't improve in 3 consecutive evals
```

### Reading the log
```python
import pandas as pd
df = pd.read_json("outputs/pfam_colbert/20260423_112700/metrics/train_log.jsonl", lines=True)
df[df.type == "train"].plot("global_step", "loss")
df[df.type == "val"].plot("global_step", ["val_loss", "val_mrr"])
```

## Files changed summary

| file | purpose |
|---|---|
| [src/models/backbone.py](../src/models/backbone.py) | PEFT + gradient checkpointing fix |
| [src/train/train_stage_a.py](../src/train/train_stage_a.py) | mid-epoch eval, early stopping, JSONL log, timestamped output dir |
| [configs/model/esm2_colbert.yaml](../configs/model/esm2_colbert.yaml) | max_length=512, regularization, GC on |
| [configs/model/esm2_meanpool.yaml](../configs/model/esm2_meanpool.yaml) | same, for meanpool |
| [configs/train/stage_a_pfam.yaml](../configs/train/stage_a_pfam.yaml) | new pair file, bigger batch, more negs, eval/stop knobs |
| [configs/train/stage_a_pfam_meanpool.yaml](../configs/train/stage_a_pfam_meanpool.yaml) | same, for meanpool |
| [scripts/subsample_pfam_pairs.py](../scripts/subsample_pfam_pairs.py) | new — per-query subsampler |
| [scripts/probe_max_batch.py](../scripts/probe_max_batch.py) | new — batch-size memory probe |
| [run.sh](../run.sh) | new `pfam-colbert` target for torchrun launch |

## Open questions / follow-ups

- Whether `temperature=0.3` is near the optimum vs `0.2` / `0.5`.
- Whether `max_length: 256` (truncating 15% of sequences) would give a worthwhile speedup for iteration.
- Whether a LR warmup + cosine decay would help; currently constant LR.
- No LR scheduler currently — worth adding if multi-epoch runs are common.
