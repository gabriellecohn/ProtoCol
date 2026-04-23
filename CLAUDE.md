# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ColBERT-style late-interaction retrieval of protein domains using ESM-2 embeddings.** Given a raw amino acid sequence, retrieve functionally or structurally similar protein domains.

Two datasets are supported:
- **Pfam-A** (primary): 189k domain sequences, 12k families, 811 clans; clan-held-out splits
- **SCOPe** (alternative): 35k domains, 2,051 superfamilies; fold-held-out splits

Two coexisting fine-tuning strategies for ESM-2:
- **LoRA** (default): ~2M trainable params on query/value projections — less memory, lower forgetting risk
- **Layerwise** (active branch): last 3 of 33 transformer layers unfrozen (~60M params) — full-rank updates, no peft dependency

Legacy code for a CN-aware diffusion model (regulatory attention maps) is present in `src/training/`, `src/inference/`, `train.py`, `infer.py` but is not the active work.

## Commands

### Data Preparation (run sequentially)

```bash
# Pfam (recommended)
mkdir -p data/raw/pfam
curl -sL -o data/raw/pfam/Pfam-A.seed.gz "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"
curl -sL -o data/raw/pfam/Pfam-A.clans.tsv.gz "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"
python -m src.data.parse_pfam --max-per-family 20 --require-clan
python -m src.data.build_pfam_pairs

# SCOPe (alternative)
python -m src.data.parse_scope
python -m src.data.build_scope_pairs
```

### Training

```bash
# Smoke test — small ESM-2 35M, 1 epoch, single GPU
CUDA_VISIBLE_DEVICES=0 python -m src.train.train_stage_a --config configs/train/stage_a_esm2_smoke.yaml

# Pfam + LoRA (primary)
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a --config configs/train/stage_a_pfam.yaml

# Pfam + layerwise (last-3-layers unfrozen)
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a --config configs/train/stage_a_pfam_unfrozen.yaml

# SCOPe + LoRA
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a --config configs/train/stage_a_esm2.yaml

# Mean-pooled baseline (same backbone, no late interaction — key architectural comparison)
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a --config configs/train/stage_a_esm2_meanpool.yaml

# Resume from checkpoint (also auto-resumes if latest.pt exists in output dir)
python -m src.train.train_stage_a --config configs/train/stage_a_pfam.yaml \
  --resume outputs/pfam_colbert/checkpoints/latest.pt
```

### Evaluation

```bash
# Zero-shot pretrained ESM-2 (no fine-tuning)
CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_pretrained --config configs/train/eval_pfam.yaml

# Fine-tuned model
CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_screen \
  --config configs/train/eval_pfam.yaml --baseline model \
  --checkpoint outputs/pfam_colbert/checkpoints/pfam_colbert_best.pt

# Classical baselines (random, tfidf_kmer, minhash, gkm_svm, oracle)
python -m src.eval.run_all_baselines --config configs/train/eval_pfam.yaml \
  --baselines random tfidf_kmer minhash gkm_svm oracle
```

### SLURM

```bash
sbatch run_slurm.sh                      # smoke (default)
sbatch --export=ALL,RUN_TARGET=stage-a run_slurm.sh
# RUN_TARGET options: smoke, synthetic, stage-a, eval, train, infer
```

Long-running training should be launched in a tmux session: `tmux new-session -d -s train "... 2>&1 | tee outputs/train.log"`.

## Architecture

### Pipeline

```
Raw sequences (Stockholm/FASTA)
  → parse_{pfam,scope}.py  (clan/fold-held-out manifest CSVs)
  → build_{pfam,scope}_pairs.py  (query + [same-family positives] + [same-clan hard negatives])
  → train_stage_a.py  (GroupedPairDataset → TokenizedCollator → ColBERT forward → InfoNCE loss)
  → eval_{screen,pretrained}.py  (encode corpus once → score all pairs → Recall@k, MRR, nDCG)
```

### Models (`src/models/`)

**`backbone.py`** — `DNAEncoderBackbone` wrapping HuggingFace transformers:
- `use_lora: true` → applies PEFT LoRA (rank 16) on `q_proj`/`v_proj`
- `num_unfrozen_layers: N` (with `use_lora: false`) → freezes all but last N transformer layers; uses `input_require_grads()` to preserve the backward path through frozen weights
- Supports ESM-2 (`encoder.layer`) and DNABERT-2 layer enumeration; `hash-dna-debug` requires no downloads

**`colbert.py`** — `ColBERTRetriever`:
- Shared encoder → linear → 64-dim L2-normalized token embeddings
- MaxSim: `einsum("qif,djf->qidj")` → max per query token → sum → score

**`meanpool.py`** — `MeanPoolRetriever`: same backbone, mean-pool over tokens → cosine similarity. The key ablation: identical setup without late interaction.

**`losses.py`** — multi-positive InfoNCE: `log(Σ exp(positives)) - log(Σ exp(all))`, temperature τ=0.1.

### Training (`src/train/`)

**`trainer_utils.py`**:
- `GroupedPairDataset`: loads manifest + pair CSVs, samples hard negatives per query on the fly
- `TokenizedCollator`: tokenizes on CPU while GPU processes the previous batch (overlapped I/O)

**`train_stage_a.py`**: YAML-driven; FP16 autocast + DataParallel; gradient accumulation + clipping; checkpoint selected by validation MRR; auto-resumes from `latest.pt`.

### Configuration System

All experiments are YAML-driven. Training config (`configs/train/`) merges with model config (`configs/model/`):

| Scenario | Model config | Train config |
|----------|-------------|--------------|
| Pfam + LoRA | `esm2_colbert.yaml` | `stage_a_pfam.yaml` (batch 6 × accum 8, LR 1e-4) |
| Pfam + layerwise | `esm2_colbert_unfrozen.yaml` | `stage_a_pfam_unfrozen.yaml` (batch 4 × accum 12, LR 5e-5) |
| SCOPe | `esm2_colbert.yaml` | `stage_a_esm2.yaml` |
| Smoke | `esm2_colbert_small.yaml` | `stage_a_esm2_smoke.yaml` |
| MeanPool | `esm2_meanpool.yaml` | any train config |

## Key Practical Notes

- **Hardware target**: 2× NVIDIA RTX 2080 Ti (11 GB each). FP16 only (no BF16 on 2080 Ti). Gradient checkpointing required for ESM-2 650M. GPU 0 carries model + optimizer state (~10 GB); GPU 1 carries replica only (~6–7 GB).
- **Layerwise vs LoRA**: Layerwise has ~30× more trainable params but full-rank updates in the last layers. Use lower LR (5e-5 vs 1e-4) and smaller batch (4 vs 6) to avoid forgetting.
- **Held-out split design**: Entire Pfam clans (or SCOPe folds) are held out — not individual sequences. This prevents leakage from near-identical sequences in related families.
- **Hard negatives**: Same clan, different family forces the model to distinguish within-clan structural variation, not just superficial sequence similarity.
- **`data/`, `outputs/`** are excluded from git; generated files are untracked.
- Python ≥ 3.10; cluster uses Miniforge 3.12. Install: `pip install -r requirements.txt`.
- DNABERT-2 and ESM-2 models download from HuggingFace on first use; `hash-dna-debug` is always available locally.
