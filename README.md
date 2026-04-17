# Protein Retrieval with ColBERT and ESM-2

Late-interaction retrieval of protein domains using ColBERT-style MaxSim scoring over ESM-2 embeddings. Given a query protein sequence, retrieve functionally or structurally similar domains.

Supports two datasets:
- **Pfam-A (InterPro)** — 189k domain sequences, 12k families, 811 clans (functional/evolutionary classification)
- **SCOPe** — 35k protein domains, 2,051 superfamilies (structural classification)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
```

## Launching Training

All training commands should run in a persistent tmux session so they survive disconnections.

### Pfam training (recommended, default)

**1. Download Pfam data (~175 MB):**

```bash
mkdir -p data/raw/pfam
curl -sL -o data/raw/pfam/Pfam-A.seed.gz \
  "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.seed.gz"
curl -sL -o data/raw/pfam/Pfam-A.clans.tsv.gz \
  "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"
```

**2. Build manifest and pairs:**

```bash
# Parse Pfam-A.seed → manifest (clan-held-out splits)
python -m src.data.parse_pfam --max-per-family 20 --require-clan

# Build retrieval pairs (same-family positives, same-clan hard negatives)
python -m src.data.build_pfam_pairs
```

**3. Launch training on 2 GPUs:**

```bash
tmux new-session -d -s train \
  "PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 \
   python -m src.train.train_stage_a --config configs/train/stage_a_pfam.yaml \
   2>&1 | tee outputs/pfam_colbert_train.log"

# Watch live:
tmux attach -t train
# (Ctrl-b then d to detach)

# Or follow the log file:
tail -f outputs/pfam_colbert_train.log
```

**Pfam training config** (`configs/train/stage_a_pfam.yaml`):
- ESM-2 650M backbone + LoRA (rank 16, on query/value projections)
- ColBERT late-interaction scoring (64-dim projection, MaxSim)
- Batch size 6 × grad accumulation 8 = effective batch of 48 queries
- Gradient checkpointing enabled (fits in 11 GB per 2080 Ti)
- 5 epochs, learning rate 1e-4, InfoNCE loss with τ=0.1

### SCOPe training (alternative)

**1. Download SCOPe data (~43 MB):**

```bash
mkdir -p data/raw/scope
curl -sL -o data/raw/scope/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa \
  "http://scop.berkeley.edu/downloads/scopeseq-2.08/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa"
curl -sL -o data/raw/scope/dir.cla.scope.2.08-stable.txt \
  "http://scop.berkeley.edu/downloads/parse/dir.cla.scope.2.08-stable.txt"
```

**2. Build manifest and pairs:**

```bash
python -m src.data.parse_scope
python -m src.data.build_scope_pairs
```

**3. Launch training on 2 GPUs:**

```bash
tmux new-session -d -s train \
  "PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 \
   python -m src.train.train_stage_a --config configs/train/stage_a_esm2.yaml \
   2>&1 | tee outputs/esm2_colbert_train.log"
```

### Smoke test (single GPU, fast validation)

Uses the small ESM-2 35M model and 1 epoch to verify the pipeline end-to-end:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2_smoke.yaml
```

### Resuming from a checkpoint

Training saves `latest.pt` after every epoch. To resume after interruption:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a \
  --config configs/train/stage_a_pfam.yaml \
  --resume outputs/pfam_colbert/checkpoints/latest.pt
```

If `latest.pt` exists, training will auto-resume even without `--resume`.

### Mean-pooled baseline (same backbone, no late interaction)

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2_meanpool.yaml
```

This is the critical architectural comparison — identical training setup but replaces ColBERT's MaxSim scoring with mean-pooled cosine similarity.

## Project Overview

**Task:** Given only a raw amino acid sequence, retrieve protein domains in the same family or superfamily.

**Why late interaction:** Protein function arises from sparse combinations of local motifs (binding sites, catalytic residues, conserved domains) at varying positions. ColBERT's token-level MaxSim scoring captures partial motif matches that mean pooling averages away.

**Architecture:**
- **Backbone:** ESM-2 650M (Meta's protein language model)
- **Fine-tuning:** LoRA (rank 16) on query/value projections
- **Scoring:** ColBERT MaxSim — shared encoder → 64-dim projection → L2 norm → token-level matching
- **Training:** Multi-positive InfoNCE loss, τ=0.1, gradient accumulation for stable updates

## Datasets

### Pfam-A (default)

Domain families from the Pfam database (InterPro's largest member database), using the curated seed alignments.

- **Size:** 189,450 sequences (30–1022 residues), 12,329 families, 811 clans
- **Splits:** Clan-held-out (entire clans held out for val/test to prevent leakage)
- **Pairs:** Positives = same family; hard negatives = same clan, different family

| Split | Sequences | Total Pairs |
|-------|-----------|-------------|
| Train | 141,214 | 2,673,848 |
| Val | 19,379 | 367,242 |
| Test | 28,857 | 546,339 |

### SCOPe

Structural Classification of Proteins (extended). Gold-standard hierarchical classification by 3D structure.

- **Size:** 35,094 domains, 2,051 superfamilies
- **Splits:** Fold-held-out (prevents structural leakage)
- **Pairs:** Positives = same superfamily; hard negatives = same fold, different superfamily

## Evaluation

```bash
# Pretrained ESM-2 (zero-shot, no fine-tuning)
CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_pretrained \
  --config configs/train/eval_pfam.yaml

# Fine-tuned ColBERT model
CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_screen \
  --config configs/train/eval_pfam.yaml --baseline model \
  --checkpoint outputs/pfam_colbert/checkpoints/pfam_colbert_best.pt

# All classical baselines
python -m src.eval.run_all_baselines \
  --config configs/train/eval_pfam.yaml \
  --baselines random tfidf_kmer minhash gkm_svm oracle
```

## Repository Structure

```
├── configs/
│   ├── model/
│   │   ├── esm2_colbert.yaml          # ESM-2 650M + ColBERT (main model)
│   │   ├── esm2_meanpool.yaml         # ESM-2 650M + mean pooling (baseline)
│   │   └── esm2_colbert_small.yaml    # ESM-2 35M (smoke test)
│   └── train/
│       ├── stage_a_pfam.yaml          # Pfam training
│       ├── stage_a_esm2.yaml          # SCOPe training
│       └── stage_a_esm2_smoke.yaml    # Fast smoke test
├── src/
│   ├── data/
│   │   ├── parse_pfam.py              # Pfam Stockholm → manifest
│   │   ├── build_pfam_pairs.py        # Pfam pair construction
│   │   ├── parse_scope.py             # SCOPe → manifest
│   │   └── build_scope_pairs.py       # SCOPe pair construction
│   ├── models/
│   │   ├── backbone.py                # ESM-2 / DNABERT-2 with LoRA
│   │   ├── colbert.py                 # ColBERT late-interaction retriever
│   │   ├── meanpool.py                # Mean-pooled baseline retriever
│   │   └── losses.py                  # Multi-positive InfoNCE loss
│   ├── train/
│   │   ├── train_stage_a.py           # Main training loop (auto-resume, logging)
│   │   └── trainer_utils.py           # Dataset, tokenizing collator, model builder
│   └── eval/
│       ├── eval_pretrained.py         # Zero-shot pretrained evaluation
│       ├── run_all_baselines.py       # Full baseline suite
│       └── eval_screen.py             # Fine-tuned model evaluation
├── report.md                          # Full research report
└── progress_report.pdf                # ICML-style progress report
```

## Hardware Notes

Developed and tested on:
- 2× NVIDIA RTX 2080 Ti (11 GB each), 12 CPU cores, 125 GB RAM
- FP16 mixed precision (2080 Ti does not support BF16)

Memory footprint on 2080 Ti for ESM-2 650M + ColBERT + LoRA + batch_size 6:
- GPU 0: ~10 GB (model + DataParallel primary + optimizer state + ColBERT head)
- GPU 1: ~6–7 GB (DataParallel replica only)

For smaller GPUs, reduce `batch_size` or switch to `esm2_colbert_small.yaml` (35M model).

## Key Design Decisions

- **Held-out splits:** Entire Pfam clans (or SCOPe folds) held out for val/test to prevent leakage from related families.
- **Hard negatives:** Same clan/fold, different family/superfamily. Forces the model to learn beyond gross similarity.
- **LoRA over full fine-tuning:** Preserves ESM-2's pretrained representations while adapting ~2M parameters.
- **Gradient accumulation:** Stabilizes contrastive learning by averaging gradients across multiple batches.
- **Gradient checkpointing:** Trades compute for memory, enabling larger effective batches on 11 GB GPUs.

## References

- Pfam: Mistry et al., "Pfam: The protein families database in 2021," Nucleic Acids Research
- InterPro: Paysan-Lafosse et al., "InterPro in 2022," Nucleic Acids Research
- SCOPe: Fox et al., "SCOPe: Structural Classification of Proteins -- extended," NAR, 2014
- ESM-2: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science, 2023
- ColBERT: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR, 2020
