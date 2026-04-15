# Protein Retrieval with ColBERT and ESM-2

Late-interaction retrieval of protein domains using ColBERT-style MaxSim scoring over ESM-2 embeddings. Given a query protein sequence, retrieve structurally similar domains from the SCOPe database.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare SCOPe dataset (35k protein domains)
python -m src.data.parse_scope
python -m src.data.build_scope_pairs

# Smoke test (ESM-2 35M, 1 epoch, single GPU)
CUDA_VISIBLE_DEVICES=0 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2_smoke.yaml

# Full training (ESM-2 650M, 10 epochs, 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2.yaml
```

## Project Overview

**Task:** Given only a raw amino acid sequence, retrieve protein domains with similar 3D structure (same SCOPe superfamily).

**Why late interaction:** Protein function arises from sparse combinations of local motifs (binding sites, catalytic residues, conserved domains) at varying positions. ColBERT's token-level MaxSim scoring captures partial motif matches that mean pooling averages away.

**Architecture:**
- **Backbone:** ESM-2 (Meta's protein language model, 650M params)
- **Fine-tuning:** LoRA (rank 16) on query/value projections
- **Scoring:** ColBERT MaxSim — shared encoder → 64-dim projection → L2 norm → token-level matching
- **Training:** Multi-positive InfoNCE loss, τ=0.1, gradient accumulation (4 steps)

## Dataset: SCOPe

We use [SCOPe](https://scop.berkeley.edu/) (Structural Classification of Proteins — extended), the gold-standard hierarchical classification of protein structural domains.

- **Source:** ASTRAL 95% sequence identity filtered set (SCOPe 2.08)
- **Size:** 35,094 domains after length filtering (30–1022 residues)
- **Hierarchy:** 7 classes → 1,249 folds → 2,051 superfamilies → 4,988 families
- **Splits:** Fold-held-out (entire folds held out for val/test to prevent leakage)
- **Pairs:** Positives = same superfamily; hard negatives = same fold, different superfamily

| Split | Domains | Pairs |
|-------|---------|-------|
| Train | 23,038 | 359,889 |
| Val | 3,552 | 58,311 |
| Test | 8,504 | 148,760 |

## Commands

### Data Preparation

```bash
# Parse SCOPe FASTA + classification → manifest CSV
python -m src.data.parse_scope \
  --fasta data/raw/scope/astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa \
  --classification data/raw/scope/dir.cla.scope.2.08-stable.txt \
  --output data/interim/manifests/scope_manifest.csv

# Build retrieval pairs (superfamily positives, fold-level hard negatives)
python -m src.data.build_scope_pairs \
  --manifest data/interim/manifests/scope_manifest.csv \
  --output data/processed/scope/scope_pairs.csv
```

### Training

```bash
# Smoke test (ESM-2 35M, 1 epoch, fast validation)
CUDA_VISIBLE_DEVICES=0 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2_smoke.yaml

# Full training (ESM-2 650M ColBERT, 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2.yaml

# Resume from checkpoint
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2.yaml \
  --resume outputs/esm2_colbert/checkpoints/latest.pt

# Mean-pooled baseline (same backbone, no late interaction)
CUDA_VISIBLE_DEVICES=0,1 python -m src.train.train_stage_a \
  --config configs/train/stage_a_esm2_meanpool.yaml
```

Training logs every 100 steps. Use tmux for long runs:
```bash
tmux new-session -d -s train "PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1 \
  python -m src.train.train_stage_a --config configs/train/stage_a_esm2.yaml \
  2>&1 | tee outputs/esm2_train.log"
```

### Evaluation

```bash
# Pretrained ESM-2 (zero-shot, no fine-tuning)
CUDA_VISIBLE_DEVICES=0 python -m src.eval.eval_pretrained \
  --config configs/train/eval_esm2.yaml

# All classical baselines (random, k-mer, TF-IDF, MinHash, gkm-SVM, oracle)
python -m src.eval.run_all_baselines \
  --config configs/train/eval_esm2.yaml \
  --baselines random gc_matched_random tfidf_kmer minhash gkm_svm oracle
```

## Repository Structure

```
├── configs/
│   ├── model/
│   │   ├── esm2_colbert.yaml          # ESM-2 650M + ColBERT (main model)
│   │   ├── esm2_meanpool.yaml         # ESM-2 650M + mean pooling (baseline)
│   │   ├── esm2_colbert_small.yaml    # ESM-2 35M (smoke test)
│   │   └── dnabert2_*.yaml            # Legacy DNA configs
│   └── train/
│       ├── stage_a_esm2_smoke.yaml    # Protein smoke test config
│       └── stage_a_dnabert2_*.yaml    # Legacy DNA training configs
├── src/
│   ├── data/
│   │   ├── parse_scope.py             # SCOPe FASTA + classification → manifest
│   │   ├── build_scope_pairs.py       # Superfamily-based pair construction
│   │   ├── parse_screen.py            # Legacy: ENCODE SCREEN cCRE parsing
│   │   └── enrich_tf_activity.py      # Legacy: TF binding vector enrichment
│   ├── models/
│   │   ├── backbone.py                # ESM-2 / DNABERT-2 backbone with LoRA
│   │   ├── colbert.py                 # ColBERT late-interaction retriever
│   │   ├── meanpool.py                # Mean-pooled single-vector retriever
│   │   └── losses.py                  # Multi-positive InfoNCE loss
│   ├── train/
│   │   ├── train_stage_a.py           # Main training loop (checkpoint resume, logging)
│   │   └── trainer_utils.py           # Dataset, collator, model builder
│   ├── eval/
│   │   ├── eval_pretrained.py         # Zero-shot pretrained model evaluation
│   │   ├── run_all_baselines.py       # Full baseline suite runner
│   │   ├── eval_screen.py             # Legacy DNA evaluation
│   │   └── baselines.py              # Scoring functions (k-mer, random, etc.)
│   └── analysis/
│       └── tf_binding_analysis.py     # Legacy: TF binding EDA
├── figures/                           # EDA plots
├── report.md                          # Full research report
├── BASELINES.md                       # Baseline suite documentation
├── progress_report.tex                # ICML-style progress report (LaTeX)
└── progress_report.pdf                # Compiled progress report
```

## Hardware

Developed and tested on:
- 2x NVIDIA RTX 2080 Ti (11 GB each)
- 12 CPU cores, 125 GB RAM
- FP16 mixed precision (2080 Ti does not support BF16)

ESM-2 650M fits comfortably on a single 2080 Ti (~2 GB fp16). With LoRA + ColBERT head + batch: ~6-8 GB per GPU.

## Key Design Decisions

- **Fold-held-out splits:** Test folds are entirely unseen during training, preventing leakage from structurally similar domains on the same fold.
- **Hard negatives:** Same fold, different superfamily = structurally similar but not evolutionarily related. Forces the model to learn beyond gross structural similarity.
- **LoRA over full fine-tuning:** Preserves ESM-2's pretrained representations while adapting ~2M parameters for the retrieval task.
- **Gradient accumulation:** Stabilizes contrastive learning by averaging gradients over 4 batches (effective batch 80 queries).

## References

- SCOPe: Fox et al., "SCOPe: Structural Classification of Proteins -- extended," NAR, 2014
- ESM-2: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science, 2023
- ColBERT: Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR, 2020
