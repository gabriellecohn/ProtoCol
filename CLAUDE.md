# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two coexisting research pipelines:

1. **DNA ColBERT MVP** (active focus): ColBERT-style late-interaction retrieval over SCREEN/ENCODE cCREs (candidate cis-Regulatory Elements). Goal: recover biologically similar regulatory elements better than simple baselines.
2. **Legacy CN-aware diffusion model** (secondary): regulatory attention map generation conditioned on DNA sequence, copy number, and expression values.

The retrieval MVP is the active work. The diffusion code is present and runnable but the repo is not yet reorganized around a single layout.

## Commands

### Retrieval MVP

```bash
# Smoke test (tiny synthetic dataset, CPU-friendly, ~1 epoch)
./run.sh smoke

# Larger synthetic run (no real data required)
./run.sh synthetic
./run.sh synthetic --examples-per-class 64
./run.sh synthetic --skip-eval

# Real-data pipeline (requires prepared SCREEN manifests and pairs)
python -m src.train.train_stage_a --config configs/train/stage_a.yaml
python -m src.eval.eval_screen --config configs/train/eval.yaml --baseline model --checkpoint outputs/checkpoints/stage_a_best.pt

# Evaluation with a specific checkpoint
python -m src.eval.eval_screen --config configs/train/eval_synthetic_medium.yaml --baseline model \
  --checkpoint outputs/synthetic_medium/checkpoints/synthetic_medium_stage_a_best.pt

# Evaluation with baselines only
python -m src.eval.eval_screen --config configs/train/eval_smoke.yaml --baseline kmer
python -m src.eval.eval_screen --config configs/train/eval_smoke.yaml --baseline random
```

### Real-data preprocessing (sequential)

```bash
python -m src.data.parse_screen
python -m src.data.extract_sequences --window 1024
python -m src.data.split_by_chromosome
python -m src.data.build_stage_a_pairs
```

### Legacy diffusion

```bash
python train.py --debug
python train.py --data-dir data/ --cache-dir cache/ --checkpoint-dir checkpoints/
python infer.py --gene MYC --mode guided --target-expr 2.5
python infer.py --gene BRCA1 --mode unconditional
```

### SLURM

```bash
sbatch run_slurm.sh                                                        # smoke (default)
sbatch --job-name=compbio_synth64 --export=ALL,RUN_TARGET=synthetic run_slurm.sh --examples-per-class 64
# RUN_TARGET options: smoke, synthetic, stage-a, eval, train, infer
```

## Architecture

### Retrieval MVP Pipeline

```
SCREEN manifest → extract sequences → chromosome splits → query-doc pairs → Stage A training → evaluation
```

**Data** (`src/data/`): `download_screen.py` → `parse_screen.py` → `extract_sequences.py` → `split_by_chromosome.py` → `build_stage_a_pairs.py`. Splits are chromosome-held-out (val: chr2/chr10, test: chr1/chr8/chr21). Pairs use weak supervision from `ccre_class`, `activity_vector`, `biosample_group`, and GC-content bins.

**Models** (`src/models/`):
- `backbone.py`: factory for DNA encoders; `hash-dna-debug` is a tiny local model for validation without downloads, `zhihan1996/DNABERT-2-117M` is the intended backbone.
- `colbert.py`: ColBERT late-interaction model — shared encoder, linear projection to 64-dim token embeddings, L2 norm, MaxSim scoring.
- `meanpool.py`: mean-pooled baseline using the same backbone.
- `losses.py`: multi-positive softmax loss.

**Training** (`src/train/`): `train_stage_a.py` is the main training loop; `trainer_utils.py` handles dataset construction, model building, and collation. Checkpoints selected by validation MRR.

**Evaluation** (`src/eval/`): `eval_screen.py` reports Recall@k, MRR, nDCG, top-k cCRE class purity, biosample activity Jaccard/correlation. `baselines.py` implements k-mer (k=5/6 Jaccard) and random baselines.

**Indexing** (`src/index/`): `encode_corpus.py` → `build_faiss.py` → `search.py` for FAISS-backed retrieval at scale.

### Configuration System

All experiments are YAML-driven from `configs/`:
- `configs/model/hash_dna_smoke.yaml`, `hash_dna_synthetic_medium.yaml` — debug backbone (no download)
- `configs/model/dnabert2_colbert.yaml`, `dnabert2_meanpool.yaml` — DNABERT-2 configs
- `configs/train/stage_a_smoke.yaml`, `stage_a_synthetic_medium.yaml`, `stage_a.yaml` — training configs
- `configs/train/eval_smoke.yaml`, `eval_synthetic_medium.yaml`, `eval.yaml` — eval configs

### Legacy Diffusion

`train.py` / `infer.py` at root are entrypoints. Implementation lives in `src/training/`, `src/inference/`, `src/validation/`, `src/models/` (denoiser, readout, borzoi_encoder). Requires real DepMap/Gencode/FASTA inputs not bundled in the repo.

## Key Practical Notes

- `data/`, `outputs/`, checkpoints, and CSV artifacts are excluded from git. Generated files are intentionally untracked.
- The `hash-dna-debug` backbone is always available without downloads; DNABERT-2 requires `transformers`/`peft` and model download.
- On synthetic data, the k-mer baseline currently outperforms the learned model — this is expected and indicates the synthetic distribution is too easy for lexical methods.
- Target hardware: single Nvidia L40S. Engineering defaults: bf16 autocast, gradient checkpointing, LoRA, gradient accumulation.
- Python ≥ 3.10 required; cluster runs use Miniforge Python 3.12.
- Install: `pip install -r requirements.txt`
