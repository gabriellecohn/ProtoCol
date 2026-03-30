# CompBio `mvp-spec` Branch

This branch contains two related but distinct research codepaths that currently coexist in one repository:

1. A newer cCRE retrieval MVP centered on DNA representation learning and ColBERT-style ranking.
2. An older CN-aware diffusion model for regulatory map generation and expression modeling.

The retrieval MVP is the active focus of this branch. The diffusion code is still present and runnable, but the repository is not yet fully reorganized around a single project layout.

## Current State

The repository is in a transitional but runnable state.

What works now:
- Stage A retrieval training and evaluation code under `src/train`, `src/eval`, `src/models`, and `src/data`.
- A smoke-test pipeline for quick end-to-end validation on synthetic data.
- A larger synthetic Stage A pipeline for scaling beyond the smoke test without requiring external datasets.
- SLURM submission helpers for cluster runs.
- The legacy diffusion `train.py` and `infer.py` entrypoints.

What is scaffolded but not bundled with the repo:
- Full real-data retrieval runs using screen-derived manifests and pair files.
- DNABERT-2-backed retrieval runs, which require model download and a compatible runtime environment.
- Real DepMap/Gencode/UCSC inputs for the diffusion pipeline.

Important practical note:
- The repository `.gitignore` excludes generated `data/`, `outputs/`, checkpoints, and CSV artifacts. This means most experiment products are intentionally not tracked in git.

## Repository Layout

```text
compbio/
├── MVP_SPEC.md                      # Retrieval MVP specification for this branch
├── README.md
├── pyproject.toml                   # Python package metadata, requires Python >= 3.10
├── requirements.txt
├── run.sh                           # Local convenience wrapper for the main workflows
├── run_slurm.sh                     # SLURM batch wrapper
├── train.py                         # Legacy CN-aware diffusion training entrypoint
├── infer.py                         # Legacy CN-aware diffusion inference entrypoint
├── config.py                        # Legacy diffusion config dataclass
├── configs/
│   ├── data/
│   │   └── screen.yaml              # Chromosome split defaults used by split assignment
│   ├── model/
│   │   ├── dnabert2_colbert.yaml
│   │   ├── dnabert2_meanpool.yaml
│   │   ├── hash_dna_smoke.yaml
│   │   └── hash_dna_synthetic_medium.yaml
│   └── train/
│       ├── stage_a.yaml
│       ├── stage_a_smoke.yaml
│       ├── stage_a_synthetic_medium.yaml
│       ├── eval.yaml
│       ├── eval_smoke.yaml
│       └── eval_synthetic_medium.yaml
├── scripts/
│   ├── smoke_test.py                # Tiny synthetic end-to-end Stage A run
│   ├── run_synthetic_stage_a.py     # Larger synthetic Stage A experiment
│   ├── run_stage_a.sh               # Minimal Stage A launcher
│   └── run_eval.sh                  # Minimal eval launcher
└── src/
    ├── data/                        # Parsing, sequence extraction, split and pair building
    ├── eval/                        # Retrieval evaluation and baselines
    ├── inference/                   # Legacy diffusion sampling helpers
    ├── models/                      # Retrieval backbones / retrievers + diffusion models
    ├── train/                       # Stage A training logic
    ├── training/                    # Legacy diffusion training logic
    ├── utils/                       # IO, logging, metrics, genome utilities
    └── validation/                  # Legacy diffusion validation utilities
```

## Retrieval MVP: Experiment Design

The retrieval side of the repo is designed around a Stage A candidate-ranking problem over cCRE-like DNA sequences.

### High-level goal

Given a query regulatory element, rank other elements that should be considered biologically similar or relevant. In the current MVP, the repository focuses on sequence-based retrieval with weak supervision.

### Data flow

The intended real-data pipeline is:

```bash
python -m src.data.parse_screen
python -m src.data.extract_sequences --window 1024
python -m src.data.split_by_chromosome
python -m src.data.build_stage_a_pairs
python -m src.train.train_stage_a --config configs/train/stage_a.yaml
python -m src.eval.eval_screen --config configs/train/eval.yaml --baseline model --checkpoint <ckpt>
```

Conceptually, the pipeline does the following:
- Parse a screen-derived manifest of candidate regulatory elements.
- Attach sequence windows such as `sequence_512`, `sequence_1024`, and `sequence_2048`.
- Assign chromosome-held-out train/val/test splits.
- Build weakly supervised query-document pairs.
- Train a retriever.
- Evaluate retrieval quality against positive pairs and simple baselines.

### Weak supervision strategy

`src.data.build_stage_a_pairs` constructs positives and negatives using fields like:
- `ccre_class`
- `activity_vector`
- `biosample_group`
- GC-content bins
- chromosome split

Positives are sampled from same-class or group-matched examples with sufficient activity overlap or Jaccard similarity. Negatives are drawn from matched-GC, hard-negative, and random pools within split.

### Models in this branch

There are currently two retrieval backbone regimes:

1. Debug / synthetic backbone
- Configs: `configs/model/hash_dna_smoke.yaml`, `configs/model/hash_dna_synthetic_medium.yaml`
- Uses `model_name: hash-dna-debug`
- This resolves to a tiny learned embedding backbone implemented locally in `src/models/backbone.py`
- Useful for validating the training pipeline quickly without external downloads

2. DNABERT-2 backbone
- Configs: `configs/model/dnabert2_colbert.yaml`, `configs/model/dnabert2_meanpool.yaml`
- Uses Hugging Face model `zhihan1996/DNABERT-2-117M`
- Intended for more realistic retrieval experiments
- Requires model download, `transformers`, and a working environment for remote model loading

### Training objective

Stage A training uses grouped query / positive / negative batches and optimizes a multi-positive softmax-style loss over retrieval scores. Evaluation during training reports validation loss and validation MRR.

## Legacy Diffusion Model

The older diffusion pipeline is still available in this branch.

It models regulatory attention maps conditioned on:
- DNA sequence-derived features
- copy number context
- expression values during training

Core legacy entrypoints:
- `python train.py --debug`
- `python train.py --data-dir data/ --cache-dir cache/ --checkpoint-dir checkpoints/`
- `python infer.py --gene MYC --mode guided --target-expr 2.5`

The diffusion path expects real DepMap, Gencode, and genome FASTA inputs that are not bundled in the repository.

## Environment and Dependencies

### Python

The project metadata in `pyproject.toml` requires:

```text
Python >= 3.10
```

In practice, the current cluster runs were executed with Miniforge Python 3.12.

### Install

For the retrieval pipeline:

```bash
pip install -r requirements.txt
```

If you want to use the DNABERT-2 configs, make sure the environment can import:
- `torch`
- `transformers`
- `peft`

For the legacy diffusion workflow, optional Borzoi integration is still described in the code comments and README history, but it is not required for the synthetic retrieval runs.

## How To Run

### 1. Local smoke test

This is the fastest way to confirm the retrieval pipeline is wired correctly.

```bash
cd /home/rgumaste/bio_proj/compbio
./run.sh smoke
```

This will:
- generate a tiny synthetic manifest
- build weakly supervised pairs
- train the Stage A model for one epoch
- run model, k-mer, and random evaluation

### 2. Larger synthetic retrieval run

This scales beyond the smoke test without needing real screen data.

```bash
cd /home/rgumaste/bio_proj/compbio
./run.sh synthetic
```

Useful options:

```bash
./run.sh synthetic --examples-per-class 64
./run.sh synthetic --examples-per-class 96
./run.sh synthetic --skip-eval
```

The default synthetic medium config currently writes to:
- `outputs/synthetic_medium/`
- `data/interim/manifests/synthetic_medium_manifest.csv`
- `data/processed/stage_a/synthetic_medium_pairs.csv`

### 3. Real-data Stage A training

If you have already prepared real manifests and pair files matching the config paths:

```bash
cd /home/rgumaste/bio_proj/compbio
python -m src.train.train_stage_a --config configs/train/stage_a.yaml
python -m src.eval.eval_screen --config configs/train/eval.yaml --baseline model --checkpoint outputs/checkpoints/stage_a_best.pt
```

This path is not turnkey in a fresh clone because the required screen-derived data files are not committed.

### 4. SLURM runs

The repository includes a batch wrapper that delegates into `run.sh`.

Default smoke run:

```bash
cd /home/rgumaste/bio_proj/compbio
sbatch run_slurm.sh
```

Scaled synthetic run:

```bash
cd /home/rgumaste/bio_proj/compbio
sbatch --job-name=compbio_synth64 --export=ALL,RUN_TARGET=synthetic run_slurm.sh --examples-per-class 64
```

You can switch the entrypoint using `RUN_TARGET`:
- `smoke`
- `synthetic`
- `stage-a`
- `eval`
- `train`
- `infer`

Useful scheduler commands:

```bash
squeue -j <job_id>
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

### 5. Legacy diffusion examples

```bash
cd /home/rgumaste/bio_proj/compbio
python train.py --debug
python infer.py --gene BRCA1 --mode unconditional
python infer.py --gene MYC --mode guided --target-expr 2.5
```

## Current Experiments and Observations

The branch currently has three practical experiment tiers.

### Smoke retrieval run

Purpose:
- Verify that the Stage A retrieval pipeline runs end to end.

Characteristics:
- Tiny synthetic manifest.
- CPU-friendly.
- Fast enough to use as a regression check.

### Synthetic medium retrieval run

Purpose:
- Stress the Stage A training and evaluation loop beyond the smoke test.
- Exercise SLURM submission and checkpoint/eval generation.

Characteristics:
- Uses the local `hash-dna-debug` backbone, not DNABERT-2.
- Current config: `configs/train/stage_a_synthetic_medium.yaml`
- Current model size is tiny compared with DNABERT-scale experiments.

Observed behavior so far:
- The run completes quickly and reliably on SLURM.
- Training loss falls sharply across epochs.
- Retrieval quality remains weak relative to the k-mer baseline on the synthetic dataset.
- This suggests the current synthetic data distribution is easy for lexical baselines and is not yet a strong proxy for the intended real retrieval task.

### Real-data retrieval path

Purpose:
- Train the intended MVP with screen-derived examples and DNABERT-2 or similar backbones.

Current limitation:
- The repository does not include the real screen manifests, pair tables, or downloaded model weights needed for a meaningful full run.

## Outputs and Artifacts

By default, experiment outputs are written under `outputs/`, including:
- checkpoints
- training summaries
- evaluation JSON metrics

Synthetic and smoke data products are written under `data/`, but those are ignored by git.

Representative output locations:
- `outputs/smoke/metrics/`
- `outputs/synthetic_medium/metrics/`
- `outputs/synthetic_medium/checkpoints/`

## Recommended Next Steps

If you are continuing development on this branch, the highest-value next steps are:
- Add or document the real screen-derived manifest format expected by `src.data.parse_screen` and downstream steps.
- Run the DNABERT-2 Stage A pipeline once the required model download path and data are available.
- Revisit the synthetic data design so that learned retrieval has a chance to outperform the k-mer baseline.
- Decide whether the legacy diffusion code should remain in this repository or be split into its own project.

## Quick Reference

Local wrappers:

```bash
./run.sh smoke
./run.sh synthetic --examples-per-class 64
./run.sh train --debug
./run.sh infer --gene EGFR --mode counterfactual --cn-amplified 8
```

Minimal direct entrypoints:

```bash
python -m src.train.train_stage_a --config configs/train/stage_a_smoke.yaml
python -m src.eval.eval_screen --config configs/train/eval_smoke.yaml --baseline kmer
python -m src.train.train_stage_a --config configs/train/stage_a_synthetic_medium.yaml
python -m src.eval.eval_screen --config configs/train/eval_synthetic_medium.yaml --baseline model --checkpoint outputs/synthetic_medium/checkpoints/synthetic_medium_stage_a_best.pt
```
