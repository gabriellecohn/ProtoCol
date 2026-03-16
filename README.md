# CN-Aware Diffusion Regulatory Model

A research model that discovers how somatic copy number alterations (CNAs) reshape distal regulatory interactions to modulate gene expression in cancer. The model uses a diffusion process over regulatory attention maps, conditioned on DNA sequence (via frozen Borzoi embeddings), copy number context, and observed expression.

## Core Idea

During **training**, the model solves an inverse problem: given observed expression, what regulatory configuration (attention over distal bins) explains it?

```
p(attention map | DNA, CN, expression)
```

During **inference**, expression conditioning is dropped, so the model generates its prior belief about regulatory configurations given only sequence and CN context:

```
p(attention map | DNA, CN)
```

Classifier-free guidance (CFG) enables interpolation between these distributions, supporting targeted queries like *"what regulatory maps explain unusually high expression for this CN context?"*

## Architecture

```
Frozen Borzoi (sequence encoder)
        │
        ▼
[Sequence embeddings per 2.5kb bin, ±250kb around TSS]
        │
        ├── CN dosage track (per-bin, from DepMap segmented CN)
        └── Expression value (scalar, training only — dropped at inference)
        │
        ▼
Conditional Diffusion Model (transformer denoiser, ~5–10M params)
   Generates: attention weight vector (promoter → 200 distal bins)
   Conditioned on: Borzoi embeddings + CN dosage + expression (with CFG dropout)
        │
        ▼
Expression Readout Head
   Weighted sum of attended sequence features → scalar expression
   Provides end-to-end training signal; attention maps are never directly supervised
```

## Project Structure

```
CompBio/
├── config.py                         # Config dataclass with all hyperparameters
├── requirements.txt
├── train.py                          # Two-stage training script
├── infer.py                          # Inference and analysis script
└── src/
    ├── data/
    │   ├── depmap_loader.py          # DepMap CN + expression loading
    │   ├── genome_utils.py           # GTF parsing, windowing, one-hot encoding
    │   └── preprocessing.py          # CN binning, expression residuals
    ├── models/
    │   ├── borzoi_encoder.py         # Borzoi feature extractor + HDF5 cache
    │   ├── denoiser.py               # CNAwareDenoiser (transformer + CFG)
    │   └── readout.py                # ExpressionReadout (attention pooling → MLP)
    ├── training/
    │   ├── warmup.py                 # Readout head warmup trainer
    │   └── diffusion_trainer.py      # DiffusionSchedule + DiffusionTrainer
    ├── inference/
    │   └── sampling.py               # DDPM/DDIM sampling, counterfactual, guided discovery
    └── validation/
        └── metrics.py                # Pearson r, expression evaluation, attention entropy
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Borzoi (sequence encoder)

```bash
git clone https://github.com/calico/borzoi.git
cd borzoi && pip install -e .
```

Download pretrained Borzoi weights following their README, then set `borzoi_model_path` in `config.py`. The pipeline works without Borzoi installed — it falls back to a `RandomEncoder` for debugging.

### 3. Download data

Place the following files in `data/`:

| File | Source |
|------|--------|
| `OmicsCNSegmentsProfile.csv` | [DepMap portal](https://depmap.org/portal/download/all/) → Copy Number |
| `OmicsExpressionProteinCodingGenesTPMLogp1.csv` | [DepMap portal](https://depmap.org/portal/download/all/) → Omics |
| `gencode.v41.annotation.gtf` | [Gencode](https://www.gencodegenes.org/human/release_41.html) |
| `hg38.fa` | [UCSC](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/) |

Use the **DepMap Public 24Q4** release (or latest available).

## Training

### Quick debug run (100 genes, 50 cell lines, no GPU required)

```bash
python train.py --debug
```

### Full training

```bash
python train.py --data-dir data/ --cache-dir cache/ --checkpoint-dir checkpoints/
```

### Resume from checkpoint

```bash
python train.py --resume
```

Training runs in two stages:

1. **Warmup** (~10 epochs): trains the readout head with random attention maps to initialize it before diffusion training begins.
2. **Diffusion training** (~100 epochs): jointly optimizes the denoiser and readout head with a combined DDPM loss + expression MSE loss. Expression loss weight is scheduled from 0.1 → 1.0 over training.

Borzoi features are pre-computed once and cached to `cache/borzoi_features.h5`.

Train/val/test split is by chromosome (chr1–17 / chr18–20 / chr21–22) to avoid data leakage from nearby genes.

## Inference

### Unconditional sampling

Sample plausible regulatory configurations for a given gene and CN context:

```bash
python infer.py --gene BRCA1 --mode unconditional
```

### Expression-guided discovery

Discover what regulatory configurations would explain high (or low) expression:

```bash
python infer.py --gene MYC --mode guided --target-expr 2.5 --guidance-scale 3.0
```

### Counterfactual CN analysis

Compare regulatory attention maps between diploid and amplified CN:

```bash
python infer.py --gene EGFR --mode counterfactual --cn-amplified 8
```

### Fast sampling (DDIM, ~20× speedup)

```bash
python infer.py --gene TP53 --mode unconditional --fast
```

Results are saved as JSON to `results/`.

## Key Design Decisions

- **Expression residuals as targets**: A per-gene linear regression of CN → expression is fit across cell lines; the residuals are used as training targets. This forces the model to learn non-linear and interaction effects rather than rediscovering the trivial dosage relationship.

- **CFG dropout (10%)**: During training, expression conditioning is randomly zeroed, so the model simultaneously learns the conditional and unconditional distributions. At inference, these are interpolated via guidance scale.

- **No supervision on attention maps**: The regulatory attention maps are purely latent structure — they emerge because they are the only way to explain expression given the sequence and CN context.

- **Chromosome-based splitting**: Prevents data leakage between genes in nearby genomic windows.

## Compute Estimates

| Step | Estimate |
|------|----------|
| Borzoi feature extraction (~18k genes) | ~2–4 GPU-hours (A100), cached once |
| Warmup (Stage 1) | ~1–2 GPU-hours |
| Diffusion training (Stage 2) | ~10–20 GPU-hours (A100) |
| Dataset size | ~16M examples (900 cell lines × 18k genes) |
| Model size | ~5–10M parameters |

## Validation

The following experiments are implemented in `src/validation/metrics.py`:

1. **Expression prediction accuracy** — Pearson r vs. linear CN baseline and null model, evaluated on held-out chromosomes.
2. **Guided vs. unguided coherence** — does expression conditioning shift attention toward known regulatory elements?
3. **Attention map interpretability** — overlap with ENCODE/Roadmap enhancers; cluster analysis by tissue type.
4. **Non-additivity detection** — co-amplified gene pairs vs. sum of individual CN effects.
5. **Counterfactual sanity checks** — diploid CN → normal expression; amplified enhancer → increased expression.
6. **Distribution quality** — does sampled attention variance correlate with empirical expression variance across cell lines?
