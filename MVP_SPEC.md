# MVP Spec: DNA ColBERT on cCRE Retrieval

This branch captures the narrowed MVP scope for a planned `dna-colbert-mvp/`
implementation. The current repository does not yet match the target layout
below; this document is the source-of-truth product spec for the first
retrieval-only version.

## Goal

Build a minimal, working retrieval system for regulatory DNA sequences using a
ColBERT-style late-interaction model over SCREEN/ENCODE cCREs.

This MVP should answer one question:

Can a late-interaction DNA retriever recover biologically similar regulatory
elements better than simple baselines?

The MVP is only about cCRE-to-cCRE retrieval.

Do not implement enhancer-promoter interaction modeling, BENGI, or any
multi-stage fine-tuning.

## Scientific framing

The claim for this MVP is intentionally narrow:

- We are trying to learn functional regulatory similarity from sequence.
- We are not claiming sequence alone fully determines regulatory function.
- We are not trying to predict enhancer-promoter interactions.
- We are testing whether retrieval embeddings can capture:
  - broad cCRE class similarity
  - similar biosample activity patterns
  - potentially similar regulatory grammar

## Target hardware

- 1x Nvidia L40S

Design everything to run on a single GPU.

## Success criteria

The MVP is successful if:

1. The repo can download and preprocess SCREEN/cCRE data.
2. The repo can extract fixed-length DNA sequences from hg38.
3. A ColBERT-style model can be trained on one L40S.
4. Evaluation runs end-to-end and reports:
   - Recall@k
   - MRR
   - nDCG
   - top-k cCRE class purity
   - top-k biosample activity similarity
5. The ColBERT model beats:
   - random retrieval
   - simple k-mer baseline
   - mean-pooled DNABERT baseline

## Non-goals

Do not implement the following in the MVP:

- BENGI
- promoter to enhancer retrieval
- DeepSEA training
- cross-assembly harmonization
- multi-GPU training
- megabase-scale context
- 3D genome features
- chromatin contact modeling
- full TF-binding integration unless trivial to add

## Fixed design decisions

These are the defaults unless there is a strong reason to change them.

### Dataset

Use human SCREEN/ENCODE cCREs in hg38 as the only core dataset.

### Sequence window

Use 1024 bp centered at the cCRE midpoint.

Also support:

- 512 bp
- 2048 bp

for ablations only.

### Backbone

Use:

- `zhihan1996/DNABERT-2-117M`

### Split policy

Use chromosome-held-out splits, not random splits.

Suggested default:

- train: all chromosomes except val/test
- val: `chr2`, `chr10`
- test: `chr1`, `chr8`, `chr21`

### Data scale

Create three tiers:

- `debug_subset`: about 10k cCREs
- `small_subset`: about 50k cCREs
- `mvp_subset`: about 200k to 300k cCREs

## Required repo structure

```text
dna-colbert-mvp/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── data/
│   │   └── screen.yaml
│   ├── model/
│   │   ├── dnabert2_colbert.yaml
│   │   └── dnabert2_meanpool.yaml
│   └── train/
│       ├── stage_a.yaml
│       └── eval.yaml
├── data/
│   ├── raw/
│   │   ├── screen/
│   │   └── refs/
│   │       └── hg38/
│   ├── interim/
│   │   ├── manifests/
│   │   ├── fasta_cache/
│   │   └── qc/
│   └── processed/
│       └── stage_a/
├── src/
│   ├── data/
│   │   ├── download_screen.py
│   │   ├── parse_screen.py
│   │   ├── extract_sequences.py
│   │   ├── build_stage_a_pairs.py
│   │   └── split_by_chromosome.py
│   ├── models/
│   │   ├── backbone.py
│   │   ├── colbert.py
│   │   ├── meanpool.py
│   │   └── losses.py
│   ├── train/
│   │   ├── train_stage_a.py
│   │   └── trainer_utils.py
│   ├── index/
│   │   ├── encode_corpus.py
│   │   ├── build_faiss.py
│   │   └── search.py
│   ├── eval/
│   │   ├── baselines.py
│   │   ├── eval_screen.py
│   │   └── interpretability.py
│   └── utils/
│       ├── io.py
│       ├── genome.py
│       ├── metrics.py
│       ├── logging.py
│       └── seed.py
├── scripts/
│   ├── run_stage_a.sh
│   └── run_eval.sh
├── outputs/
│   ├── checkpoints/
│   ├── indexes/
│   ├── metrics/
│   └── figures/
└── notebooks/
    ├── 01_dataset_qc.ipynb
    ├── 02_retrieval_examples.ipynb
    └── 03_interpretability.ipynb
```

## Required data products

Build a manifest with one row per cCRE.

### Required columns

- `ccre_id`
- `chrom`
- `start`
- `end`
- `assembly`
- `ccre_class`
- `midpoint`
- `sequence_512`
- `sequence_1024`
- `sequence_2048`
- `gc_content`
- `activity_vector`
- `activity_count`
- `split`

### Optional columns

Only add if easy:

- `tf_binding_vector`
- `biosample_group`
- `length`

## Pair construction for training

This is the core of the MVP.

### Query

A cCRE sequence.

### Positive examples

A positive should satisfy at least one of:

1. same `ccre_class` and high biosample activity similarity
2. same dominant biosample group
3. same class with overlapping activity support

Use a simple threshold-based heuristic.

### Negative examples

Sample negatives from:

1. same GC bin, low activity similarity
2. same class but clearly different activity profile
3. different class with similar length or GC
4. random negatives

After the first training epoch, add hard negatives mined from current retrieval
results.

### Recommended counts

Per query:

- positives: 1 to 4
- negatives: 15 to 31

## Baselines

Implement these baselines first.

### 1. Random baseline

Return random cCREs.

### 2. k-mer baseline

Implement a lightweight k-mer similarity baseline:

- `k = 5` or `6`
- Jaccard or cosine over k-mer count vectors

### 3. Mean-pooled encoder baseline

Use the exact same DNABERT-2 backbone as ColBERT, but:

- mean-pool token embeddings
- score with cosine similarity or dot product

This baseline is mandatory.

## Model requirements

### A. ColBERT retrieval model

Implement:

- shared query/document encoder
- DNABERT-2 backbone
- linear projection from token hidden states to low-dim token embeddings
- L2 normalization
- MaxSim late interaction
- `score(query, doc) = sum over query tokens of max token similarity in doc`

### Default settings

- projection dim: 64
- bf16 mixed precision
- gradient checkpointing enabled
- LoRA enabled by default
- gradient accumulation enabled
- in-batch negatives enabled

### B. Mean-pool baseline

Implement with the same encoder backbone and tokenizer.

## Training objective

Use a retrieval-style loss.

Preferred options:

- in-batch contrastive loss
- listwise softmax loss
- pairwise ranking loss

The implementation should support multiple positives and many negatives per
query.

Default training behavior:

- use in-batch negatives
- allow hard negatives after epoch 1
- select checkpoint by validation MRR

## Training schedule

Run in this exact order.

### Phase 1: debug

- dataset: 10k cCREs
- goal: verify pipeline, loss decreases, evaluation runs

### Phase 2: small

- dataset: 50k cCREs
- goal: check baselines and memory/runtime

### Phase 3: MVP

- dataset: 200k to 300k cCREs
- goal: final metrics and retrieval examples

Save:

- best checkpoint by validation MRR
- best checkpoint by biological metric composite

## Biological evaluation

This is the key deliverable.

### Retrieval metrics

Report:

- Recall@1
- Recall@5
- Recall@10
- Recall@50
- MRR
- nDCG

### Biological metrics

Report:

- top-k cCRE class purity
- mean top-k biosample activity Jaccard
- mean top-k biosample activity correlation

At minimum, evaluate on:

- `k = 5`
- `k = 10`
- `k = 50`

### Required comparisons

Compare:

- random
- k-mer baseline
- mean-pooled DNABERT
- ColBERT

## Interpretability

Implement a minimal interpretability module.

Save at least:

- one retrieval success case
- one retrieval failure case
- one example where mean-pooling fails but ColBERT succeeds

Artifacts should include:

- query sequence ID
- top retrieved IDs
- token-level similarity heatmap or local match visualization

Save to:

- `outputs/figures/`

## Required ablations

Only run these three MVP ablations:

1. Late interaction vs mean pooling
2. Weak labels: class only vs class + activity
3. Window size: 512 vs 1024 vs 2048

Do not expand beyond these until the base MVP works.

## L40S constraints

The code must be written for one L40S.

### Required engineering choices

- bf16 autocast
- gradient checkpointing
- LoRA or partial fine-tuning
- gradient accumulation
- cached tokenization
- FAISS indexing in chunks

### Avoid

- full end-to-end backbone training without memory safeguards
- unnecessarily large corpora during debugging
- long-sequence training by default

## Implementation order

Build in this order.

### Step 1: scaffold

1. create repo structure
2. add configs
3. add logging, seed control, checkpoint utils

### Step 2: data

4. implement SCREEN download/parsing
5. implement hg38 FASTA extraction
6. create cCRE manifest
7. create chromosome-held-out splits
8. create debug/small/MVP subsets

### Step 3: baselines

9. implement random baseline
10. implement k-mer baseline
11. implement mean-pooled DNABERT baseline

### Step 4: ColBERT

12. implement ColBERT model
13. implement training pair generator
14. run 10k debug experiment
15. run 50k experiment
16. run 200k to 300k MVP experiment

### Step 5: evaluation

17. run full retrieval evaluation
18. compute biological metrics
19. export tables and figures
20. export qualitative retrieval examples

### Step 6: ablations

21. run required ablations
22. save comparison plots

## CLI requirements

At minimum, support these commands:

```bash
python -m src.data.download_screen
python -m src.data.parse_screen
python -m src.data.extract_sequences --window 1024
python -m src.data.build_stage_a_pairs
python -m src.train.train_stage_a --config configs/train/stage_a.yaml
python -m src.eval.eval_screen --checkpoint outputs/checkpoints/best.pt
```

Also provide shell scripts in `scripts/`.

## Deliverables

The agent should produce:

### Code

- full modular repo

### Data artifacts

- parsed cCRE manifest
- sequence cache
- split files
- train pair tables

### Model artifacts

- best ColBERT checkpoint
- best mean-pool checkpoint
- FAISS index for retrieval

### Metrics

- CSV/JSON summaries in `outputs/metrics/`

### Figures

- baseline comparison plots
- retrieval examples
- token-level similarity visualizations

### Documentation

A `README.md` that explains:

- environment setup
- dataset download
- preprocessing
- training
- evaluation
- expected outputs

## Acceptance tests

The MVP is complete when:

1. `README.md` is sufficient for a fresh user to run the pipeline.
2. The 10k debug run completes on one L40S.
3. The 50k run completes on one L40S.
4. The 200k to 300k MVP run completes on one L40S.
5. Evaluation reports retrieval metrics and biological metrics.
6. Random, k-mer, mean-pool, and ColBERT baselines all run.
7. At least one interpretability figure is generated.
8. The required three ablations are completed.

## Final instruction

Keep the first version narrow and robust.

Do not add BENGI, DeepSEA training, or extra biological tasks until the cCRE
retrieval MVP is complete. The main result should be a clean demonstration that
a ColBERT-style late-interaction model can retrieve biologically similar
regulatory elements better than simpler baselines on SCREEN-derived cCRE data.
