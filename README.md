# ProtoCol:  Late Interaction Retrieval for Protein Homolog Search

This repository contains the experiment code for ProtoCol, 
which performs ColBERT-style late-interaction retrieval for proteins. 
The current set of experiments instantiates ProtoCol with ESM-2 35M, along with
baselines (trained mean-pool single-vector, MinHash Jaccard, MMseqs2, and
frozen mean-pool ESM-2 650M) and a per-query latency benchmark.

## Requirements

- Python 3.10+
- A CUDA GPU is strongly recommended (the script runs on CPU but very slowly).

## Installation

1. (Recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```

2. Install the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Install MMseqs2 (needed for the MMseqs2 baseline). It is a **system binary**,
   not a pip package:

   ```bash
   sudo apt-get install -y mmseqs2
   ```

   or, with conda:

   ```bash
   conda install -c bioconda mmseqs2
   ```

   The script also tries to `apt-get install` it automatically if `mmseqs` is
   not on your `PATH`, but installing it ahead of time avoids needing sudo at
   run time.

The datasets (SCOPe / Pfam) are downloaded automatically on first run via
`curl`, so an internet connection is required the first time. Model weights are
pulled from the Hugging Face Hub on first use.

## Running the experiments

The dataset is selected by the `DATASET` constant near the top of
`protocol_experiments.py`:

```python
DATASET = "scope"   # "scope" or "pfam"
```

### Experiment 1 — SCOPe

Make sure the constant is set to `"scope"` (this is the default), then run:

```bash
python protocol_experiments.py
```

### Experiment 2 — Pfam

Edit the constant to `"pfam"`:

```python
DATASET = "pfam"
```

then run:

```bash
python protocol_experiments.py
```

To run both back-to-back without manually editing the file, you can override
the constant per run from the shell:

```bash
# SCOPe
python -c "import protocol_experiments as m; m.DATASET='scope'; m.main()"

# Pfam
python -c "import protocol_experiments as m; m.DATASET='pfam'; m.main()"
```

## What each run does

1. Downloads and parses the dataset (SCOPe superfamilies or Pfam clans as
   homology labels). This repo also contains a local copy of the SCOPe dataset
   in case the server is down.
2. Splits into train/test with **disjoint groups** and subsamples the test set
   (capped at `MAX_TEST_PROTEINS = 3000`) for tractable evaluation.
3. Fine-tunes the last 3 layers of ESM-2 35M with ColBERT MaxSim + InfoNCE
   (3 epochs for SCOPe, 1 epoch for Pfam).
4. Evaluates retrieval **Recall@{1, 5, 10, 100}** for ColBERT (untrained and
   trained) and all baselines.
5. Runs a **median per-query latency** benchmark across methods.

## Output

Everything is printed to stdout. Each run ends with two tables:

- A **retrieval summary** (Recall@k per method).
- A **combined latency + retrieval summary** (Recall@k plus median ms/query).

## Tunable settings

Edit the constants at the top of `protocol_experiments.py`:

| Constant | Default | Meaning |
|---|---|---|
| `DATASET` | `"scope"` | `"scope"` or `"pfam"` |
| `SEED` | `42` | Random seed |
| `MODEL_NAME` | `facebook/esm2_t12_35M_UR50D` | Backbone encoder |
| `ESM_650M` | `facebook/esm2_t33_650M_UR50D` | Large frozen baseline |
| `MAX_LEN` | `256` | Max tokenized sequence length |
| `MAX_TEST_PROTEINS` | `3000` | Test-set size cap |
| `KS` | `(1, 5, 10, 100)` | Recall@k cutoffs |
| `LATENCY_NUM_QUERIES` | `100` | Queries timed in the latency benchmark |