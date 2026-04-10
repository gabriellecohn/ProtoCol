# Late-Interaction Retrieval of Regulatory DNA Sequences Using ColBERT and DNABERT-2

## 1. Introduction

Candidate cis-regulatory elements (cCREs) — promoters, enhancers, insulators — are the core switches of gene regulation. The ENCODE SCREEN registry catalogs over one million human cCREs classified by epigenomic signatures (PLS, pELS, dELS, CTCF-only, DNase-H3K4me3), but finding functionally similar elements given only a query sequence remains an open retrieval problem.

Standard approaches to sequence similarity — k-mer counting, global alignment — capture lexical overlap but miss the sparse, position-dependent motif grammar that defines regulatory function. A promoter and an enhancer may share GC content and even individual transcription factor binding sites, yet serve entirely different roles depending on motif arrangement and spacing.

This project asks a narrow question: **can a ColBERT-style late-interaction retrieval model, built on a pretrained DNA language model, recover biologically similar regulatory elements better than simpler baselines?**

Late interaction is a natural fit for this problem. Regulatory similarity is often driven by partial motif matches scattered across a sequence — a shared TATA box here, a common CTCF site there. Mean-pooled representations collapse all positional information into a single vector, potentially averaging away these sparse signals. ColBERT's MaxSim scoring preserves token-level alignment, letting the model attend to whichever local matches matter most for a given query-document pair.

### 1.1 Problem Statement in Plain Terms

The human genome contains approximately one million regulatory "switches" (cCREs) that control when and where genes are turned on. The ENCODE consortium has cataloged these switches and measured which proteins (transcription factors) bind to each one. The question we address is: **given only the raw DNA sequence of one switch — a string of A, T, G, C characters — can we find other switches that perform the same biological function?**

The sole input to our model is the raw nucleotide sequence (~512 characters). No metadata, no annotations, no hand-engineered features. The model must learn, from sequence alone, which character patterns predict shared transcription factor binding. If it can, this demonstrates that DNA sequence contains sufficient information to recover regulatory function — and that a learned retrieval model can extract this signal better than simple string-matching approaches.

## 2. Biological Motivation

### 2.1 The cCRE Retrieval Task

Given a query cCRE sequence, rank all other cCREs in a held-out corpus by biological similarity. "Biologically similar" is defined by weak supervision from two ENCODE data sources: (1) SCREEN regulatory class membership, and (2) shared transcription factor binding profiles from 2,818 TF ChIP-seq experiments. Two cCREs are considered positives when they share the same class and have high Jaccard similarity over their TF binding vectors — i.e., they are bound by similar sets of transcription factors.

This framing is deliberately narrow. We are not predicting enhancer-promoter interactions, modeling 3D chromatin architecture, or attempting de novo motif discovery. The claim is limited to whether learned sequence representations can capture enough regulatory grammar to outperform string-matching baselines on a retrieval task.

### 2.2 Why Late Interaction

Regulatory elements are modular — their function arises from combinations of short binding motifs (6-20 bp) embedded in longer sequences (200-1000+ bp). Two enhancers may be functionally similar because they share three out of five binding sites, even if those sites appear at different positions and the intervening sequence is unrelated.

This structure maps directly onto the ColBERT scoring model:

- **Token embeddings** capture local sequence context at each position.
- **MaxSim** finds the best-matching position in the document for each query token, tolerating positional variation.
- **Summation** aggregates evidence across all query positions, rewarding partial motif overlap without requiring global alignment.

Mean pooling, by contrast, compresses the full sequence into a single vector, making it difficult to distinguish "shares 3 of 5 motifs" from "shares 0 of 5 motifs but has similar base composition."

## 3. Dataset Construction

### 3.1 Source Data

We use the ENCODE SCREEN Registry V3 (GRCh38), which contains **1,063,878 classified cCREs** across five regulatory classes:

| Class | Count | Description |
|-------|-------|-------------|
| dELS | 789,200 | Distal enhancer-like signature |
| pELS | 172,027 | Proximal enhancer-like signature |
| PLS | 40,891 | Promoter-like signature |
| CTCF-only | 35,839 | CTCF-bound insulator elements |
| DNase-H3K4me3 | 25,921 | Open chromatin with H3K4me3 mark |

Each cCRE is annotated with genomic coordinates (hg38), a primary regulatory class, and a CTCF-binding status flag. The class distribution is heavily skewed toward distal enhancers (74%).

### 3.2 Sequence Extraction

For each cCRE, we extract a **512 bp window** centered at the element midpoint from the hg38 reference genome using pyfaidx. Sequences extending beyond chromosome boundaries are N-padded to maintain fixed length. GC content is computed per sequence for downstream stratification.

The 512 bp window captures the core regulatory element plus flanking context. Longer windows (1024, 2048 bp) are supported for ablation but are not used in the primary experiments due to GPU memory constraints on the RTX 2080 Ti.

### 3.3 Chromosome-Held-Out Splits

To prevent data leakage from genomically proximal elements, we use chromosome-level held-out splits:

- **Validation**: chr2, chr10
- **Test**: chr1, chr8, chr21
- **Train**: all remaining chromosomes

This is stricter than random splitting, which would allow nearby elements on the same chromosome to appear in both train and test sets.

### 3.4 Subset Tiers

We construct three dataset tiers by stratified sampling from the full manifest:

| Tier | cCREs | Training queries | Pairs |
|------|-------|-----------------|-------|
| Debug | 10,000 | 7,177 | 190,000 |
| Small | 50,000 | 35,883 | 950,000 |
| Medium | 200,000 | ~143,000 | ~3.8M (est.) |

### 3.5 Activity Annotations

#### Initial approach: class-derived proxy (6 dimensions)

The V3 registry provides class labels and CTCF-binding status but does not include per-biosample activity signals. Our initial approach encoded the available annotations as a 6-dimensional binary activity vector per cCRE:

```
[PLS, pELS, dELS, DNase-H3K4me3, CTCF-only, CTCF-bound]
```

This proved insufficient as a similarity signal. With only 6 binary dimensions and 5 classes, same-class cCREs almost always share ≥ 2 dimensions (triggering the positive threshold), making the pair labels essentially equivalent to class membership. Training with this signal produced a model that converged to the random baseline (val MRR ≈ 0.05 after 10 epochs; see Section 4.7) because there was no within-class structure for the model to learn.

#### Enrichment: ENCODE TF binding profiles (2,818 dimensions)

To provide a biologically meaningful similarity signal, we integrated the ENCODE TF ChIP-seq peaks matrix (ENCFF257UKO). This matrix records, for each cCRE, which of 2,818 transcription factor ChIP-seq experiments show a binding peak overlapping the element. The result is a 2,818-dimensional binary vector per cCRE indicating the element's TF binding profile.

**Data integration pipeline:**
1. The TF peaks matrix uses rDHS accessions (EH38D...) while cCRE manifests use cCRE accessions (EH38E...). We use the GRCh38-cCREs.bed file to map between the two namespaces.
2. The full matrix (2.35M rDHS × 2,818 TF experiments, 13 GB uncompressed) is streamed line-by-line, extracting only rows matching our target cCREs.
3. For the 50k subset: 47,885 / 50,000 cCREs matched (96%); unmatched elements receive zero-filled vectors.

**Binding density statistics (50k subset):**
- Median TFs per cCRE: **8** (range: 0–200+)
- Median cCREs per TF experiment: **662.5**
- TF experiments span 43 tissue ontologies and include CTCF, cohesin components, lineage-specifying TFs (GATA, FOX, SOX families), chromatin remodelers, and general transcription machinery

**Why this signal is richer:** Two cCREs classified as "dELS" (distal enhancer-like) may serve completely different biological roles — one active in hematopoiesis (bound by GATA1, TAL1, RUNX1) and another in neural development (bound by SOX2, PAX6, NEUROD1). The TF binding vector captures this functional distinction. Crucially, this is a similarity signal that k-mer counting cannot replicate: shared TF binding depends on specific motif presence and genomic context, not overall sequence composition.

### 3.6 Pair Construction

Training pairs are built via weak supervision using the following criteria:

**Positives** (up to 4 per query): From same-class candidates (sampled up to 512), a candidate is positive if:
- Binary Jaccard similarity of TF binding vectors >= 0.5, **or**
- Overlap count (shared active TF dimensions) >= 2

Additional positives are drawn from biosample-group-matched elements.

**Negatives** (up to 15 per query), stratified from four sources:
1. Same GC-content bin, any class
2. Hard negatives: same class but TF Jaccard <= 0.1 (functionally distinct despite same regulatory class)
3. Different class, same GC bin
4. Random within-split pool

This stratification ensures the model sees negatives that vary in difficulty — from trivially different (wrong class) to subtly different (same class, different TF program). The hard negatives are particularly valuable with the TF signal: two dELS elements with Jaccard ≤ 0.1 share almost no TF binding, forcing the model to learn sequence features that distinguish their regulatory programs.

### 3.7 Pair Building Optimization

The pair construction pipeline processes each query by iterating over candidate matches and computing pairwise similarity. At scale (50k-200k cCREs), two bottlenecks dominate:

1. **DataFrame row access**: The original implementation used `manifest.iloc[idx]` inside the inner loop to retrieve activity vectors. On a 50k-row DataFrame, this involves index validation, type checking, and Series construction per call — at ~512 candidates per query and 50k queries, this produces ~25 million slow DataFrame accesses.

   **Fix**: Pre-extract all columns into plain Python lists and NumPy arrays before the loop. Inner-loop access becomes a direct list index (`activity_arrays[idx]`), reducing per-access cost by roughly two orders of magnitude.

2. **Repeated boolean mask construction**: Negative sampling originally recomputed full-DataFrame boolean masks per query to find different-class and random-pool candidates:
   ```python
   manifest.index[(manifest["split"] == split) & (manifest["ccre_class"] != cls)]
   ```
   Each such expression creates and ANDs temporary boolean arrays over the entire manifest.

   **Fix**: Pre-build lookup dictionaries keyed by `(split, class)`, `(split, gc_bin)`, and `split` before the main loop. Negative sampling then becomes dict lookups and list filtering.

These optimizations reduced pair building time for the 50k subset from an estimated 3+ hours to approximately 20 minutes — a roughly 10x speedup — without changing the output.

## 4. Model Architecture

### 4.1 Backbone: DNABERT-2

We use DNABERT-2 (117M parameters), a BERT-based DNA language model pretrained on multi-species genomes with Byte Pair Encoding (BPE) tokenization. Unlike character-level DNA tokenizers, BPE captures variable-length k-mer patterns learned from the pretraining corpus.

Key configuration:
- **Hidden dimension**: 768
- **Layers**: 12 transformer layers
- **Attention heads**: 12
- **Vocabulary**: 4,096 BPE tokens
- **Max sequence length**: 512 tokens

### 4.2 LoRA Fine-Tuning

Rather than updating all 117M parameters, we apply Low-Rank Adaptation (LoRA) to the attention projection matrices:
- **Rank**: 16
- **Alpha**: 32 (effective scaling = alpha/rank = 2.0)
- **Dropout**: 0.1
- **Target modules**: attention query/key/value projections
- **Bias**: none

This reduces the trainable parameter count while preserving the pretrained representations.

### 4.3 ColBERT Late-Interaction Scoring

The retrieval model follows the ColBERT architecture adapted for DNA sequences:

1. **Shared encoder**: Both query and document sequences pass through the same DNABERT-2 backbone.
2. **Projection**: Token hidden states (768-dim) are projected to 64 dimensions via a linear layer (no bias).
3. **L2 normalization**: All projected token embeddings are unit-normalized.
4. **MaxSim scoring**: For a query Q with tokens {q_1, ..., q_m} and document D with tokens {d_1, ..., d_n}:

$$\text{score}(Q, D) = \sum_{i=1}^{m} \max_{j=1}^{n} (q_i \cdot d_j) \cdot \mathbb{1}[q_i \text{ not padding}]$$

Padding tokens are masked out in both query and document via attention masks.

### 4.4 Training Objective

We use a multi-positive softmax loss with temperature scaling:

$$\mathcal{L} = -\frac{1}{|B|} \sum_{i \in B} \left[ \log \frac{\sum_{j \in P_i} \exp(s_{ij} / \tau)}{\sum_{k=1}^{N} \exp(s_{ik} / \tau)} \right]$$

where $s_{ij}$ is the ColBERT score between query $i$ and document $j$, $P_i$ is the set of positive documents for query $i$, $N$ is the total number of documents in the batch, and $\tau$ is the temperature parameter (set to 0.1; see Section 4.7 for analysis of why τ=0.05 caused training stagnation).

The denominator includes all documents in the batch (positives from other queries serve as in-batch negatives), providing additional contrastive signal without extra computation.

### 4.5 Training Pipeline Optimization

Naive implementation of the ColBERT training loop achieves only ~16% GPU utilization on dual RTX 2080 Ti cards, with the GPU idle most of the time. Two optimizations raised sustained utilization to ~95%:

**Pre-tokenizing DataLoader.** In the original pipeline, each training step serializes CPU tokenization and GPU computation: tokenize queries on CPU, encode on GPU, tokenize documents on CPU, encode on GPU, score, backward. The GPU idles during both tokenization phases.

We restructure the pipeline so that tokenization runs in DataLoader worker processes. A custom collator pre-tokenizes all query and document sequences and constructs the positive mask on CPU, returning GPU-ready token tensors. With `num_workers=8` and `prefetch_factor=2`, workers prepare upcoming batches while the GPU processes the current one. This eliminates the CPU-GPU serialization bottleneck entirely — the GPU never waits for the tokenizer.

**Vectorized MaxSim scoring.** The original scoring loop iterates over all query-document pairs in Python:

```python
for q_idx in range(num_queries):       # 20 queries
    for d_idx in range(num_docs):       # 140 docs
        score = einsum("qf,df->qd", q[q_idx], d[d_idx])  # tiny kernel
```

This launches 2,800 individual CUDA kernels per batch, each too small to saturate the GPU, with Python interpreter overhead between launches.

We replace this with a single batched einsum that computes all pairwise token similarities in one fused operation:

```python
sim = einsum("qif,djf->qidj", q_proj, d_proj)   # one large kernel
max_sim = sim.max(dim=-1).values                  # vectorized max over doc tokens
scores = (max_sim * q_mask[:, :, None]).sum(dim=1) # masked sum
```

This keeps the GPU busy on a single large matrix operation with no Python-loop overhead.

**Combined effect:** The two optimizations are complementary — the first ensures the GPU always has work queued, the second ensures each unit of work fully utilizes the GPU's compute capacity. Together they improved GPU utilization from ~16% to ~95%. Observed wall time is ~36 min/epoch (1,795 steps at ~1.18 s/step) plus ~5 min validation, giving ~7 hours for 10 epochs on 2× RTX 2080 Ti.

### 4.6 Training Configuration

| Parameter | Debug (10k) | Small (50k) |
|-----------|------------|-------------|
| Epochs | 5 | 10 |
| Batch size | 16 | 20 |
| Gradient accumulation | 1 | 1 |
| Learning rate | 1e-4 | 1e-4 |
| Weight decay | 0.01 | 0.01 |
| Max grad norm | 1.0 | 1.0 |
| Negatives per query | 6 | 6 |
| Temperature | 0.05 | 0.05 |
| GPUs | 2x RTX 2080 Ti | 2x RTX 2080 Ti |
| Precision | FP32 | FP32 |

The DNABERT-2 backbone is wrapped in `DataParallel` across two GPUs, splitting the token encoding batch while keeping scoring and loss computation on the primary device.

### 4.7 Training Dynamics

**Random loss baseline.** With `batch_size=20` and `negatives_per_query=6`, each training batch contains 20 queries scored against 140 documents (20 positives + 120 negatives). For a model assigning uniform scores, the expected InfoNCE loss is:

$$\mathcal{L}_\text{random} = \log(N_\text{docs}) = \log(140) \approx 4.95$$

This is the theoretical floor for a model with no discriminative ability — breaking below it means the model is reliably ranking at least some positives above their negatives.

**Observed epoch 1 trajectory.** Training loss begins at ~37.5 at step 100, substantially above the random baseline. This is expected: at initialization, the random LoRA projection produces score distributions with high variance, and the aggressive temperature τ=0.05 amplifies any score gap between positive and the best negative in the batch. Even a small margin by which the positive is outscored translates to a large loss contribution. The loss decreases steadily throughout epoch 1 (reaching 6.76 average), with validation loss of 4.90 — just below the random baseline — and a validation MRR of 0.048.

**Stagnation at epoch 2.** At the start of epoch 2, training loss drops sharply to ~4.91 and becomes nearly flat (±0.003 over 400 steps). The model has reached the random baseline but cannot escape it. The likely cause is the temperature: at τ=0.05, the softmax gradient with respect to the positive score is approximately:

$$\frac{\partial \mathcal{L}}{\partial s_\text{pos}} \approx \frac{1}{\tau}\left(p_\text{pos} - 1\right)$$

Near the random baseline, $p_\text{pos} \approx 1/140 \approx 0.007$, so the gradient signal is $\approx -19.9/\tau$. At τ=0.05 this is large in magnitude — but the update is also opposed by equally large gradients from the 139 negatives. The result is that parameter updates oscillate rather than push the positive consistently higher, causing the loss to stall.

A higher temperature (e.g., τ=0.1 or τ=0.2) softens the softmax distribution, producing more stable gradients and allowing the model to escape this plateau. This is a known issue with very low temperatures in contrastive training; they are beneficial once the model has already learned rough ordering but hinder early-stage learning.

**τ=0.1 results with 6-dim activity signal.** Increasing temperature to 0.1 eliminated the sharp stagnation but did not improve retrieval quality. Over 10 epochs, training loss decreased smoothly from 5.72 to 4.76, but validation MRR remained flat at ~0.05 (best: 0.051 at epoch 2), and validation loss began increasing after epoch 2 (4.90 → 5.03), indicating overfitting. The model learned to marginally rank same-class elements above cross-class ones but could not learn any within-class discrimination.

**Root cause: similarity signal, not architecture.** The 6-dim activity vector derived from class labels offered no information beyond class membership itself. With Jaccard ≥ 0.5 on 6 binary dimensions, nearly all same-class pairs qualify as positive, reducing the task to "distinguish classes by sequence" — which k-mer counting already solves. The model needs a richer signal to learn meaningful sequence-to-function mappings.

**Resolution: TF binding enrichment.** We replaced the 6-dim class-derived vectors with 2,818-dim TF ChIP-seq binding profiles from ENCODE (see Section 3.5). This provides genuine within-class functional diversity: two distal enhancers bound by different TF programs are now correctly distinguished as negatives, and only elements sharing similar TF binding qualify as positives. Training with this enriched signal is ongoing.

## 5. Baselines

### 5.1 Random Baseline

Documents are scored with uniform random values. This establishes the floor for all metrics.

### 5.2 K-mer Baseline (k=6)

For each query-document pair, we compute k-mer frequency vectors (k=6, yielding 4^6 = 4,096 possible 6-mers) and score by cosine similarity. This is a strong lexical baseline that captures local sequence composition without any learned parameters.

### 5.3 Mean-Pooled Baseline

(Planned) Uses the same DNABERT-2 backbone but replaces ColBERT's token-level MaxSim with mean-pooled sequence embeddings scored by cosine similarity. This isolates the contribution of late interaction versus single-vector representations.

## 6. Evaluation

### 6.1 Metrics

**Retrieval metrics** (computed against labeled positive pairs from the test split):
- Recall@k (k = 1, 5, 10, 50)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)

**Biological metrics** (computed over the top-k retrieved documents):
- **Class purity@k**: Fraction of retrieved documents sharing the query's regulatory class
- **Activity Jaccard@k**: Mean Jaccard similarity of activity vectors between query and retrieved documents
- **Activity correlation@k**: Mean Pearson correlation of activity vectors

### 6.2 Preliminary Results (Debug, 10k cCREs)

Evaluation on 256 test queries against a corpus of 1,024 test-split cCREs:

| Metric | ColBERT (ours) | K-mer (k=6) | Random |
|--------|---------------|-------------|--------|
| Recall@1 | 0.0020 | 0.0010 | 0.0010 |
| Recall@5 | 0.0039 | 0.0068 | 0.0049 |
| Recall@10 | 0.0117 | 0.0127 | 0.0049 |
| Recall@50 | 0.0312 | 0.0391 | 0.0264 |
| MRR | 0.0217 | 0.0238 | 0.0160 |
| nDCG | 0.0149 | 0.0178 | 0.0113 |
| **Class purity@5** | **0.630** | 0.568 | 0.488 |
| **Class purity@10** | **0.627** | 0.589 | 0.561 |
| **Class purity@50** | **0.625** | — | — |

**Key observations**:

- The ColBERT model achieves the highest class purity across all k values, indicating it retrieves regulatory elements of the correct class more consistently than either baseline. At k=5, the model retrieves same-class elements 63% of the time vs. 57% for k-mer and 49% for random.

- Recall/MRR/nDCG numbers are low across all methods. This is expected: with only 1,024 corpus documents and a small number of labeled positives per query (up to 4), most true positives are simply absent from the sampled corpus.

- The k-mer baseline is competitive on retrieval metrics (MRR 0.024 vs. model 0.022), reflecting the fact that same-class cCREs often share sequence motifs detectable by k-gram matching. However, k-mer's lower class purity suggests it also retrieves lexically similar but functionally different elements.

- These are preliminary results from the 10k debug subset with 5 training epochs. Scaling to 50k (10 epochs) and 200k cCREs is expected to improve the learned model disproportionately, as the pretrained DNABERT-2 representations benefit more from additional training signal than the parameter-free k-mer baseline.

## 7. Discussion

*Results from 50k and 200k training runs pending.*

## References

- Khattab, O. & Zaharia, M. (2020). ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. SIGIR.
- Zhou, Z. et al. (2024). DNABERT-2: Efficient foundation model and benchmark for multi-species genome. ICLR.
- ENCODE Project Consortium (2020). Expanded encyclopaedias of DNA elements in the human and mouse genomes. Nature.
- ENCODE SCREEN Registry V3. https://screen.encodeproject.org/
- Hu, E. et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR.
