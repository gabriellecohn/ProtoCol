# Baseline Suite

## Overview

Each baseline answers a specific question about our retrieval system. The suite is ordered from simplest (confound checks) to most complex (architectural comparisons).

| # | Baseline | Question it answers | Priority | Est. time |
|---|----------|-------------------|----------|-----------|
| 1 | Random | Is the benchmark meaningful at all? | ✅ Done | — |
| 2 | GC-matched random | Are we just exploiting GC content confounds? | High | 30 min |
| 3 | K-mer cosine (k=6) | Does raw sequence composition solve it? | ✅ Done | — |
| 4 | TF-IDF k-mer | Do weighted motif frequencies beat raw counts? | High | 1 hr |
| 5 | MinHash k-mer Jaccard | Can approximate set similarity match cosine? | Medium | 1 hr |
| 6 | BLASTn / local alignment | Does classic alignment-based search solve it? | Medium | 2 hr |
| 7 | gkm-SVM | Do gapped k-mers (motif grammar) solve it? | Medium | 3 hr |
| 8 | Pretrained DNABERT-2 (mean-pooled) | Does generic DNA representation already solve it? | ✅ Done | — |
| 9 | Oracle label-space retrieval | What is the ceiling under our relevance definition? | High | 30 min |
| 10 | Single-vector neural (fine-tuned, mean-pooled) | Does learned representation help over pretrained? | Critical | 2 hr |
| 11 | Multi-vector ColBERT (fine-tuned) | Does late interaction help over single-vector? | Critical | — (training) |
| 12 | DeepSEA-style direct predictor | Is retrieval even worth doing vs direct prediction? | Future | 4 hr |

---

## Nuisance / confound baselines

### 1. Random retrieval ✅

**Status:** Complete

**Method:** Score each document with a uniform random value. Rank by score.

**Purpose:** Absolute floor. If any method barely beats this, the benchmark is weak.

**Current results:**
- MRR: 0.0054
- Recall@50: 0.0073

---

### 2. GC-matched random retrieval

**Status:** TODO

**Method:** For each query, restrict the retrieval corpus to documents in the same GC-content bin (decile), then score randomly within that bin.

**Purpose:** GC content is the strongest single confounder in DNA sequence analysis. Promoters are GC-rich, distal enhancers are GC-poor. If GC-matched random retrieval achieves high class purity, then methods that appear to "retrieve same-class elements" may just be matching GC content.

**Implementation:**
- Bin all corpus sequences by GC content (10 bins)
- For each query, score documents in the same bin uniformly, score others at -inf
- Evaluate as usual

---

## Alignment-free sequence baselines

### 3. K-mer cosine (k=6) ✅

**Status:** Complete

**Method:** Compute 6-mer frequency vectors (4^6 = 4,096 dimensions) for each sequence. Score query-document pairs by cosine similarity.

**Purpose:** Strong lexical baseline that captures local sequence composition without any learned parameters. DNA k-mer methods are often surprisingly competitive with deep models.

**Current results:**
- MRR: 0.0074
- Recall@50: 0.0127

---

### 4. TF-IDF weighted k-mer retrieval

**Status:** TODO

**Method:** Same 6-mer frequency vectors as baseline #3, but weight each k-mer by its inverse document frequency across the corpus:

$$\text{tfidf}(k, d) = \text{tf}(k, d) \times \log \frac{N}{|\{d' : k \in d'\}|}$$

Score by cosine similarity on TF-IDF vectors.

**Purpose:** Raw k-mer cosine treats all k-mers equally. TF-IDF downweights common k-mers (e.g., poly-A runs) and upweights rare, discriminative k-mers (e.g., specific TF binding motifs). This is the "smart" version of k-mer retrieval and is probably the most important non-neural baseline.

**Implementation:**
- Compute 6-mer counts for entire corpus
- Compute IDF for each k-mer: log(N / doc_freq)
- Multiply each document's k-mer vector by IDF weights
- L2-normalize
- Score by cosine

---

### 5. MinHash k-mer Jaccard

**Status:** TODO

**Method:** For each sequence, extract the set of unique k-mers (k=6). Compute approximate Jaccard similarity using MinHash signatures (e.g., 128 or 256 hash functions).

$$J(A, B) = \frac{|K_A \cap K_B|}{|K_A \cup K_B|}$$

where $K_A$ and $K_B$ are the sets of k-mers in sequences A and B.

**Purpose:** Jaccard over k-mer sets is a fundamentally different similarity measure than cosine over k-mer frequencies. Cosine is sensitive to k-mer abundance; Jaccard only cares about presence/absence. MinHash makes this efficient for large corpora (O(1) per pair after O(n) preprocessing).

**Implementation:**
- Extract unique 6-mer sets per sequence
- Build MinHash signatures (datasketch library or manual implementation)
- Score by estimated Jaccard from MinHash
- Evaluate as usual

---

## Alignment-based baselines

### 6. BLASTn / local alignment

**Status:** TODO

**Method:** Use BLASTn (or a simpler Smith-Waterman local alignment) to score each query against all corpus documents. Use the top HSP bit score as the similarity measure.

**Purpose:** BLAST is the canonical baseline for nucleotide sequence similarity search. It finds local alignments — shared subsequences allowing gaps and mismatches. This answers a clean biological question:

> Does the model retrieve **functionally similar** sequences, or just sequences with obvious alignment-based similarity?

If BLAST does as well as the learned model, the model is not learning anything beyond what alignment already captures.

**Implementation:**
- Build a BLAST database from corpus sequences
- Run blastn for each query with `-outfmt 6` for tabular output
- Parse bit scores as similarity measure
- Rank and evaluate

---

### 7. gkm-SVM (gapped k-mer)

**Status:** TODO

**Method:** Extract gapped k-mer features (e.g., k=11, informative positions=7, gap positions=4) for each sequence. These features capture motif-like patterns with variable spacing — e.g., "GATA...GATA" with 4 wildcard positions.

Score documents by kernel similarity in gapped k-mer feature space, or train an SVM on the retrieval task.

**Purpose:** gkm-SVM was designed specifically for regulatory DNA and is a standard classical baseline in enhancer/TFBS prediction. Gapped k-mers are much closer to the biological motif grammar than plain k-mers: a TF binding site like the GATA motif tolerates specific positions of variation. If the learned retrieval model does not beat gkm-SVM, it is hard to argue the model captures regulatory structure.

**Implementation:**
- Install `lsgkm` or use the Python wrapper
- Extract gapped k-mer feature vectors
- Score by kernel similarity (inner product in feature space)
- Evaluate as usual

**Reference:** Ghandi et al., "Enhanced Regulatory Sequence Prediction Using Gapped k-mer Features," PLoS Computational Biology, 2014.

---

## Neural baselines

### 8. Pretrained DNABERT-2 (mean-pooled, no fine-tuning) ✅

**Status:** Complete

**Method:** Load pretrained DNABERT-2 weights (117M params). Encode each sequence through the full transformer. Mean-pool token embeddings (weighted by attention mask) into a single 768-dim vector. L2-normalize. Score by cosine similarity.

No fine-tuning, no task-specific training.

**Purpose:** Isolates what the pretrained DNA language model already knows about sequence similarity. This is the "foundation model embeddings" baseline — it tells us whether task-specific training adds value over off-the-shelf representations.

**Current results:**
- MRR: 0.0093
- Recall@50: 0.0161
- Already beats k-mer by 25% on MRR

---

### 9. Oracle label-space retrieval

**Status:** TODO

**Method:** Score each query-document pair by cosine similarity directly on their TF binding vectors (2,818-dim binary vectors). No sequence information used at all.

**Purpose:** This is the **upper bound** — the best any method could do if it perfectly predicted TF binding from sequence. It tells us how much headroom exists. If the oracle is only marginally better than random (because TF labels are sparse), the benchmark itself is limited. If the oracle is much better, there is room for models to improve.

**Implementation:**
- Load TF vectors for all queries and corpus documents
- Score by cosine similarity on TF vectors
- Evaluate as usual

---

### 10. Single-vector neural (fine-tuned, mean-pooled)

**Status:** TODO — **CRITICAL BASELINE**

**Method:** Same DNABERT-2 backbone with LoRA, same contrastive training pairs, same loss function, same hyperparameters — but replace ColBERT's MaxSim scoring with mean-pooled cosine similarity:

$$\text{score}(Q, D) = \cos(\text{mean}(E_Q), \text{mean}(E_D))$$

where $E_Q$ and $E_D$ are the token embedding sequences after the shared projection layer.

**Purpose:** This is the **load-bearing comparison** for our central hypothesis. It isolates the contribution of late interaction (token-level MaxSim) versus single-vector compression (mean pooling). If this baseline matches or beats ColBERT, then late interaction does not help for regulatory DNA retrieval — the signal is captured well enough by global representations.

**Implementation:**
- Modify the ColBERT model to mean-pool and cosine-score instead of MaxSim
- Train with identical config (same pairs, same epochs, same LR, same grad_accum)
- Evaluate on the same test set

---

### 11. Multi-vector ColBERT (fine-tuned)

**Status:** Training in progress

**Method:** Full ColBERT architecture: shared DNABERT-2 backbone → 64-dim projection → L2-normalized token embeddings → MaxSim scoring. Fine-tuned with LoRA on TF-enriched contrastive pairs.

**Purpose:** This is our proposed method. The hypothesis is that token-level matching captures positional motif structure that mean pooling averages away.

---

## Prediction baseline (non-retrieval)

### 12. DeepSEA-style direct multilabel predictor

**Status:** Future (after dataset switch)

**Method:** Train a CNN (or DNABERT-2 + classification head) to directly predict the 919 DeepSEA chromatin features from 1kb sequence input. Evaluate by comparing predicted label vectors rather than doing retrieval.

**Purpose:** Answers the question: **is retrieval even worth doing, versus just directly predicting the labels?** If a simple predictor achieves high accuracy on the chromatin features, retrieval adds unnecessary complexity. If prediction is hard but retrieval works, the retrieval approach provides value through annotation transfer that direct prediction cannot.

**Reference:** Zhou & Troyanskaya, "Predicting effects of noncoding variants with deep learning–based sequence model," Nature Methods, 2015.

---

## Summary: what each baseline tests

| Question | Baselines |
|----------|-----------|
| Are we exploiting confounds? | Random, GC-matched random |
| Does plain sequence similarity solve it? | BLASTn |
| Do classical motif features solve it? | K-mer cosine, TF-IDF k-mer, MinHash, gkm-SVM |
| Does generic learned representation solve it? | Pretrained DNABERT-2 |
| Does task-specific learned representation help? | Single-vector fine-tuned |
| Does late interaction help over single-vector? | ColBERT (ours) |
| Is retrieval even necessary? | DeepSEA direct predictor |
| What is the theoretical ceiling? | Oracle label-space retrieval |
