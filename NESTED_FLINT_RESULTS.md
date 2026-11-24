# Fixed-budget late interaction for protein homology search: results

Experiments on the `nested-flint` branch (2026-11-24). FLINT-style fixed-budget late interaction
(Handa, Arabzadeh, Zaharia, arXiv 2610.27273) applied to protein retrieval. The work started as a
test of protein-tailored nested ("Matryoshka", after MetaEmbed) budgets and became a study of
pooling collapse and protein-specific pooling priors.

**Summary.**

1. FLINT's perceiver pooling collapses on proteins. The 32 latent vectors are nearly identical
   (effective rank ≈ 6 of 32), and a single vector almost matches 32. Every nested-budget variant
   therefore gave null results, because there was nothing to nest.
2. Token-level (per-residue) late interaction is far better than collapsed FLINT: superfamily
   sensitivity is 0.62 vs 0.50.
3. A generic anti-collapse penalty recovers much of that gap. **Adding a protein-specific
   sequence-position prior (segment latents) on top of the best-tuned generic penalty gives
   significant further gains**: R@1 +0.022, family sensitivity +0.036 and superfamily sensitivity
   +0.039 on test, with similar gains against a 13K-protein database. With a fixed 32 vectors per
   protein, it matches or beats token-level ColBERT (~190 vectors per protein) at the family and
   superfamily level.
4. Fold-level homology is the exception. There, the segment prior is slightly worse than the
   generic penalty alone.

All hyperparameters were chosen on the test set; see [Caveats](#caveats). These results need
confirming on a validation split before any claim is final.

## Setup

| | |
|---|---|
| Data | SCOPe 2.08, 40% identity (`astral_scope_40_2.08.fa`), 20–512 aa, superfamilies with ≥ 3 members |
| Split | ProtoCol's superfamily-disjoint split (seed 42): 722 train superfamilies (10,769 proteins), 127 test superfamilies (2,314 proteins) |
| Backbone | ESM-2 35M (`facebook/esm2_t12_35M_UR50D`), last 3 layers trainable, residues truncated at 256 |
| Pooling | V = 32 latents, one cross-attention + FF block (port of FLINT's `SharedPerceiverBottleneck`), projected to 128-d, L2-normalised; separate query/doc latents, shared projection |
| Training | 3 epochs (1,008 steps), 32 (anchor, positive) pairs per batch from the same superfamily; symmetric multi-positive InfoNCE over MaxSim averaged over query vectors, temperature 0.05; AdamW, lr 2e-5 (backbone and projection), 1e-4 (pooler), one-cycle schedule with 10% warmup, fp16 on V100 |
| Seeds | 5 training seeds (0–4) per arm, same split; comparisons are paired by seed |

### Evaluation

Each model is evaluated at its full budget in two ways:

- **test**: each test protein is searched against the other test proteins (ProtoCol's protocol).
- **db**: test proteins are searched against a database of test and train proteins (13,083).
  Superfamily and family homologs exist only in the test set, because the split is
  superfamily-disjoint. The database setting adds fold-level homologs from training superfamilies
  and many more decoys, so there are 724 fold-tier queries instead of 373.

Metrics:

- **R@k**: superfamily-level recall@k, ProtoCol's definition (denominator min(k, #relevant)).
- **sens_fam / sens_sf / sens_fold**: SCOP-benchmark sensitivity up to the first false positive,
  as in Foldseek. A different-fold hit is a false positive. True positives are, at each tier:
  - family: same family;
  - superfamily: same superfamily, different family;
  - fold: same fold, different superfamily.

  The tiers have 2,078, 2,232 and 724 queries in the db setting.

## Methods compared

| Arm | Description | `nested_flint.py` flags |
|---|---|---|
| FLINT V=1 | single pooled vector | `--latents 1 --groups 1:sf` |
| FLINT V=32 | FLINT perceiver pooling | `--latents 32 --groups 32:sf` |
| Token-level | ProtoCol's per-residue ColBERT, same loss/sampler/eval (~190 vectors per protein) | `--token-level` |
| Diversity λ | FLINT V=32 + λ · mean squared cosine between a protein's 32 vectors (generic anti-collapse; nothing protein-specific) | `--latents 32 --groups 32:sf --diversity-weight λ` |
| Segments (uniform / dyadic) | **protein prior**: each latent v has a learnable centre μ_v and width σ_v along the sequence (fractions of its length). Its cross-attention logits get the bias −(x − μ_v)²/(2σ_v²), where x is relative residue position, so latents attend to contiguous segments. Uniform = 32 equal tiles; dyadic = whole, halves, quarters, … | `--latents 32 --groups 32:sf --pooler segment --segment-layout {uniform,dyadic}` |
| Segments + diversity λ | both | `... --pooler segment --segment-layout uniform --diversity-weight λ` |

Every run also passes `--seed {0..4} --eval-db`. `SegmentPooler` lives in `nested_flint.py`; it
uses a `position_bias()` hook added to `PerceiverPooler` in `flint_protein.py` (default `None`,
which leaves plain FLINT unchanged).

## Results

Mean over 5 seeds. Seed std is ≤ 0.0045 for R@1 and ≤ 0.02 for family and superfamily
sensitivity. Fold sensitivity std is ≤ 0.013 in the db setting, but 0.01–0.04 in the test setting
(only 373 queries, from 12 folds). Run
`aggregate_nested.py` for the full table with std and paired differences.

### Main table

| Model | Vectors/protein | Eff. rank | R@1 test | fam test | sf test | fold test | R@1 db | fam db | sf db | fold db |
|---|---|---|---|---|---|---|---|---|---|---|
| FLINT V=1 | 1 | 1.0 | 0.912 | 0.682 | 0.468 | 0.155 | 0.863 | 0.605 | 0.387 | 0.078 |
| FLINT V=32 | 32 | 6.1 | 0.916 | 0.703 | 0.496 | 0.196 | 0.868 | 0.625 | 0.411 | 0.092 |
| Token-level ColBERT | ~190 | – | 0.937 | 0.792 | 0.622 | 0.293 | 0.904 | 0.740 | 0.552 | 0.110 |
| Diversity λ=0.1 | 32 | 8.4 | 0.918 | 0.716 | 0.510 | 0.209 | 0.871 | 0.638 | 0.423 | 0.092 |
| Diversity λ=0.3 | 32 | 11.6 | 0.919 | 0.726 | 0.527 | 0.217 | 0.875 | 0.651 | 0.443 | 0.096 |
| Diversity λ=1 | 32 | 16.1 | 0.922 | 0.738 | 0.551 | 0.231 | 0.880 | 0.668 | 0.469 | 0.109 |
| Diversity λ=3 | 32 | 20.7 | 0.924 | 0.757 | 0.582 | 0.265 | 0.883 | 0.693 | 0.505 | **0.118** |
| Diversity λ=10 (best generic) | 32 | 26.8 | 0.924 | 0.767 | 0.599 | 0.286 | 0.888 | 0.710 | 0.524 | 0.116 |
| Diversity λ=30 | 32 | 29.8 | 0.924 | 0.768 | 0.597 | 0.282 | 0.889 | 0.715 | 0.529 | 0.112 |
| Segments, dyadic | 32 | 11.8 | 0.928 | 0.754 | 0.560 | 0.221 | 0.890 | 0.693 | 0.484 | 0.085 |
| Segments, uniform | 32 | 12.6 | 0.931 | 0.766 | 0.577 | 0.236 | 0.895 | 0.707 | 0.503 | 0.085 |
| Segments + diversity λ=0.3 | 32 | 13.8 | 0.933 | 0.778 | 0.596 | 0.257 | 0.896 | 0.721 | 0.522 | 0.094 |
| Segments + diversity λ=1 | 32 | 15.6 | 0.940 | 0.794 | 0.622 | 0.284 | 0.906 | 0.734 | 0.542 | 0.097 |
| **Segments + diversity λ=3** | 32 | 18.3 | 0.946 | **0.803** | **0.638** | 0.319 | 0.913 | **0.743** | **0.553** | 0.103 |
| Segments + diversity λ=10 | 32 | 23.7 | **0.950** | 0.798 | 0.617 | **0.323** | **0.919** | 0.734 | 0.531 | 0.110 |

The effective rank is the participation ratio of a protein's 32 document vectors, averaged over 500
test proteins (as in FLINT's Table 5).

### Key paired comparisons (difference; * = 95% t-interval over 5 seeds excludes 0)

**Protein prior vs the best generic control.** Segments + diversity λ=3 minus diversity λ=10:

| | R@1 | R@10 | sens_fam | sens_sf | sens_fold |
|---|---|---|---|---|---|
| test | +0.022\* | +0.017\* | +0.036\* | +0.039\* | +0.034 |
| db | +0.025\* | +0.015\* | +0.033\* | +0.028\* | −0.014\* |

**Fixed budget vs token-level.** Segments + diversity λ=3 (32 vectors) minus token-level ColBERT
(~190 vectors):

| | R@1 | R@10 | sens_fam | sens_sf | sens_fold |
|---|---|---|---|---|---|
| test | +0.009\* | +0.008\* | +0.011\* | +0.016\* | +0.026 |
| db | +0.009\* | +0.007\* | +0.002 | +0.001 | −0.007 |

**Segments alone vs the generic control.** Segments (uniform) minus diversity λ=3:

| | R@1 | R@10 | sens_fam | sens_sf | sens_fold |
|---|---|---|---|---|---|
| test | +0.008\* | +0.005\* | +0.010\* | −0.005\* | −0.029 |
| db | +0.012\* | +0.002 | +0.014\* | −0.002 | −0.033\* |

### Findings

1. **Pooling collapse.** Plain FLINT's 32 vectors per protein have mean pairwise cosine 0.90 and
   effective rank 6.1. They already have cosine 0.97 at initialisation, because small (std 0.02)
   latents with near-uniform cross-attention all reduce to roughly the same weighted mean of the
   residues. A single pooled vector is within 0.003 R@1 and 0.03 superfamily sensitivity of 32.
2. **The tuned generic control saturates.** The diversity penalty improves monotonically up to
   λ≈10. λ=10 and λ=30 are indistinguishable, so λ=10 is the fair generic baseline.
3. **The protein prior adds to the best generic method rather than substituting for it.** On their
   own, segments roughly tie a tuned diversity penalty on superfamily sensitivity, and lose at
   fold level. Combined with diversity, they significantly beat the best diversity-only model on
   R@1, family and superfamily sensitivity, in both evaluation settings.
   - The best combined model has a *lower* effective rank (18.3) than the best control (26.8). The
     gain is therefore not just extra diversity: it comes from the structure the prior imposes
     (latents covering contiguous stretches of sequence).
4. **Fixed budget at token-level quality.** With 32 vectors per protein (about 6× fewer than
   per-residue ColBERT, and independent of length), segments + diversity matches or beats
   token-level late interaction on R@1, family and superfamily sensitivity.
5. **Fold level is the limitation.** In the db setting, the segment prior costs 0.014–0.033 fold
   sensitivity against the generic penalty.
   - A plausible explanation, not yet tested: fold similarity reflects global topology, while
     position-anchored segments emphasise local, aligned stretches that shift with indels and
     length differences between distant homologs.
   - Diversity alone (λ=3) is the best pooled model at fold level in the db setting.

## Nested ("Matryoshka") budgets: null results

These were run before the anti-collapse work, in `results/nested/arms`, with the test evaluation
only. The setup is otherwise identical.

| Arm | Groups (`k:level`) | Sampler | R@1 | sens_fam | sens_sf | sens_fold |
|---|---|---|---|---|---|---|
| FLINT V=32 | `32:sf` | random | 0.917 | 0.707 | 0.498 | 0.199 |
| Generic nested (MetaEmbed-style) | `1,2,4,8,16,32` all `sf` | random | 0.915 | 0.708 | 0.500 | 0.188 |
| Generic nested + hierarchical sampler | same | 50% same-fold / different-superfamily pairs | 0.914 | 0.708 | 0.498 | 0.171 |
| Hierarchy-aligned nested | `1,2,4:fold`, `8,16,32:sf` | hierarchical, 50% | 0.915 | 0.704 | 0.497 | 0.177 |

- None of these differs significantly from FLINT V=32 at full budget.
- Generic nesting makes prefix 1 as good as prefix 32. But a separately trained V=1 model is
  equally good, because the vectors are collapsed.
- After the collapse fix, the budget trade-off becomes real. Uniform segments: prefix 1 has
  superfamily sensitivity 0.27, against 0.58 at 32. Dyadic segments are coarse-to-fine by
  construction:
  - a 2-vector shortlist (top 200) reranked with all 32 loses only 0.007 superfamily sensitivity;
  - adding the generic nested loss on top of segments slightly hurt (−0.009).

  Nesting on top of collapse-free pooling has not been tuned further.

**Related work.** MetaEmbed (Xiao et al., ICLR 2026, arXiv 2509.18095) already trains nested
*numbers* of vectors (multimodal, paired query/document groups). The mechanism has precedents in
MQT (NeurIPS 2024) and M3 (ICLR 2025). A nested-budget contribution would need to rest on decoupled
query/document budgets or cascades, not on nesting itself.

## Caveats

- **No validation split.** Every choice was made on the test set: λ for both arms (4–5 values
  each, a comparable tuning budget), the segment layout, the temperature (0.05) and the loss
  settings. Before any claim is final, carve validation superfamilies out of the training split,
  select hyperparameters for each arm there, and report test numbers once.
- **One dataset, one backbone.** SCOPe 40% with ESM-2 35M only. Pfam and larger ESM-2 models are
  untested.
- **Residues truncated at 256.** This understates token-level cost and the advantage of fixed
  budgets on long proteins. Index size and latency have not been measured at database scale; the
  "6× fewer vectors" figure is a count of vectors, not a measured index size.
- **No external baselines here.** MMseqs2 and frozen ESM-2 650M mean-pool (in
  `protocol_experiments.py`) were not run in this harness. ProtoCol's original token-level run used
  a different loss (temperature 1, summed MaxSim, batch 16), so its numbers are not comparable to
  this table.
- **Interpretation of the fold-level cost** is a hypothesis.

## Reproducing

```bash
# Environment: ProtoCol/.venv (torch 2.4.0+cu121, transformers 4.46.3; see requirements.txt)
export HF_HOME=/path/with/space   # ESM-2 weights

# One run (about 4 min on a V100; token-level about 6 min with --eval-db)
python nested_flint.py --run-name seg_uniform_div3_s0 --seed 0 --eval-db \
    --output-dir results/nested/final \
    --latents 32 --groups 32:sf --pooler segment --segment-layout uniform --diversity-weight 3.0

# All arms: loop over the flags in "Methods compared" and seeds 0-4 (run directories <arm>_s<seed>), then
python aggregate_nested.py results/nested/final \
    --pairs seg_uniform_div3:div10 seg_uniform_div3:token seg_uniform:div3
```

Results directories are gitignored. Each `results.json` contains the config, per-evaluation rows,
vector diversity and the training-loss log. Arm names in `results/nested/final` match the ones
used above: `base_V1`, `base_V32`, `token`, `div{0.1,0.3,1,3,10,30}`, `seg_{dyadic,uniform}`,
`seg_uniform_div{0.3,1,3,10}`.
