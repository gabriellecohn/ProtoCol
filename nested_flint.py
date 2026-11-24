#!/usr/bin/env python3
"""
Nested (Matryoshka) FLINT for protein homology search.

One FLINT model whose latent *prefixes* are usable retrievers on their own:
the first k of V vectors per protein score with MaxSim like a k-latent model.
Following MetaEmbed (Xiao et al., ICLR 2026), the loss is a weighted sum of
InfoNCE terms, one per group of (query prefix, document prefix) sizes.

Protein-specific twist (hierarchy-aligned nesting): each group also names the
SCOP level that defines its positives, so small prefixes can be trained on
coarse homology (fold) and larger prefixes on finer homology (superfamily).
Prefix size then tracks granularity in the classification, which supports a
cascade: shortlist with a few vectors, rerank with all of them.

Groups are given as "k:level" or "kq/kd:level", comma-separated:
    --groups 32:sf                          per-budget FLINT baseline (V=32)
    --groups 2:sf,4:sf,8:sf,16:sf,32:sf     generic nested port (MetaEmbed-style)
    --groups 2:fold,4:fold,8:sf,16:sf,32:sf hierarchy-aligned nesting

Levels: fold (same SCOP fold), sf (same superfamily), fam (same family).
Positives are multi-positive: every in-batch document sharing the anchor's
label at that level counts as a positive.

Scores are MaxSim averaged over query vectors (not summed), divided by a
temperature, so logits have the same scale for every prefix size.

Evaluation, for each prefix k (and the requested cascades):
  * R@{1,5,10,100} with superfamily labels (same definition as protocol_experiments)
  * SCOP-benchmark sensitivity up to the first false positive (as in Foldseek):
    TP_fam = same family; TP_sf = same superfamily, different family;
    TP_fold = same fold, different superfamily; FP = different fold.
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoTokenizer

import protocol_experiments as pe
from flint_protein import PerceiverPooler, ProteinFLINT, encode_fixed

DEVICE = pe.DEVICE
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
LEVELS = ("fold", "sf", "fam")


# =============================================================================
# Labels
# =============================================================================
def scop_levels(p):
    """SCOPe sccs 'a.1.1.2' -> fold 'a.1', superfamily 'a.1.1', family 'a.1.1.2'."""
    parts = p.family.split(".")
    return {"fold": ".".join(parts[:2]), "sf": p.superfamily, "fam": p.family}


def label_ids(proteins):
    """{level: LongTensor of integer ids}, one id per protein."""
    out = {}
    for lvl in LEVELS:
        vocab = {}
        out[lvl] = torch.tensor([vocab.setdefault(scop_levels(p)[lvl], len(vocab)) for p in proteins])
    return out


# =============================================================================
# Groups
# =============================================================================
def parse_groups(spec, V):
    groups = []
    for item in spec.split(","):
        size, level = item.strip().split(":")
        kq, kd = (size.split("/") if "/" in size else (size, size))
        kq, kd = int(kq), int(kd)
        assert level in LEVELS, f"unknown level {level!r}"
        assert 1 <= kq <= V and 1 <= kd <= V, f"group {item} exceeds V={V}"
        groups.append((kq, kd, level))
    return groups


class TokenLevel(torch.nn.Module):
    """ProtoCol's per-residue ColBERT behind the (vectors, mask) interface used here."""

    def __init__(self, model_name, proj_dim, n_trainable_layers):
        super().__init__()
        self.colbert = pe.ProteinColBERT(model_name, proj_dim=proj_dim, n_trainable_layers=n_trainable_layers)

    def forward(self, input_ids, attention_mask, mode="doc"):
        return self.colbert(input_ids, attention_mask), attention_mask.bool()


def segment_layout(V, layout):
    """Initial (centre, width) per latent, as fractions of sequence length.
    uniform: V equal tiles. dyadic: latent 0 = whole protein, 1-2 = halves,
    3-6 = quarters, ...; leftover latents are extra whole-protein latents. With
    nesting, a prefix is then a coarser-resolution view of the protein."""
    if layout == "uniform":
        return [((v + 0.5) / V, 1.0 / V) for v in range(V)]
    out, level = [], 0
    while len(out) < V:
        n = 2 ** level
        if len(out) + n > V:
            out += [(0.5, 1.0)] * (V - len(out))
            break
        out += [((i + 0.5) / n, 0.5 / n) for i in range(n)]
        level += 1
    return out


class SegmentPooler(PerceiverPooler):
    """FLINT pooler whose latents attend to learnable contiguous segments.

    Each latent v has a centre mu_v in (0, 1) and width sigma_v (fractions of the
    protein's length); its cross-attention logits get the Gaussian bias
    -(x - mu_v)^2 / (2 sigma_v^2), x = relative residue position. Content-based
    attention still operates within the segment. Shared by query and document sides.
    """

    def __init__(self, dim, V, layout="dyadic", **kw):
        super().__init__(dim, V, V, **kw)
        mu, sigma = zip(*segment_layout(V, layout))
        mu = torch.tensor(mu).clamp(1e-3, 1 - 1e-3)
        self.mu_logit = torch.nn.Parameter(torch.log(mu / (1 - mu)))
        self.log_sigma = torch.nn.Parameter(torch.log(torch.tensor(sigma)))

    def position_bias(self, attention_mask, V):
        m = attention_mask.float()
        length = m.sum(1, keepdim=True)
        x = (m.cumsum(1) - 1) / (length - 1).clamp(min=1)                 # (B, L) in [0, 1]
        mu, sigma = torch.sigmoid(self.mu_logit), self.log_sigma.exp()
        return -((x[:, None, :] - mu[None, :, None]) ** 2) / (2 * sigma[None, :, None] ** 2)


class FixedBudget(torch.nn.Module):
    """ProteinFLINT behind the (vectors, mask) interface; mask is None (all V vectors are real)."""

    def __init__(self, flint):
        super().__init__()
        self.flint, self.pooler = flint, flint.pooler

    def forward(self, input_ids, attention_mask, mode="doc"):
        return self.flint(input_ids, attention_mask, mode=mode), None


# =============================================================================
# Batch sampling
# =============================================================================
class PairBatchSampler:
    """Yields batches of (anchor, positive) index pairs; each pair shares a superfamily.

    hier_frac = 0 reproduces ProtoCol's PairDataset (pick a random superfamily,
    then two of its members). hier_frac > 0 fills that fraction of each batch
    with pairs drawn from two different superfamilies of the same fold, so the
    batch contains same-fold positives (for fold-level groups) and hard
    same-fold negatives (for superfamily-level groups).
    """

    def __init__(self, proteins, batch_pairs, steps, hier_frac=0.0, seed=0):
        self.B, self.steps, self.hier_frac = batch_pairs, steps, hier_frac
        self.rng = random.Random(seed)
        by_sf = defaultdict(list)
        for i, p in enumerate(proteins):
            by_sf[p.superfamily].append(i)
        self.by_sf = {sf: idx for sf, idx in by_sf.items() if len(idx) >= 2}
        self.sfs = list(self.by_sf)
        fold_sfs = defaultdict(set)
        for sf, idx in self.by_sf.items():
            fold_sfs[scop_levels(proteins[idx[0]])["fold"]].add(sf)
        self.multi_folds = [sorted(s) for s in fold_sfs.values() if len(s) >= 2]
        if hier_frac > 0:
            print(f"Hierarchical sampler: {len(self.multi_folds)} folds with >= 2 superfamilies")

    def _pair(self, sf):
        return tuple(self.rng.sample(self.by_sf[sf], 2))

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            batch = []
            n_hier = int(round(self.hier_frac * self.B)) // 2 * 2
            while len(batch) < n_hier:
                for sf in self.rng.sample(self.rng.choice(self.multi_folds), 2):
                    batch.append(self._pair(sf))
            while len(batch) < self.B:
                batch.append(self._pair(self.rng.choice(self.sfs)))
            yield batch


# =============================================================================
# Loss
# =============================================================================
def mean_maxsim(Q, D, qm=None, dm=None):
    """Q: (q, a, d), D: (b, c, d) -> (q, b), MaxSim averaged over the a query vectors.
    qm: (q, a) / dm: (b, c) boolean masks of real vectors (None = all real, as in FLINT)."""
    sim = torch.einsum("qad,bcd->qbac", Q, D)
    if dm is not None:
        sim = sim.masked_fill(~dm[None, :, None, :], float("-inf"))
    best = sim.max(-1).values
    if qm is None:
        return best.mean(-1)
    qm = qm[:, None, :].to(best.dtype)
    return (best * qm).sum(-1) / qm.sum(-1)


def multi_positive_nce(logits, pos):
    """Symmetric InfoNCE where pos[i, j] marks positives (diagonal always included)."""
    logits = logits.float()
    neg_inf = torch.finfo(logits.dtype).min
    l_qd = torch.logsumexp(logits.masked_fill(~pos, neg_inf), 1) - torch.logsumexp(logits, 1)
    l_dq = torch.logsumexp(logits.T.masked_fill(~pos.T, neg_inf), 1) - torch.logsumexp(logits.T, 1)
    return -0.5 * (l_qd.mean() + l_dq.mean())


def nested_loss(Q, D, pos_by_level, groups, weights, temperature, qm=None, dm=None):
    """pos_by_level: {level: BoolTensor(B, B)}, [i, j] = query i and document j are
    homologous at that level. Returns (weighted mean loss, per-group losses)."""
    eye = torch.eye(Q.size(0), dtype=torch.bool, device=Q.device)
    total, parts = 0.0, []
    for (kq, kd, level), w in zip(groups, weights):
        logits = mean_maxsim(Q[:, :kq], D[:, :kd],
                             None if qm is None else qm[:, :kq], None if dm is None else dm[:, :kd]) / temperature
        loss = multi_positive_nce(logits, pos_by_level[level] | eye)
        total = total + w * loss
        parts.append(round(loss.item(), 4))
    return total / sum(weights), parts


def redundancy(X):
    """Mean squared off-diagonal cosine among each protein's V (unit-norm) vectors."""
    V = X.size(1)
    if V == 1:
        return X.new_zeros(())
    G = X @ X.transpose(1, 2)
    off = ~torch.eye(V, dtype=torch.bool, device=X.device)
    return (G[:, off] ** 2).mean()


# =============================================================================
# Training
# =============================================================================
def train(model, train_p, tokenizer, groups, args):
    steps_per_epoch = pe.PairDataset(train_p).epoch_size // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    sampler = PairBatchSampler(train_p, args.batch_size, total_steps,
                               hier_frac=args.hier_frac, seed=args.seed)
    labels_all = label_ids(train_p)
    weights = [1.0] * len(groups) if args.group_weights is None else args.group_weights
    assert len(weights) == len(groups)

    pooler = getattr(model, "pooler", None)
    pooler_ids = {id(p) for p in pooler.parameters()} if pooler is not None else set()
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in pooler_ids]
    pool = [p for p in pooler.parameters() if p.requires_grad] if pooler is not None else []
    opt = torch.optim.AdamW([{"params": base, "lr": args.lr}, {"params": pool, "lr": args.pooler_lr}],
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.pooler_lr],
                                                total_steps=total_steps, pct_start=args.warmup_frac)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]

    def tok(idx):
        enc = tokenizer([train_p[i].sequence for i in idx], padding=True, truncation=True,
                        max_length=args.max_len, return_tensors="pt")
        return {k: v.to(DEVICE) for k, v in enc.items()}

    log, ema = [], None
    model.train()
    for step, batch in enumerate(tqdm(sampler, desc="train", mininterval=10)):
        a_idx, p_idx = [b[0] for b in batch], [b[1] for b in batch]
        enc_a, enc_p = tok(a_idx), tok(p_idx)
        # Anchor and positive of a pair share superfamily (hence fold), so fold/sf
        # homology between query i and document j can use anchor labels; families
        # can differ within a pair, so fam compares query and document families.
        la = {lvl: labels_all[lvl][a_idx].to(DEVICE) for lvl in LEVELS}
        fam_p = labels_all["fam"][p_idx].to(DEVICE)
        pos = {"fold": la["fold"][:, None] == la["fold"][None, :],
               "sf": la["sf"][:, None] == la["sf"][None, :],
               "fam": la["fam"][:, None] == fam_p[None, :]}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            Q, qm = model(enc_a["input_ids"], enc_a["attention_mask"], mode="query")
            D, dm = model(enc_p["input_ids"], enc_p["attention_mask"], mode="doc")
        loss, parts = nested_loss(Q.float(), D.float(), pos, groups, weights, args.temperature, qm, dm)
        if args.diversity_weight > 0 and qm is None:
            loss = loss + args.diversity_weight * (redundancy(Q.float()) + redundancy(D.float())) / 2
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if (step + 1) % 100 == 0:
            log.append({"step": step + 1, "loss_ema": ema, "groups": parts})
    return log


# =============================================================================
# Evaluation
# =============================================================================
@torch.no_grad()
def score_matrix(Q, D, qm=None, dm=None, q_chunk=256, d_chunk=None):
    """All-pairs mean-MaxSim. Token-level inputs are chunked over documents too."""
    d_chunk = d_chunk or (D.size(0) if dm is None else 256)
    q_chunk = q_chunk if dm is None else 16
    rows = []
    for i in range(0, Q.size(0), q_chunk):
        q = Q[i:i + q_chunk].float()
        m = None if qm is None else qm[i:i + q_chunk]
        rows.append(torch.cat([mean_maxsim(q, D[j:j + d_chunk].float(), m,
                                           None if dm is None else dm[j:j + d_chunk])
                               for j in range(0, D.size(0), d_chunk)], dim=1))
    return torch.cat(rows)


@torch.no_grad()
def encode_tokens(model, proteins, tokenizer, max_len, batch_size=64):
    """Per-residue vectors padded to max_len: (n, max_len, d) fp16 and (n, max_len) mask."""
    model.eval()
    X, M = [], []
    for i in range(0, len(proteins), batch_size):
        enc = tokenizer([p.sequence for p in proteins[i:i + batch_size]], padding="max_length",
                        truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda", dtype=torch.float16):
            x, m = model(enc["input_ids"], enc["attention_mask"])
        X.append(x.half())
        M.append(m)
    return torch.cat(X), torch.cat(M)


@torch.no_grad()
def metrics_from_scores(S, labels, ks=pe.KS):
    """S: (nq, nd) scores of queries against a database whose first nq entries are
    the queries themselves (square when searching the test set against itself).
    labels: {level: LongTensor(nd)} over the database. Self-matches are removed."""
    nq, nd = S.shape
    n = nd
    S = S.clone()
    S[torch.arange(nq), torch.arange(nq)] = float("-inf")
    order = S.argsort(dim=1, descending=True)
    # Remove self explicitly: with -inf ties (e.g. outside a cascade shortlist) it need not be last.
    order = order[order != torch.arange(nq, device=S.device)[:, None]].view(nq, nd - 1)
    lab = {k: v.to(S.device) for k, v in labels.items()}
    same = {lvl: lab[lvl][order] == lab[lvl][:nq, None] for lvl in LEVELS}

    out = {}
    # Recall@k with superfamily labels, ProtoCol definition (denominator min(k, #relevant)).
    n_rel = same["sf"].sum(1)
    valid = n_rel > 0
    for k in ks:
        hits = same["sf"][:, :k].sum(1).float()
        out[f"R@{k}"] = (hits[valid] / n_rel[valid].clamp(max=k).float()).mean().item()

    # Sensitivity up to the first FP (different fold).
    fp = ~same["fold"]
    first_fp = torch.where(fp.any(1), fp.float().argmax(1), torch.full((nq,), n - 1, device=S.device))
    before = torch.arange(n - 1, device=S.device)[None, :] < first_fp[:, None]
    tp = {"fam": same["fam"],
          "sf": same["sf"] & ~same["fam"],
          "fold": same["fold"] & ~same["sf"]}
    for lvl, m in tp.items():
        total = m.sum(1)
        ok = total > 0
        out[f"sens_{lvl}"] = ((m & before).sum(1)[ok].float() / total[ok].float()).mean().item()
        out[f"n_{lvl}"] = int(ok.sum().item())
    return out


@torch.no_grad()
def vector_diversity(X, n=500):
    """How different are a protein's V vectors? Mean off-diagonal cosine and the
    participation-ratio effective rank (as in FLINT's Table 5), over n proteins."""
    X = X[:n].float()
    V = X.size(1)
    if V == 1:
        return {"mean_cos": 1.0, "eff_rank": 1.0}
    off = ~torch.eye(V, dtype=torch.bool, device=X.device)
    cos = (X @ X.transpose(1, 2))[:, off].mean().item()
    s = torch.linalg.svdvals(X)
    pr = (s.sum(1) ** 2 / (s ** 2).sum(1)).mean().item()
    return {"mean_cos": round(cos, 4), "eff_rank": round(pr, 3)}


@torch.no_grad()
def cascade_scores(Q, D, k_small, shortlist):
    """Shortlist with the first k_small vectors, rerank the shortlist with all vectors."""
    S_small = score_matrix(Q[:, :k_small], D[:, :k_small])
    S_small[torch.arange(Q.size(0)), torch.arange(Q.size(0))] = float("-inf")
    top = S_small.topk(shortlist, dim=1).indices
    S = torch.full_like(S_small, float("-inf"))
    for i in range(Q.size(0)):
        S[i, top[i]] = mean_maxsim(Q[i:i + 1].float(), D[top[i]].float())[0]
    return S


# =============================================================================
# Main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--latents", type=int, default=32, help="V")
    ap.add_argument("--pooler", choices=["perceiver", "segment"], default="perceiver",
                    help="segment: latents attend to learnable contiguous segments (see SegmentPooler)")
    ap.add_argument("--segment-layout", choices=["dyadic", "uniform"], default="dyadic")
    ap.add_argument("--diversity-weight", type=float, default=0.0,
                    help="penalty on squared cosine between a protein's pooled vectors (generic anti-collapse)")
    ap.add_argument("--token-level", action="store_true",
                    help="per-residue ColBERT (ProtoCol's ProteinColBERT) instead of FLINT pooling")
    ap.add_argument("--groups", default=None, help="default: '<V>:sf' (per-budget FLINT)")
    ap.add_argument("--group-weights", type=float, nargs="+", default=None)
    ap.add_argument("--hier-frac", type=float, default=0.0,
                    help="fraction of each batch drawn as same-fold / different-superfamily pairs")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--model-name", default=pe.MODEL_NAME)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--trainable-layers", type=int, default=3)
    ap.add_argument("--shared-latents", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32, help="pairs per batch")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--pooler-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=pe.MAX_LEN)
    ap.add_argument("--eval-ks", type=int, nargs="+", default=None,
                    help="prefix sizes to evaluate (default: powers of two up to V)")
    ap.add_argument("--cascades", nargs="*", default=["2:200", "4:200"],
                    help="k_small:shortlist pairs, reranked with all V vectors")
    ap.add_argument("--eval-db", action="store_true",
                    help="also search test queries against a test+train database (more fold-tier queries)")
    ap.add_argument("--seed", type=int, default=0, help="training seed")
    ap.add_argument("--split-seed", type=int, default=pe.SEED, help="train/test split seed")
    ap.add_argument("--output-dir", default="results/nested")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--save-model", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    V = args.max_len if args.token_level else args.latents  # token level: one group over all residues
    groups = parse_groups(args.groups or f"{V}:sf", V)
    out_dir = os.path.join(os.path.abspath(args.output_dir), args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(REPO_DIR)

    pe.SEED, pe.MAX_LEN = args.split_seed, args.max_len
    proteins = pe.load_scope()
    train_p, test_p = pe.split_by_superfamily(proteins, test_frac=0.15, min_members=3)
    test_p = pe.subsample_test_set(test_p, pe.MAX_TEST_PROTEINS)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    kind = "token-level" if args.token_level else f"FLINT V={V}"
    print(f"Run {args.run_name}: {kind} groups={groups} hier_frac={args.hier_frac}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.token_level:
        model = TokenLevel(args.model_name, args.proj_dim, args.trainable_layers).to(DEVICE)
    else:
        flint = ProteinFLINT(args.model_name, num_latents=V, proj_dim=args.proj_dim,
                             n_trainable_layers=args.trainable_layers, shared_latents=args.shared_latents)
        if args.pooler == "segment":
            flint.pooler = SegmentPooler(flint.backbone.config.hidden_size, V, layout=args.segment_layout,
                                         shared_latents=args.shared_latents)
        model = FixedBudget(flint).to(DEVICE)
    t0 = time.time()
    log = train(model, train_p, tokenizer, groups, args)
    train_min = (time.time() - t0) / 60

    labels = label_ids(test_p)
    rows = []
    if args.token_level:
        X, M = encode_tokens(model, test_p, tokenizer, args.max_len)
        diversity = {"vectors_per_protein": round(M.sum(1).float().mean().item(), 1)}
        rows.append({"eval": "full", **metrics_from_scores(score_matrix(X, X, M, M), labels)})
        eval_ks, args.cascades = [], []
    else:
        Q = encode_fixed(model.flint, test_p, tokenizer, "query", args.max_len)
        D = encode_fixed(model.flint, test_p, tokenizer, "doc", args.max_len)
        eval_ks = args.eval_ks or [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= V]
        diversity = vector_diversity(D)
    print(f"Vector diversity: {diversity}")

    for k in eval_ks:
        m = metrics_from_scores(score_matrix(Q[:, :k], D[:, :k]), labels)
        rows.append({"eval": f"prefix{k}", "kq": k, "kd": k, **m})
    for k in eval_ks:
        if k < V:  # small query budget against the full database
            m = metrics_from_scores(score_matrix(Q[:, :k], D), labels)
            rows.append({"eval": f"query{k}", "kq": k, "kd": V, **m})
    for c in args.cascades:
        k_small, shortlist = map(int, c.split(":"))
        if k_small < V:
            m = metrics_from_scores(cascade_scores(Q, D, k_small, shortlist), labels)
            rows.append({"eval": f"cascade{k_small}x{shortlist}", "kq": V, "kd": V, **m})

    if args.eval_db:
        # Queries = test proteins; database = test + train (queries first, so query i is db item i).
        db_p = test_p + train_p
        db_labels = label_ids(db_p)
        nq = len(test_p)
        if args.token_level:
            Xd, Md = encode_tokens(model, db_p, tokenizer, args.max_len)
            S = score_matrix(Xd[:nq], Xd, Md[:nq], Md)
            del Xd, Md
            rows.append({"eval": "db_full", **metrics_from_scores(S, db_labels)})
        else:
            Dd = encode_fixed(model.flint, db_p, tokenizer, "doc", args.max_len)
            rows.append({"eval": "db_full", **metrics_from_scores(score_matrix(Q, Dd), db_labels)})
            for c in args.cascades:
                k_small, shortlist = map(int, c.split(":"))
                if k_small < V:
                    m = metrics_from_scores(cascade_scores(Q, Dd, k_small, shortlist), db_labels)
                    rows.append({"eval": f"db_cascade{k_small}x{shortlist}", **m})
        torch.cuda.empty_cache()

    hdr = ["R@1", "R@10", "sens_fam", "sens_sf", "sens_fold"]
    print(f"\n{'eval':<16}" + "".join(f"{h:>10}" for h in hdr))
    for r in rows:
        print(f"{r['eval']:<16}" + "".join(f"{r[h]:>10.4f}" for h in hdr))

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"config": vars(args), "groups": groups, "n_test": len(test_p),
                   "train_minutes": train_min, "diversity": diversity, "rows": rows,
                   "train_log": log}, f, indent=2)
    if args.save_model:
        torch.save({"state_dict": model.state_dict(), "config": vars(args)}, os.path.join(out_dir, "model.pt"))
    print(f"Wrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
