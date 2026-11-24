#!/usr/bin/env python3
"""
Fixed-budget late interaction (FLINT) for protein retrieval.

FLINT (Handa, Arabzadeh, Zaharia; arXiv 2610.27273) pools the per-token
encoder outputs of every query and every document into the same fixed number
V of vectors, using V learned latents that cross-attend over the tokens.
MaxSim scoring is unchanged, but the index costs V vectors per protein instead
of one per residue, and scoring cost no longer grows with sequence length.

This script plugs that pooling into the ProtoCol setup:

    ESM-2 (last N layers trainable)
      -> per-residue embeddings (B, L, H)
      -> perceiver pooling: V latents x cross-attention + FF   (B, V, H)
      -> linear projection to proj_dim, L2-normalise           (B, V, d)
      -> MaxSim over the V x V vector pairs, in-batch InfoNCE

Query and document sides have separate learned latents (as in FLINT); pass
--shared-latents to use one set, which makes scoring symmetric.

Data loading, the group-disjoint split, test subsampling, and the pair
sampler are imported from protocol_experiments.py, so numbers are comparable
with the token-level ColBERT run there (same seed -> same split).

Example:
    python flint_protein.py --dataset scope --latents 32
    python flint_protein.py --dataset pfam --latents 16 --epochs 1 --with-colbert-baseline
"""

import argparse
import json
import os
import random
import statistics
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import protocol_experiments as pe

DEVICE = pe.DEVICE
REPO_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Model
# =============================================================================
class PerceiverPooler(nn.Module):
    """V learned latents cross-attend over token embeddings (one CA + FF block).

    Port of FLINT's SharedPerceiverBottleneck (colbert/modeling/flint_perceiver.py,
    sdpa path): pre-LN cross-attention with a residual on the latents, then a
    pre-LN feed-forward block with a residual.
    """

    def __init__(self, dim, num_query_latents=32, num_doc_latents=32, num_heads=8,
                 ff_ratio=4, dropout=0.1, shared_latents=False):
        super().__init__()
        assert dim % num_heads == 0, f"hidden size {dim} not divisible by {num_heads} heads"
        self.dim, self.num_heads, self.head_dim = dim, num_heads, dim // num_heads
        self.attn_dropout = dropout

        self.query_latents = nn.Parameter(torch.empty(num_query_latents, dim))
        if shared_latents:
            assert num_query_latents == num_doc_latents
            self.doc_latents = self.query_latents
        else:
            self.doc_latents = nn.Parameter(torch.empty(num_doc_latents, dim))

        self.norm_latents = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * ff_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * ff_ratio), dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.query_latents, std=0.02)
        nn.init.normal_(self.doc_latents, std=0.02)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.out_proj,
                    self.ff[0], self.ff[2]):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, tokens, attention_mask, mode="doc"):
        B, L, _ = tokens.shape
        latents = self.query_latents if mode == "query" else self.doc_latents
        V = latents.size(0)
        latents = latents.unsqueeze(0).expand(B, -1, -1)

        q = self.norm_latents(latents)
        kv = self.norm_context(tokens)
        H, hd = self.num_heads, self.head_dim
        Q = self.q_proj(q).view(B, V, H, hd).transpose(1, 2)
        K = self.k_proj(kv).view(B, L, H, hd).transpose(1, 2)
        Vv = self.v_proj(kv).view(B, L, H, hd).transpose(1, 2)
        # Boolean mask: True = attend. Padding residues are never attended.
        attn_mask = attention_mask.bool()[:, None, None, :]
        out = F.scaled_dot_product_attention(
            Q, K, Vv, attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0)
        out = self.out_proj(out.transpose(1, 2).reshape(B, V, self.dim))

        latents = latents + self.dropout(out)
        return latents + self.ff(self.norm_ff(latents))


def freeze_all_but_last_n(backbone, n):
    for p in backbone.parameters():
        p.requires_grad = False
    layers = backbone.encoder.layer
    for layer in layers[len(layers) - n:]:
        for p in layer.parameters():
            p.requires_grad = True
    if getattr(backbone, "emb_layer_norm_after", None) is not None:
        for p in backbone.emb_layer_norm_after.parameters():
            p.requires_grad = True


class ProteinFLINT(nn.Module):
    """ESM-2 + perceiver pooling: every protein becomes V vectors of size proj_dim."""

    def __init__(self, model_name=pe.MODEL_NAME, num_latents=32, proj_dim=128,
                 n_trainable_layers=3, num_heads=8, ff_ratio=4, dropout=0.1,
                 shared_latents=False, separate_proj=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.num_latents = num_latents
        self.pooler = PerceiverPooler(hidden, num_latents, num_latents, num_heads=num_heads,
                                      ff_ratio=ff_ratio, dropout=dropout,
                                      shared_latents=shared_latents)
        self.query_proj = nn.Linear(hidden, proj_dim, bias=False)
        self.doc_proj = (nn.Linear(hidden, proj_dim, bias=False) if separate_proj
                         else self.query_proj)
        freeze_all_but_last_n(self.backbone, n_trainable_layers)

        n_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_a = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {n_t:,}/{n_a:,} ({100 * n_t / n_a:.1f}%)")

    def forward(self, input_ids, attention_mask, mode="doc"):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        z = self.pooler(h, attention_mask, mode=mode)
        proj = self.query_proj if mode == "query" else self.doc_proj
        return F.normalize(proj(z), dim=-1)          # (B, V, proj_dim)


# =============================================================================
# Scoring + loss
# =============================================================================
def fixed_maxsim(Q, D):
    """Q: (q, V, d), D: (b, W, d) -> (q, b). No masks: every vector is real."""
    sim = torch.einsum("qvd,bwd->qbvw", Q, D)
    return sim.max(dim=-1).values.sum(dim=-1)


def info_nce(scores, labels=None, temperature=1.0):
    """Symmetric in-batch InfoNCE (as in ProtoCol). If labels are given, other
    in-batch items from the anchor's group are masked out instead of being
    treated as negatives."""
    logits = scores.float() / temperature
    if labels is not None:
        same = labels[:, None] == labels[None, :]
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


# =============================================================================
# Training
# =============================================================================
def train_flint(model, train_ds, tokenizer, args):
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=pe.make_collate_pairs(tokenizer, max_len=args.max_len),
                        num_workers=2, drop_last=True)

    # As in FLINT: the pooling module gets its own (higher) learning rate.
    pooler_ids = {id(p) for p in model.pooler.parameters()}
    base = [p for p in model.parameters() if p.requires_grad and id(p) not in pooler_ids]
    pool = [p for p in model.pooler.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": base, "lr": args.lr},
                             {"params": pool, "lr": args.pooler_lr}],
                            weight_decay=args.weight_decay)
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lr, args.pooler_lr], total_steps=total_steps,
        pct_start=args.warmup_frac)
    use_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]

    log = []
    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        running = None
        for enc_a, enc_p, sfs in pbar:
            enc_a = {k: v.to(DEVICE) for k, v in enc_a.items()}
            enc_p = {k: v.to(DEVICE) for k, v in enc_p.items()}
            labels = None
            if args.mask_false_negatives:
                ids = {sf: i for i, sf in enumerate(dict.fromkeys(sfs))}
                labels = torch.tensor([ids[sf] for sf in sfs], device=DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                Q = model(enc_a["input_ids"], enc_a["attention_mask"], mode="query")
                D = model(enc_p["input_ids"], enc_p["attention_mask"], mode="doc")
                scores = fixed_maxsim(Q, D)
            loss = info_nce(scores, labels, temperature=args.temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
            pbar.set_postfix(loss=f"{running:.4f}")
            if step % 50 == 0:
                log.append({"step": step, "epoch": epoch, "loss_ema": running})
    return log


# =============================================================================
# Encoding + evaluation
# =============================================================================
@torch.no_grad()
def encode_fixed(model, proteins, tokenizer, mode, max_len, batch_size=64):
    """Returns a dense (n, V, d) fp16 tensor on DEVICE; no masks needed."""
    model.eval()
    out = []
    for i in tqdm(range(0, len(proteins), batch_size), desc=f"encoding ({mode})"):
        seqs = [p.sequence for p in proteins[i:i + batch_size]]
        enc = tokenizer(seqs, padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda", dtype=torch.float16):
            out.append(model(enc["input_ids"], enc["attention_mask"], mode=mode).half())
    return torch.cat(out)


@torch.no_grad()
def evaluate_fixed(Q, D, labels, ks=pe.KS, q_chunk=64, d_chunk=1024):
    """All-vs-all retrieval over the test set, same protocol as
    protocol_experiments.evaluate_retrieval: self excluded, Recall@k with
    denominator min(k, #relevant)."""
    n = Q.size(0)
    counts = defaultdict(int)
    for lab in labels:
        counts[lab] += 1
    recalls = {k: [] for k in ks}
    max_k = max(ks)
    for qs in tqdm(range(0, n, q_chunk), desc="scoring"):
        qe = min(qs + q_chunk, n)
        scores = torch.cat([fixed_maxsim(Q[qs:qe].float(), D[ds:ds + d_chunk].float())
                            for ds in range(0, n, d_chunk)], dim=1)
        scores[torch.arange(qe - qs), torch.arange(qs, qe)] = float("-inf")
        topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()
        for i, gi in enumerate(range(qs, qe)):
            total = counts[labels[gi]] - 1
            if total <= 0:
                continue
            for k in ks:
                hits = sum(1 for r in topk[i][:k] if labels[r] == labels[gi])
                recalls[k].append(hits / min(k, total))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in recalls.items()}


@torch.no_grad()
def time_flint(model, queries, D, tokenizer, max_len, max_k, n_warmup=pe.LATENCY_WARMUP):
    """Median per-query time: encode one query + score the whole corpus + top-k."""
    model.eval()
    times = []
    for qi, p in enumerate(queries):
        enc = tokenizer([p.sequence], truncation=True, max_length=max_len,
                        return_tensors="pt").to(DEVICE)
        pe._sync()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda", dtype=torch.float16):
            q = model(enc["input_ids"], enc["attention_mask"], mode="query").half()
            scores = fixed_maxsim(q, D)
        torch.topk(scores.float(), k=max_k, dim=1)
        pe._sync()
        if qi >= n_warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return times


# =============================================================================
# Main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", choices=["scope", "pfam"], default="scope")
    ap.add_argument("--model-name", default=pe.MODEL_NAME)
    ap.add_argument("--latents", type=int, default=32, help="V: vectors per query and per document")
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--trainable-layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff-ratio", type=float, default=4)
    ap.add_argument("--pooler-dropout", type=float, default=0.1)
    ap.add_argument("--shared-latents", action="store_true",
                    help="one latent set for queries and documents (symmetric scoring)")
    ap.add_argument("--separate-proj", action="store_true",
                    help="separate query/doc projections (FLINT-BERT); default shared (FLINT-ModernBERT)")
    ap.add_argument("--epochs", type=int, default=None, help="default: 3 for scope, 1 for pfam")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5, help="backbone + projection")
    ap.add_argument("--pooler-lr", type=float, default=1e-4, help="perceiver pooling module")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--mask-false-negatives", action="store_true",
                    help="don't treat same-group in-batch items as negatives")
    ap.add_argument("--max-len", type=int, default=pe.MAX_LEN)
    ap.add_argument("--max-test", type=int, default=pe.MAX_TEST_PROTEINS)
    ap.add_argument("--max-train-pairs", type=int, default=None,
                    help="cap pairs per epoch (for smoke tests)")
    ap.add_argument("--seed", type=int, default=pe.SEED)
    ap.add_argument("--with-colbert-baseline", action="store_true",
                    help="also train/eval the token-level ProtoCol ColBERT on the same split")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--output-dir", default="results/flint")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-save-model", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.epochs is None:
        args.epochs = 1 if args.dataset == "pfam" else 3
    run_name = args.run_name or f"{args.dataset}_V{args.latents}_s{args.seed}"
    out_dir = os.path.join(os.path.abspath(args.output_dir), run_name)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(REPO_DIR)  # dataset files are read relative to the repo

    # protocol_experiments reads these module globals at call time.
    pe.SEED, pe.MAX_LEN = args.seed, args.max_len
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"Using device: {DEVICE}\nRun: {out_dir}")

    proteins = pe.load_scope() if args.dataset == "scope" else pe.load_pfam(max_per_family=50)
    train_p, test_p = pe.split_by_superfamily(proteins, test_frac=0.15, min_members=3)
    test_p = pe.subsample_test_set(test_p, args.max_test)
    test_labels = [p.superfamily for p in test_p]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = pe.PairDataset(train_p, max_len=args.max_len)
    if args.max_train_pairs:
        train_ds.epoch_size = min(train_ds.epoch_size, args.max_train_pairs)

    results = {"config": vars(args), "n_test": len(test_p), "methods": {}}

    # --- FLINT ---------------------------------------------------------------
    print(f"\n=== FLINT (V={args.latents}): training ===")
    model = ProteinFLINT(args.model_name, num_latents=args.latents, proj_dim=args.proj_dim,
                         n_trainable_layers=args.trainable_layers, num_heads=args.heads,
                         ff_ratio=args.ff_ratio, dropout=args.pooler_dropout,
                         shared_latents=args.shared_latents,
                         separate_proj=args.separate_proj).to(DEVICE)
    t0 = time.time()
    train_log = train_flint(model, train_ds, tokenizer, args)
    train_min = (time.time() - t0) / 60

    print(f"\n=== FLINT (V={args.latents}): evaluation ===")
    Q = encode_fixed(model, test_p, tokenizer, "query", args.max_len)
    D = encode_fixed(model, test_p, tokenizer, "doc", args.max_len)
    rec = evaluate_fixed(Q, D, test_labels)
    n_vec = args.latents
    flint_row = {"recall": rec, "vectors_per_protein": n_vec,
                 "index_mb_fp16": D.numel() * 2 / 1e6, "train_minutes": train_min}
    results["methods"][f"FLINT (V={args.latents})"] = flint_row
    for k, v in rec.items():
        print(f"R@{k}: {v:.4f}")

    # --- Optional token-level ColBERT baseline on the same split -------------
    colbert = None
    if args.with_colbert_baseline:
        print("\n=== ColBERT (token-level): training ===")
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        colbert = pe.ProteinColBERT(args.model_name, proj_dim=args.proj_dim,
                                    n_trainable_layers=args.trainable_layers).to(DEVICE)
        t0 = time.time()
        pe.train_colbert(colbert, train_ds, tokenizer, epochs=args.epochs,
                         batch_size=args.batch_size, lr=args.lr, temperature=args.temperature)
        cb_min = (time.time() - t0) / 60
        cb_emb, cb_labels = pe.encode_all(colbert, test_p, tokenizer, batch_size=32)
        cb_rec = pe.evaluate_retrieval(cb_emb, cb_labels)
        n_tok = sum(e.size(0) for e in cb_emb)
        results["methods"]["ColBERT (token-level)"] = {
            "recall": cb_rec, "vectors_per_protein": n_tok / len(cb_emb),
            "index_mb_fp16": n_tok * args.proj_dim * 2 / 1e6, "train_minutes": cb_min}
        for k, v in cb_rec.items():
            print(f"R@{k}: {v:.4f}")

    # --- Latency -------------------------------------------------------------
    if not args.skip_latency:
        n_q = min(pe.LATENCY_NUM_QUERIES, len(test_p))
        queries = [test_p[i] for i in random.Random(args.seed).sample(range(len(test_p)), n_q)]
        t = time_flint(model, queries, D, tokenizer, args.max_len, max(pe.KS))
        flint_row["latency_ms_median"] = statistics.median(t)
        if colbert is not None:
            t = pe.time_colbert(colbert, queries, cb_emb, tokenizer, max(pe.KS))
            results["methods"]["ColBERT (token-level)"]["latency_ms_median"] = statistics.median(t)

    # --- Summary + outputs ---------------------------------------------------
    ks = list(pe.KS)
    print(f"\n{'Method':<24}" + "".join(f"{'R@' + str(k):>8}" for k in ks)
          + f"{'vec/prot':>10}{'index MB':>10}{'ms/q':>8}")
    for name, row in results["methods"].items():
        lat = row.get("latency_ms_median")
        print(f"{name:<24}" + "".join(f"{row['recall'][k]:>8.4f}" for k in ks)
              + f"{row['vectors_per_protein']:>10.1f}{row['index_mb_fp16']:>10.1f}"
              + (f"{lat:>8.2f}" if lat is not None else f"{'-':>8}"))

    results["train_log"] = train_log
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    if not args.no_save_model:
        torch.save({"state_dict": model.state_dict(), "config": vars(args)},
                   os.path.join(out_dir, "model.pt"))
    print(f"\nWrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
