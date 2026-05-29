#!/usr/bin/env python3
"""
ColBERT-style late-interaction retrieval for proteins using ESM-2 35M.
Supports either SCOPe or Pfam as the dataset.

Pipeline:
  1. Load proteins (SCOPe superfamilies or Pfam clans as homology labels).
  2. Split by group so train/test groups are disjoint.
  3. Subsample test set for tractable evaluation.
  4. Fine-tune last 3 layers of ESM-2 35M with ColBERT MaxSim + InfoNCE.
  5. Evaluate retrieval Recall@k.
  6. Run baselines for comparison:
     - Trained mean-pool single-vector (ColBERT paper Model [A])
     - MinHash Jaccard (sequence-only floor)
     - MMseqs2 (state-of-the-art sequence search)
     - Mean-pool ESM-2 650M frozen
  7. Latency benchmark (median per-query wall-clock time).

Installation
------------
    pip install -r requirements.txt
MMseqs2 is a system binary (installed via apt in eval_mmseqs), e.g.:
    sudo apt-get install -y mmseqs2
or:
    conda install -c bioconda mmseqs2
"""

import os
import re
import gzip
import time
import random
import shutil
import statistics
import subprocess
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
DATASET = "scope"            # "scope" or "pfam"
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
ESM_650M = "facebook/esm2_t33_650M_UR50D"
MAX_LEN = 256
MAX_TEST_PROTEINS = 3000
KS = (1, 5, 10, 100)

# MinHash baseline
KMER_K = 5
NUM_PERM = 256

# Latency benchmark
LATENCY_NUM_QUERIES = 100
LATENCY_WARMUP = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Protein:
    sid: str
    sequence: str
    superfamily: str            # SCOPe superfamily OR Pfam clan
    family: str = ""
    family_name: str = ""


# =============================================================================
# Dataset loaders
# =============================================================================
def load_scope() -> List[Protein]:
    scope_url = (
        "https://scop.berkeley.edu/downloads/scopeseq-2.08/"
        "astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
    )
    scope_path = "astral_scope_40_2.08.fa"
    if not os.path.exists(scope_path):
        print(f"Downloading {scope_url} ...")
        subprocess.run(["curl", "-kL", "-o", scope_path, scope_url], check=True)
    print(f"  {scope_path}: {os.path.getsize(scope_path) / 1e6:.1f} MB")

    proteins: List[Protein] = []
    state = {"sid": None, "sccs": None, "chunks": []}

    def flush():
        if state["sid"] is None:
            return
        seq = "".join(state["chunks"]).upper()
        if re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq) and 20 <= len(seq) <= 512:
            sf = ".".join(state["sccs"].split(".")[:3])
            proteins.append(
                Protein(sid=state["sid"], sequence=seq, superfamily=sf,
                        family=state["sccs"])
            )

    with open(scope_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                parts = line[1:].split()
                state["sid"] = parts[0]
                state["sccs"] = parts[1] if len(parts) > 1 else "x.x.x.x"
                state["chunks"] = []
            else:
                state["chunks"].append(line)
        flush()
    print(f"Parsed {len(proteins)} SCOPe proteins")
    return proteins


def load_pfam(max_per_family: int = 50) -> List[Protein]:
    pfam_base = "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release"
    seed_gz = "Pfam-A.seed.gz"
    clans_gz = "Pfam-A.clans.tsv.gz"
    for fname in (seed_gz, clans_gz):
        if not os.path.exists(fname):
            url = f"{pfam_base}/{fname}"
            print(f"Downloading {url} ...")
            subprocess.run(["curl", "-kL", "-o", fname, url], check=True)
        print(f"  {fname}: {os.path.getsize(fname) / 1e6:.1f} MB")

    fam2clan: Dict[str, str] = {}
    with gzip.open(clans_gz, "rt", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                fam2clan[parts[0]] = parts[1]
    print(f"Loaded {len(fam2clan)} family->clan mappings")

    proteins: List[Protein] = []
    family_acc: Optional[str] = None
    family_id: Optional[str] = None
    seqs: List[Tuple[str, str]] = []
    n_blocks = n_clan_blocks = 0

    with gzip.open(seed_gz, "rt", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("# STOCKHOLM"):
                family_acc, family_id, seqs = None, None, []
            elif line.startswith("#=GF AC"):
                family_acc = line.split()[2].split(".")[0]
            elif line.startswith("#=GF ID"):
                family_id = line.split(maxsplit=2)[2]
            elif line == "//":
                n_blocks += 1
                if family_acc and family_acc in fam2clan:
                    n_clan_blocks += 1
                    clan = fam2clan[family_acc]
                    for sid, gapped in seqs[:max_per_family]:
                        ungapped = re.sub(r"[^A-Za-z]", "", gapped).upper()
                        ungapped = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", ungapped)
                        if 20 <= len(ungapped) <= 512:
                            proteins.append(
                                Protein(sid=sid, sequence=ungapped,
                                        superfamily=clan, family=family_acc,
                                        family_name=family_id or "")
                            )
                family_acc, family_id, seqs = None, None, []
            elif line.startswith("#") or not line:
                continue
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    seqs.append((parts[0], parts[1]))

    n_clans = len({p.superfamily for p in proteins})
    n_fams = len({p.family for p in proteins})
    print(f"Total family blocks: {n_blocks}; in a clan: {n_clan_blocks}")
    print(f"Kept {len(proteins)} sequences from {n_fams} families in {n_clans} clans")
    return proteins


# =============================================================================
# Train/test split + subsampling
# =============================================================================
def split_by_superfamily(proteins, test_frac=0.15, min_members=3):
    by_sf = defaultdict(list)
    for p in proteins:
        by_sf[p.superfamily].append(p)
    sfs = [sf for sf, m in by_sf.items() if len(m) >= min_members]
    random.Random(SEED).shuffle(sfs)
    n_test = max(1, int(len(sfs) * test_frac))
    test_sfs = set(sfs[:n_test])
    train_sfs = set(sfs[n_test:])
    train = [p for sf in train_sfs for p in by_sf[sf]]
    test = [p for sf in test_sfs for p in by_sf[sf]]
    print(f"Total groups (>= {min_members} members): {len(sfs)}")
    print(f"Train: {len(train_sfs)} groups, {len(train)} proteins")
    print(f"Test:  {len(test_sfs)} groups, {len(test)} proteins")
    return train, test


def subsample_test_set(test_proteins, max_test=MAX_TEST_PROTEINS):
    if len(test_proteins) <= max_test:
        return test_proteins

    by_sf = defaultdict(list)
    for p in test_proteins:
        by_sf[p.superfamily].append(p)
    rng = random.Random(SEED)

    kept = []
    # First pass: 2 per group (every group gets retrievable positives).
    for members in by_sf.values():
        rng.shuffle(members)
        kept.extend(members[:2])
    # Second pass: random fill up to budget.
    leftovers = [p for members in by_sf.values() for p in members[2:]]
    rng.shuffle(leftovers)
    remaining = max_test - len(kept)
    if remaining > 0:
        kept.extend(leftovers[:remaining])

    n_groups = len({p.superfamily for p in kept})
    print(f"Subsampled test set to {len(kept)} proteins across {n_groups} groups")
    return kept


# =============================================================================
# Dataset + collation
# =============================================================================
class PairDataset(Dataset):
    def __init__(self, proteins, max_len=MAX_LEN):
        self.proteins = proteins
        self.max_len = max_len
        self.by_sf = defaultdict(list)
        for i, p in enumerate(proteins):
            self.by_sf[p.superfamily].append(i)
        self.valid_sfs = [sf for sf, idxs in self.by_sf.items() if len(idxs) >= 2]
        self.epoch_size = sum(len(self.by_sf[sf]) for sf in self.valid_sfs)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        sf = random.choice(self.valid_sfs)
        i, j = random.sample(self.by_sf[sf], 2)
        return self.proteins[i].sequence, self.proteins[j].sequence, sf


def make_collate_pairs(tokenizer, max_len=MAX_LEN):
    def collate_pairs(batch):
        anchors = [b[0] for b in batch]
        positives = [b[1] for b in batch]
        sfs = [b[2] for b in batch]
        enc_a = tokenizer(anchors, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
        enc_p = tokenizer(positives, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
        return enc_a, enc_p, sfs
    return collate_pairs


# =============================================================================
# Models
# =============================================================================
class ProteinColBERT(nn.Module):
    """Per-residue late-interaction encoder (MaxSim scoring)."""

    def __init__(self, model_name=MODEL_NAME, proj_dim=128, n_trainable_layers=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        self.proj_dim = proj_dim
        self._freeze_all_but_last_n(n_trainable_layers)

    def _freeze_all_but_last_n(self, n):
        for p in self.backbone.parameters():
            p.requires_grad = False
        total = len(self.backbone.encoder.layer)
        for i in range(total - n, total):
            for p in self.backbone.encoder.layer[i].parameters():
                p.requires_grad = True
        if getattr(self.backbone, "emb_layer_norm_after", None) is not None:
            for p in self.backbone.emb_layer_norm_after.parameters():
                p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
        n_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_a = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {n_t:,}/{n_a:,} ({100 * n_t / n_a:.1f}%)")

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state
        emb = self.proj(out)
        return F.normalize(emb, dim=-1)


class ProteinMeanPoolSingleVec(nn.Module):
    """Single-vector mean-pool encoder (ColBERT paper Model [A] baseline)."""

    def __init__(self, model_name=MODEL_NAME, proj_dim=128, n_trainable_layers=3):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim, bias=False)
        for p in self.backbone.parameters():
            p.requires_grad = False
        total = len(self.backbone.encoder.layer)
        for i in range(total - n_trainable_layers, total):
            for p in self.backbone.encoder.layer[i].parameters():
                p.requires_grad = True
        if getattr(self.backbone, "emb_layer_norm_after", None) is not None:
            for p in self.backbone.emb_layer_norm_after.parameters():
                p.requires_grad = True
        for p in self.proj.parameters():
            p.requires_grad = True
        n_t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_a = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {n_t:,}/{n_a:,} ({100 * n_t / n_a:.1f}%)")

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids,
                          attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()      # (B, L, 1)
        summed = (h * mask).sum(dim=1)                   # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)         # (B, 1)
        pooled = summed / counts                         # (B, H)
        return self.proj(pooled)


# =============================================================================
# Scoring + loss
# =============================================================================
def maxsim(q_emb, q_mask, d_emb, d_mask):
    sim = torch.einsum("qtd,bsd->qbts", q_emb, d_emb)
    d_mask_expand = d_mask[None, :, None, :].bool()
    sim = sim.masked_fill(~d_mask_expand, float("-inf"))
    max_over_d = sim.max(dim=-1).values
    q_mask_expand = q_mask[:, None, :].bool()
    max_over_d = max_over_d.masked_fill(~q_mask_expand, 0.0)
    return max_over_d.sum(dim=-1)


def contrastive_loss(scores, temperature=1.0):
    logits = scores / temperature
    labels = torch.arange(scores.size(0), device=scores.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.T, labels))


# =============================================================================
# Training loops
# =============================================================================
def train_colbert(model, train_ds, tokenizer, epochs=3, batch_size=16,
                  lr=2e-5, temperature=1.0):
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=make_collate_pairs(tokenizer),
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    total_steps = len(loader) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        running = 0.0
        for step, (enc_a, enc_p, _) in enumerate(pbar):
            enc_a = {k: v.to(DEVICE) for k, v in enc_a.items()}
            enc_p = {k: v.to(DEVICE) for k, v in enc_p.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                    dtype=torch.float16):
                q_emb = model(enc_a["input_ids"], enc_a["attention_mask"])
                d_emb = model(enc_p["input_ids"], enc_p["attention_mask"])
                scores = maxsim(q_emb, enc_a["attention_mask"],
                                d_emb, enc_p["attention_mask"])
                loss = contrastive_loss(scores, temperature=temperature)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running = 0.98 * running + 0.02 * loss.item() if step > 0 else loss.item()
            pbar.set_postfix(loss=f"{running:.4f}")


def train_singlevec(model, train_ds, tokenizer, epochs=3, batch_size=16,
                    lr=2e-5, temperature=1.0):
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=make_collate_pairs(tokenizer),
                        num_workers=2, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=len(loader) * epochs, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{epochs}")
        running = 0.0
        for step, (enc_a, enc_p, _) in enumerate(pbar):
            enc_a = {k: v.to(DEVICE) for k, v in enc_a.items()}
            enc_p = {k: v.to(DEVICE) for k, v in enc_p.items()}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                    dtype=torch.float16):
                q = model(enc_a["input_ids"], enc_a["attention_mask"])
                d = model(enc_p["input_ids"], enc_p["attention_mask"])
                logits = (q @ d.T) / temperature
                labels = torch.arange(logits.size(0), device=DEVICE)
                loss = 0.5 * (F.cross_entropy(logits, labels) +
                              F.cross_entropy(logits.T, labels))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running = 0.98 * running + 0.02 * loss.item() if step > 0 else loss.item()
            pbar.set_postfix(loss=f"{running:.4f}")


# =============================================================================
# Encoding
# =============================================================================
@torch.no_grad()
def encode_all(model, proteins, tokenizer, batch_size=32):
    """Per-residue encodings (list of variable-length tensors) for ColBERT eval."""
    model.eval()
    embeddings, labels = [], []
    for i in tqdm(range(0, len(proteins), batch_size), desc="encoding"):
        batch = proteins[i:i + batch_size]
        seqs = [p.sequence for p in batch]
        enc = tokenizer(seqs, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                dtype=torch.float16):
            emb = model(enc["input_ids"], enc["attention_mask"])
        mask = enc["attention_mask"].bool().cpu()
        emb = emb.float().cpu()
        for k in range(emb.size(0)):
            t = mask[k].sum().item()
            embeddings.append(emb[k, :t].clone())
            labels.append(batch[k].superfamily)
    return embeddings, labels


@torch.no_grad()
def encode_singlevec_model(model, proteins, tokenizer, batch_size=32):
    model.eval()
    vecs, labels = [], []
    for i in tqdm(range(0, len(proteins), batch_size), desc="encoding"):
        batch = proteins[i:i + batch_size]
        seqs = [p.sequence for p in batch]
        enc = tokenizer(seqs, padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                dtype=torch.float16):
            v = model(enc["input_ids"], enc["attention_mask"])
        vecs.append(v.float().cpu())
        labels.extend(p.superfamily for p in batch)
    return torch.cat(vecs, dim=0), labels


@torch.no_grad()
def encode_meanpool_650m(proteins, batch_size=8):
    tok = AutoTokenizer.from_pretrained(ESM_650M)
    base = AutoModel.from_pretrained(ESM_650M).to(DEVICE).eval()
    vecs, labels = [], []
    for i in tqdm(range(0, len(proteins), batch_size), desc="650M encoding"):
        batch = proteins[i:i + batch_size]
        enc = tok([p.sequence for p in batch], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                dtype=torch.float16):
            out = base(**enc).last_hidden_state
        special = ((enc["input_ids"] == tok.cls_token_id) |
                   (enc["input_ids"] == tok.eos_token_id))
        mask = (enc["attention_mask"].unsqueeze(-1).float() *
                (~special).unsqueeze(-1).float())
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        vecs.append(F.normalize(pooled.float(), dim=-1).cpu())
        labels.extend(p.superfamily for p in batch)
    del base
    torch.cuda.empty_cache()
    return torch.cat(vecs, dim=0), labels


# =============================================================================
# Retrieval evaluation
# =============================================================================
@torch.no_grad()
def evaluate_retrieval(embeddings, labels, ks=KS, chunk_size=64):
    """ColBERT MaxSim retrieval, Recall@k with a capped denominator."""
    n = len(embeddings)
    d = embeddings[0].size(-1)
    max_len = max(e.size(0) for e in embeddings)
    all_doc = torch.zeros(n, max_len, d)
    all_mask = torch.zeros(n, max_len, dtype=torch.bool)
    for i, e in enumerate(embeddings):
        all_doc[i, :e.size(0)] = e
        all_mask[i, :e.size(0)] = True
    all_doc = all_doc.to(DEVICE)
    all_mask = all_mask.to(DEVICE)

    sf_counts = defaultdict(int)
    for lab in labels:
        sf_counts[lab] += 1

    recalls = {k: [] for k in ks}
    max_k = max(ks)

    for qs in tqdm(range(0, n, chunk_size), desc="scoring"):
        qe = min(qs + chunk_size, n)
        q_lens = [embeddings[i].size(0) for i in range(qs, qe)]
        qmax = max(q_lens)
        q_pad = torch.zeros(qe - qs, qmax, d, device=DEVICE)
        q_mask = torch.zeros(qe - qs, qmax, dtype=torch.bool, device=DEVICE)
        for i, gi in enumerate(range(qs, qe)):
            q_pad[i, :q_lens[i]] = embeddings[gi].to(DEVICE)
            q_mask[i, :q_lens[i]] = True

        scores = torch.empty(qe - qs, n, device=DEVICE)
        doc_chunk = 256
        for ds in range(0, n, doc_chunk):
            de = min(ds + doc_chunk, n)
            scores[:, ds:de] = maxsim(q_pad, q_mask,
                                      all_doc[ds:de], all_mask[ds:de])

        for i, gi in enumerate(range(qs, qe)):
            scores[i, gi] = float("-inf")

        topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()
        for i, gi in enumerate(range(qs, qe)):
            q_label = labels[gi]
            total_rel = sf_counts[q_label] - 1
            if total_rel <= 0:
                continue
            retrieved = topk[i]
            for k in ks:
                hits = sum(1 for r in retrieved[:k] if labels[r] == q_label)
                recalls[k].append(hits / min(k, total_rel))

    return {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}


@torch.no_grad()
def eval_singlevec(vecs, labels, ks=KS, normalize=False):
    """Single-vector dot-product retrieval (CLS / mean-pool baselines)."""
    n = vecs.size(0)
    v = vecs.to(DEVICE)
    if normalize:
        v = F.normalize(v, dim=-1)
    scores = v @ v.T
    scores.fill_diagonal_(float("-inf"))
    sf_counts = defaultdict(int)
    for lab in labels:
        sf_counts[lab] += 1
    max_k = max(ks)
    topk = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()
    recalls = {k: [] for k in ks}
    for i in range(n):
        q = labels[i]
        tot = sf_counts[q] - 1
        if tot <= 0:
            continue
        retr = topk[i]
        for k in ks:
            hits = sum(1 for r in retr[:k] if labels[r] == q)
            recalls[k].append(hits / min(k, tot))
    return {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}


# =============================================================================
# Baseline: MinHash Jaccard
# =============================================================================
def build_mh(seq):
    from datasketch import MinHash
    m = MinHash(num_perm=NUM_PERM)
    for i in range(len(seq) - KMER_K + 1):
        m.update(seq[i:i + KMER_K].encode("utf-8"))
    return m


def eval_minhash(proteins, ks=KS):
    sigs = np.empty((len(proteins), NUM_PERM), dtype=np.uint64)
    labels = []
    for i, p in enumerate(tqdm(proteins, desc="MinHash")):
        sigs[i] = build_mh(p.sequence).hashvalues
        labels.append(p.superfamily)
    sigs_t = torch.from_numpy(sigs.astype(np.int64)).to(DEVICE)
    n = len(proteins)
    sf_counts = defaultdict(int)
    for lab in labels:
        sf_counts[lab] += 1
    max_k = max(ks)
    recalls = {k: [] for k in ks}
    chunk = 128
    for qs in tqdm(range(0, n, chunk), desc="scoring"):
        qe = min(qs + chunk, n)
        eq = (sigs_t[qs:qe, None, :] == sigs_t[None, :, :]).float()
        jacc = eq.mean(dim=-1)
        jacc[torch.arange(qe - qs), torch.arange(qs, qe)] = -1.0
        topk = torch.topk(jacc, k=max_k, dim=1).indices.cpu().tolist()
        for i, gi in enumerate(range(qs, qe)):
            q = labels[gi]
            tot = sf_counts[q] - 1
            if tot <= 0:
                continue
            retr = topk[i]
            for k in ks:
                hits = sum(1 for r in retr[:k] if labels[r] == q)
                recalls[k].append(hits / min(k, tot))
    return {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}


# =============================================================================
# Baseline: MMseqs2
# =============================================================================
def install_mmseqs():
    if shutil.which("mmseqs"):
        print("mmseqs already installed")
        return
    print("Installing mmseqs2...")
    subprocess.run(["apt-get", "install", "-y", "mmseqs2"], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Installed:", shutil.which("mmseqs"))


def write_fasta(proteins, path):
    with open(path, "w") as f:
        for i, p in enumerate(proteins):
            f.write(f">{i}\n{p.sequence}\n")


def eval_mmseqs(proteins, ks=KS, sensitivity=7.5):
    install_mmseqs()
    workdir = "/tmp/mmseqs_work"
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    fasta = f"{workdir}/seqs.fasta"
    write_fasta(proteins, fasta)

    out_tsv = f"{workdir}/results.m8"
    cmd = [
        "mmseqs", "easy-search", fasta, fasta, out_tsv, f"{workdir}/tmp",
        "-s", str(sensitivity),
        "--max-seqs", str(max(ks) + 1),
        "--format-output", "query,target,evalue",
        "-v", "1",
    ]
    print(f"Running MMseqs2 (sensitivity={sensitivity})...")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)

    hits = defaultdict(list)
    with open(out_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            q, t, ev = int(parts[0]), int(parts[1]), float(parts[2])
            if q != t:
                hits[q].append((t, ev))

    labels = [p.superfamily for p in proteins]
    sf_counts = defaultdict(int)
    for lab in labels:
        sf_counts[lab] += 1

    max_k = max(ks)
    recalls = {k: [] for k in ks}
    for qi in range(len(proteins)):
        q = labels[qi]
        tot = sf_counts[q] - 1
        if tot <= 0:
            continue
        ranked = sorted(hits.get(qi, []), key=lambda x: x[1])
        retr = [t for t, _ in ranked[:max_k]]
        for k in ks:
            hits_k = sum(1 for r in retr[:k] if labels[r] == q)
            recalls[k].append(hits_k / min(k, tot))
    return {k: sum(v) / len(v) if v else 0.0 for k, v in recalls.items()}


# =============================================================================
# Latency benchmark
# =============================================================================
def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_colbert(model, queries, corpus_emb, tokenizer, max_k,
                 n_warmup=LATENCY_WARMUP):
    model.eval()
    n = len(corpus_emb)
    d = corpus_emb[0].size(-1)
    max_len = max(e.size(0) for e in corpus_emb)
    all_doc = torch.zeros(n, max_len, d)
    all_mask = torch.zeros(n, max_len, dtype=torch.bool)
    for i, e in enumerate(corpus_emb):
        all_doc[i, :e.size(0)] = e
        all_mask[i, :e.size(0)] = True
    all_doc = all_doc.to(DEVICE)
    all_mask = all_mask.to(DEVICE)

    times_ms = []
    for qi, p in enumerate(queries):
        enc = tokenizer([p.sequence], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        _sync()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                dtype=torch.float16):
            q_emb = model(enc["input_ids"], enc["attention_mask"])
        scores = torch.empty(1, n, device=DEVICE)
        doc_chunk = 256
        for ds in range(0, n, doc_chunk):
            de = min(ds + doc_chunk, n)
            scores[:, ds:de] = maxsim(q_emb, enc["attention_mask"],
                                      all_doc[ds:de], all_mask[ds:de])
        _ = torch.topk(scores, k=max_k, dim=1).indices
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if qi >= n_warmup:
            times_ms.append(elapsed_ms)
    return times_ms


@torch.no_grad()
def time_trained_meanpool(model, queries, corpus_vecs, tokenizer, max_k,
                          n_warmup=LATENCY_WARMUP):
    model.eval()
    corpus = corpus_vecs.to(DEVICE)
    times_ms = []
    for qi, p in enumerate(queries):
        enc = tokenizer([p.sequence], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        _sync()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                dtype=torch.float16):
            qv = model(enc["input_ids"], enc["attention_mask"])
        qv = qv.float()
        scores = qv @ corpus.T
        _ = torch.topk(scores, k=max_k, dim=1).indices
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if qi >= n_warmup:
            times_ms.append(elapsed_ms)
    return times_ms


@torch.no_grad()
def time_meanpool_650m(queries, corpus_vecs, max_k, n_warmup=LATENCY_WARMUP):
    tok = AutoTokenizer.from_pretrained(ESM_650M)
    base = AutoModel.from_pretrained(ESM_650M).to(DEVICE).eval()
    corpus = corpus_vecs.to(DEVICE)
    times_ms = []
    for qi, p in enumerate(queries):
        enc = tok([p.sequence], padding=True, truncation=True,
                  max_length=MAX_LEN, return_tensors="pt").to(DEVICE)
        _sync()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda"),
                                dtype=torch.float16):
            out = base(**enc).last_hidden_state
        special = ((enc["input_ids"] == tok.cls_token_id) |
                   (enc["input_ids"] == tok.eos_token_id))
        mask = (enc["attention_mask"].unsqueeze(-1).float() *
                (~special).unsqueeze(-1).float())
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        qv = F.normalize(pooled.float(), dim=-1)
        scores = qv @ corpus.T
        _ = torch.topk(scores, k=max_k, dim=1).indices
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if qi >= n_warmup:
            times_ms.append(elapsed_ms)
    del base
    torch.cuda.empty_cache()
    return times_ms


def time_minhash(queries, corpus_proteins, max_k, n_warmup=LATENCY_WARMUP):
    corpus_sigs = np.empty((len(corpus_proteins), NUM_PERM), dtype=np.uint64)
    for i, p in enumerate(corpus_proteins):
        corpus_sigs[i] = build_mh(p.sequence).hashvalues
    corpus_t = torch.from_numpy(corpus_sigs.astype(np.int64)).to(DEVICE)

    times_ms = []
    for qi, p in enumerate(queries):
        _sync()
        t0 = time.perf_counter()
        qsig = build_mh(p.sequence).hashvalues
        qt = torch.from_numpy(qsig.astype(np.int64))[None, :].to(DEVICE)
        eq = (qt == corpus_t).float()
        jacc = eq.mean(dim=-1, keepdim=False)
        _ = torch.topk(jacc, k=max_k).indices
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if qi >= n_warmup:
            times_ms.append(elapsed_ms)
    return times_ms


def time_mmseqs(queries, corpus_proteins, max_k, n_warmup=LATENCY_WARMUP,
                sensitivity=7.5):
    install_mmseqs()
    workdir = "/tmp/mmseqs_latency"
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    corpus_fa = f"{workdir}/corpus.fasta"
    write_fasta(corpus_proteins, corpus_fa)

    # Build corpus DB once (the index; not counted in latency).
    corpus_db = f"{workdir}/corpusDB"
    subprocess.run(
        ["mmseqs", "createdb", corpus_fa, corpus_db, "-v", "1"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    times_ms = []
    for qi, p in enumerate(queries):
        qdir = f"{workdir}/q{qi}"
        os.makedirs(qdir, exist_ok=True)
        qfa = f"{qdir}/q.fasta"
        with open(qfa, "w") as f:
            f.write(f">q\n{p.sequence}\n")
        out_tsv = f"{qdir}/out.m8"
        tmpd = f"{qdir}/tmp"
        cmd = [
            "mmseqs", "easy-search", qfa, corpus_db, out_tsv, tmpd,
            "-s", str(sensitivity),
            "--max-seqs", str(max_k + 1),
            "--format-output", "query,target,evalue",
            "-v", "1",
        ]
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if qi >= n_warmup:
            times_ms.append(elapsed_ms)
    return times_ms


# =============================================================================
# Summary printing
# =============================================================================
def print_retrieval_summary(all_results, n_test):
    print("\n" + "=" * 78)
    print(f"  SUMMARY ({DATASET.upper()}, n_test = {n_test})")
    print("=" * 78)
    header = f"{'Method':<32} " + " ".join(f"R@{k:<5}" for k in [1, 5, 10, 100])
    print(header)
    print("-" * len(header))
    for name, results in all_results.items():
        row = f"{name:<32} " + " ".join(f"{results[k]:.4f} " for k in [1, 5, 10, 100])
        print(row)


def print_combined_summary(all_results, latency_results, n_test, n_queries_timed):
    print("\n" + "=" * 92)
    print(f"  LATENCY + RETRIEVAL SUMMARY ({DATASET.upper()}, "
          f"n_test={n_test}, n_queries_timed={n_queries_timed})")
    print("=" * 92)
    header = (f"{'Method':<32} " + " ".join(f"R@{k:<5}" for k in [1, 5, 10, 100])
              + "  median ms/query")
    print(header)
    print("-" * len(header))
    for name, results in all_results.items():
        lat = latency_results.get(name)
        lat_str = f"{lat:>10.2f}" if lat is not None else f"{'n/a':>10}"
        row = (f"{name:<32} "
               + " ".join(f"{results[k]:.4f} " for k in [1, 5, 10, 100])
               + f"  {lat_str}")
        print(row)


# =============================================================================
# Main
# =============================================================================
def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Using device: {DEVICE}")

    # --- Load + split data --------------------------------------------------
    if DATASET == "scope":
        proteins = load_scope()
    elif DATASET == "pfam":
        proteins = load_pfam(max_per_family=50)
    else:
        raise ValueError(f"Unknown DATASET: {DATASET!r}")

    lengths = [len(p.sequence) for p in proteins]
    sf_sizes = Counter(p.superfamily for p in proteins)
    print(f"\nLength: median={int(np.median(lengths))}, max={max(lengths)}")
    print(f"Groups: {len(sf_sizes)}, largest={max(sf_sizes.values())}, "
          f"median={int(np.median(list(sf_sizes.values())))}")

    train_proteins, test_proteins = split_by_superfamily(
        proteins, test_frac=0.15, min_members=3)
    test_proteins = subsample_test_set(test_proteins, MAX_TEST_PROTEINS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PairDataset(train_proteins)
    n_epochs = 1 if DATASET == "pfam" else 3

    all_results: Dict[str, Dict[int, float]] = {}

    # --- ColBERT: untrained baseline ---------------------------------------
    print("\n=== ColBERT: baseline (no training) ===")
    model = ProteinColBERT(MODEL_NAME, proj_dim=128, n_trainable_layers=3).to(DEVICE)
    test_emb, test_labels = encode_all(model, test_proteins, tokenizer, batch_size=32)
    all_results["ColBERT (untrained)"] = evaluate_retrieval(test_emb, test_labels)
    for k, v in all_results["ColBERT (untrained)"].items():
        print(f"R@{k}: {v:.4f}")

    # --- ColBERT: train + evaluate -----------------------------------------
    print("\n=== ColBERT: training ===")
    train_colbert(model, train_ds, tokenizer, epochs=n_epochs,
                  batch_size=16, lr=2e-5, temperature=1.0)

    print("\n=== ColBERT: after fine-tuning ===")
    test_emb, test_labels = encode_all(model, test_proteins, tokenizer, batch_size=32)
    all_results["ColBERT (trained)"] = evaluate_retrieval(test_emb, test_labels)
    for k, v in all_results["ColBERT (trained)"].items():
        print(f"R@{k}: {v:.4f}")

    # --- Baseline 1: trained mean-pool single-vector (ColBERT Model [A]) ----
    print("\n=== Baseline: Trained mean-pool single-vector (ColBERT Model [A]) ===")
    mp_model = ProteinMeanPoolSingleVec().to(DEVICE)
    train_singlevec(mp_model, train_ds, tokenizer, epochs=n_epochs)
    mp_v, mp_l = encode_singlevec_model(mp_model, test_proteins, tokenizer)
    all_results["Trained mean-pool single-vec"] = eval_singlevec(
        mp_v, mp_l, normalize=False)
    for k, v in all_results["Trained mean-pool single-vec"].items():
        print(f"R@{k}: {v:.4f}")
    torch.cuda.empty_cache()

    # --- Baseline 2: MinHash Jaccard ---------------------------------------
    print("\n=== Baseline: MinHash Jaccard (k=5) ===")
    all_results["MinHash Jaccard"] = eval_minhash(test_proteins)
    for k, v in all_results["MinHash Jaccard"].items():
        print(f"R@{k}: {v:.4f}")

    # --- Baseline 3: MMseqs2 -----------------------------------------------
    print("\n=== Baseline: MMseqs2 (sensitivity 7.5) ===")
    all_results["MMseqs2"] = eval_mmseqs(test_proteins)
    for k, v in all_results["MMseqs2"].items():
        print(f"R@{k}: {v:.4f}")

    # --- Baseline 4: mean-pool ESM-2 650M (frozen) -------------------------
    print("\n=== Baseline: Mean-pool ESM-2 650M (frozen) ===")
    esm650_v, esm650_l = encode_meanpool_650m(test_proteins)
    all_results["Mean-pool ESM-2 650M"] = eval_singlevec(
        esm650_v, esm650_l, normalize=False)
    for k, v in all_results["Mean-pool ESM-2 650M"].items():
        print(f"R@{k}: {v:.4f}")

    # --- Retrieval summary --------------------------------------------------
    print_retrieval_summary(all_results, len(test_proteins))

    # --- Latency benchmark --------------------------------------------------
    # `test_emb` currently holds the trained-ColBERT corpus encodings.
    max_k = max(KS)
    n_queries = min(LATENCY_NUM_QUERIES, len(test_proteins))
    rng_lat = random.Random(SEED)
    query_indices = rng_lat.sample(range(len(test_proteins)), n_queries)
    query_proteins = [test_proteins[i] for i in query_indices]
    latency_results: Dict[str, float] = {}

    print("\n=== Latency: ColBERT (trained) ===")
    t = time_colbert(model, query_proteins, test_emb, tokenizer, max_k)
    latency_results["ColBERT (trained)"] = statistics.median(t)
    print(f"  median: {latency_results['ColBERT (trained)']:.2f} ms/query (n={len(t)})")

    print("\n=== Latency: ColBERT (untrained) ===")
    untrained_model = ProteinColBERT(
        MODEL_NAME, proj_dim=128, n_trainable_layers=3).to(DEVICE)
    untrained_emb, _ = encode_all(untrained_model, test_proteins, tokenizer,
                                  batch_size=32)
    t = time_colbert(untrained_model, query_proteins, untrained_emb, tokenizer, max_k)
    latency_results["ColBERT (untrained)"] = statistics.median(t)
    print(f"  median: {latency_results['ColBERT (untrained)']:.2f} ms/query (n={len(t)})")
    del untrained_model, untrained_emb
    torch.cuda.empty_cache()

    print("\n=== Latency: Trained mean-pool single-vec ===")
    trained_mp_v, _ = encode_singlevec_model(mp_model, test_proteins, tokenizer)
    t = time_trained_meanpool(mp_model, query_proteins, trained_mp_v, tokenizer, max_k)
    latency_results["Trained mean-pool single-vec"] = statistics.median(t)
    print(f"  median: {latency_results['Trained mean-pool single-vec']:.2f} "
          f"ms/query (n={len(t)})")
    del trained_mp_v
    torch.cuda.empty_cache()

    print("\n=== Latency: Mean-pool ESM-2 650M (frozen) ===")
    t = time_meanpool_650m(query_proteins, esm650_v, max_k)
    latency_results["Mean-pool ESM-2 650M"] = statistics.median(t)
    print(f"  median: {latency_results['Mean-pool ESM-2 650M']:.2f} ms/query (n={len(t)})")

    print("\n=== Latency: MinHash Jaccard ===")
    t = time_minhash(query_proteins, test_proteins, max_k)
    latency_results["MinHash Jaccard"] = statistics.median(t)
    print(f"  median: {latency_results['MinHash Jaccard']:.2f} ms/query (n={len(t)})")

    print("\n=== Latency: MMseqs2 ===")
    t = time_mmseqs(query_proteins, test_proteins, max_k)
    latency_results["MMseqs2"] = statistics.median(t)
    print(f"  median: {latency_results['MMseqs2']:.2f} ms/query (n={len(t)})")

    # --- Combined summary ---------------------------------------------------
    print_combined_summary(all_results, latency_results, len(test_proteins),
                           n_queries - LATENCY_WARMUP)


if __name__ == "__main__":
    main()
