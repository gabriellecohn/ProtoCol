"""Probe the largest trainable ``batch_size`` for a given model/training config.

Runs one forward+backward pass at each candidate batch size in a fresh
subprocess, so an OOM from one size doesn't poison CUDA state for the next.

Usage:
    python -m scripts.probe_max_batch \\
        --config configs/train/stage_a_pfam.yaml \\
        --batch-sizes 4 8 12 16 18 \\
        --negatives 15 --seq-len 512
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


CHILD = r"""
import argparse, json, os, sys, random, string
import torch
sys.path.insert(0, {repo!r})

from src.train.trainer_utils import load_training_bundle, build_retriever, TokenizedCollator
from src.models.losses import multi_positive_softmax_loss

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--negatives", type=int, required=True)
p.add_argument("--seq-len", type=int, required=True)
args = p.parse_args()

train_cfg, model_cfg = load_training_bundle(args.config)
device = "cuda:0"
model = build_retriever(model_cfg).to(device)
model.train()

def rand_seq(n):
    return "".join(random.choice(string.ascii_uppercase[:20]) for _ in range(n))

B, K, L = args.batch_size, args.negatives, args.seq_len
queries = [rand_seq(L) for _ in range(B)]
positives = [rand_seq(L) for _ in range(B)]
negatives = [[rand_seq(L) for _ in range(K)] for _ in range(B)]

tok = model.backbone.tokenize
batch_items = [
    {{
        "query_id": f"q{{i}}", "query_sequence": queries[i],
        "positive_id": f"p{{i}}", "positive_sequence": positives[i],
        "negative_ids": [f"n{{i}}_{{j}}" for j in range(K)],
        "negative_sequences": negatives[i],
    }}
    for i in range(B)
]
collator = TokenizedCollator(tok)
batch = collator(batch_items)

torch.cuda.reset_peak_memory_stats()
scores = model.score_token_batches(batch["query_tokens"], batch["doc_tokens"])
loss = multi_positive_softmax_loss(scores, batch["positive_mask"].to(scores.device),
                                   temperature=float(model_cfg.get("score_temperature", 0.1)))
loss.backward()
peak = torch.cuda.max_memory_allocated() / (1024**3)
print(json.dumps({{"batch_size": B, "peak_gib": peak, "ok": True}}))
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 12, 16, 18])
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()

    child_src = CHILD.format(repo=str(REPO))
    print(f"{'batch':>6}  {'peak (GiB)':>12}  status")
    print("-" * 40)
    for bs in args.batch_sizes:
        proc = subprocess.run(
            [sys.executable, "-c", child_src,
             "--config", args.config,
             "--batch-size", str(bs),
             "--negatives", str(args.negatives),
             "--seq-len", str(args.seq_len)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            status = "OOM" if "out of memory" in (proc.stderr.lower() + proc.stdout.lower()) else "FAIL"
            print(f"{bs:>6}  {'—':>12}  {status}")
            if status == "FAIL":
                print(proc.stderr[-500:], file=sys.stderr)
            continue
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            print(f"{result['batch_size']:>6}  {result['peak_gib']:>12.2f}  ok")
        except Exception:
            print(f"{bs:>6}  {'—':>12}  parse-error")
            print(proc.stdout[-500:], file=sys.stderr)


if __name__ == "__main__":
    main()
