"""Run the full Pfam baseline suite and any trained checkpoints, then collate.

Default suite:
  random
  kmer-3, kmer-4, kmer-5
  meanpool-esm2-35m, meanpool-esm2-650m
  randproj-esm2-35m, randproj-esm2-650m

Each is run in a fresh subprocess (so a single CUDA OOM doesn't kill the rest).
After all runs finish, a summary CSV is written next to the per-baseline JSONs.

Usage:
    PYTHONPATH=$(pwd) python scripts/run_pfam_baselines.py \\
        --config configs/train/eval_pfam.yaml \\
        --device cuda:6 \\
        --checkpoints \\
            outputs/pfam_colbert_small/<run>/checkpoints/pfam_colbert_small_best.pt \\
            outputs/pfam_meanpool_small/<run>/checkpoints/pfam_meanpool_small_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

DEFAULT_BASELINES = [
    "random",
    "kmer-3", "kmer-4", "kmer-5",
    "meanpool-esm2-35m", "meanpool-esm2-650m",
    "randproj-esm2-35m", "randproj-esm2-650m",
]


def run_one(args, baseline: str | None = None, checkpoint: str | None = None) -> Path | None:
    if baseline is not None:
        out = Path("outputs/pfam_baselines") / f"eval_pfam_{baseline}.json"
        cmd = [sys.executable, "-m", "src.eval.eval_pfam",
               "--config", args.config, "--baseline", baseline,
               "--output", str(out), "--device", args.device]
    else:
        ckpt_path = Path(checkpoint)
        out = ckpt_path.parent.parent / "metrics" / "eval_pfam.json"
        cmd = [sys.executable, "-m", "src.eval.eval_pfam",
               "--config", args.config, "--checkpoint", str(ckpt_path),
               "--output", str(out), "--device", args.device]

    if out.exists() and not args.force:
        print(f"[SKIP] {out} already exists (use --force to rerun)")
        return out

    print(f"[RUN ] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd())
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        print(f"[FAIL] {baseline or checkpoint}  (exit {proc.returncode})")
        return None
    return out


def collate(paths: list[Path], out_csv: Path) -> None:
    rows = []
    for p in paths:
        if not p or not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append({
            "scorer": d.get("scorer", p.stem),
            "fam_mrr": d.get("fam_mrr"),
            "fam_recall@1": d.get("fam_recall@1"),
            "fam_recall@5": d.get("fam_recall@5"),
            "fam_recall@10": d.get("fam_recall@10"),
            "fam_recall@50": d.get("fam_recall@50"),
            "fam_ndcg@10": d.get("fam_ndcg@10"),
            "clan_mrr": d.get("clan_mrr"),
            "clan_recall@10": d.get("clan_recall@10"),
            "hard_mrr": d.get("hard_mrr"),
            "top10_pct_same_family": d.get("top10_pct_same_family"),
            "top10_pct_hard_neg": d.get("top10_pct_hard_neg"),
            "n_queries": d.get("n_queries_evaluated"),
        })
    if not rows:
        print("No results to collate")
        return
    df = pd.DataFrame(rows).sort_values("fam_mrr", ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote summary to {out_csv}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/eval_pfam.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baselines", nargs="*", default=DEFAULT_BASELINES,
                        help="Baselines to run (default: all)")
    parser.add_argument("--checkpoints", nargs="*", default=[],
                        help="Trained checkpoints to also evaluate")
    parser.add_argument("--summary", default="outputs/pfam_baselines/summary.csv")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output JSON already exists")
    args = parser.parse_args()

    out_paths = []
    for b in args.baselines:
        out_paths.append(run_one(args, baseline=b))
    for ckpt in args.checkpoints:
        out_paths.append(run_one(args, checkpoint=ckpt))

    collate([p for p in out_paths if p], Path(args.summary))


if __name__ == "__main__":
    main()
