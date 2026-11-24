#!/usr/bin/env python3
"""
Aggregate nested_flint.py runs (directories named <arm>_s<seed>) into tables.

For each arm, at its full budget (prefix<V> for FLINT, "full" for token-level):
  * test:  test queries searched against the test set (ProtoCol protocol)
  * db:    test queries searched against test + train (more fold-tier queries)
Reports mean ± std over seeds and paired differences (same seeds) with 95% t-intervals.

Usage:
    python aggregate_nested.py results/nested/final [--base base_V32] [--pairs A:B ...]
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

METRICS = ["R@1", "R@10", "sens_fam", "sens_sf", "sens_fold"]
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def load(root):
    runs = defaultdict(dict)  # arm -> seed -> {"test": row, "db": row, "div": {...}, "rows": {...}}
    for f in glob.glob(os.path.join(root, "*", "results.json")):
        arm, seed = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)
        r = json.load(open(f))
        rows = {row["eval"]: row for row in r["rows"]}
        V = r["config"]["latents"]
        test = rows.get("full") or rows.get(f"prefix{V}")
        runs[arm][int(seed)] = {"test": test, "db": rows.get("db_full"), "rows": rows,
                                "div": r.get("diversity", {})}
    return runs


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


def paired(runs, a, b, split, metric):
    seeds = sorted(s for s in set(runs[a]) & set(runs[b])
                   if runs[a][s][split] is not None and runs[b][s][split] is not None)
    d = [runs[a][s][split][metric] - runs[b][s][split][metric] for s in seeds]
    if len(d) < 2:
        return None
    m, sd = mean_sd(d)
    hw = T95[len(d) - 1] * sd / math.sqrt(len(d))
    return m, hw, len(d)


def table(runs, arms, split, base):
    print(f"\n=== {split}: mean ± std over seeds; (paired diff vs {base}, * = 95% CI excludes 0) ===")
    print(f"{'arm':<20}{'n':>3}" + "".join(f"{m:>25}" for m in METRICS))
    for a in arms:
        vals = [s for s in runs[a].values() if s[split] is not None]
        if not vals:
            continue
        cells = []
        for metric in METRICS:
            m, sd = mean_sd([s[split][metric] for s in vals])
            cell = f"{m:.4f}±{sd:.4f}"
            if a != base and base in runs:
                p = paired(runs, a, base, split, metric)
                if p:
                    cell += f" ({p[0]:+.3f}{'*' if abs(p[0]) > p[1] else ' '})"
            cells.append(cell)
        print(f"{a:<20}{len(vals):>3}" + "".join(f"{c:>25}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--base", default="base_V32")
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B paired comparisons to print in full")
    args = ap.parse_args()
    runs = load(args.root)
    arms = sorted(runs, key=lambda a: (not a.startswith("base"), a != "token", a))

    print("Vector diversity (doc side): mean cosine / effective rank")
    for a in arms:
        d = [s["div"] for s in runs[a].values() if "mean_cos" in s["div"]]
        if d:
            c, e = mean_sd([x["mean_cos"] for x in d]), mean_sd([x["eff_rank"] for x in d])
            print(f"  {a:<20} cos {c[0]:.3f}±{c[1]:.3f}  eff_rank {e[0]:.2f}±{e[1]:.2f}")

    any_seed = next(iter(next(iter(runs.values())).values()))
    n_q = {k: any_seed[sp][f"n_{k}"] for sp in ("test", "db") if any_seed[sp] for k in ("fam", "sf", "fold")}
    print(f"Queries per tier (last split listed wins): {n_q}")

    for split in ("test", "db"):
        table(runs, arms, split, args.base)

    for pair in args.pairs:
        a, b = pair.split(":")
        print(f"\n=== {a} - {b} (paired) ===")
        for split in ("test", "db"):
            cells = []
            for metric in METRICS:
                p = paired(runs, a, b, split, metric)
                if p:
                    cells.append(f"{metric} {p[0]:+.4f}±{p[1]:.4f}{'*' if abs(p[0]) > p[1] else ' '}")
            print(f"  {split:<5}" + "  ".join(cells))

    # Fraction of the base -> token gap closed, per arm (means over seeds).
    if "token" in runs and args.base in runs:
        print(f"\n=== Share of the {args.base} -> token gap closed ===")
        for split in ("test", "db"):
            print(f"  [{split}]")
            for a in arms:
                if a in ("token", args.base) or a.startswith("base"):
                    continue
                cells = []
                for metric in METRICS:
                    mean = lambda arm: mean_sd([s[split][metric] for s in runs[arm].values() if s[split]])[0]
                    gap = mean("token") - mean(args.base)
                    cells.append(f"{metric} {100 * (mean(a) - mean(args.base)) / gap:5.0f}%" if gap else "-")
                print(f"    {a:<20}" + "  ".join(cells))


if __name__ == "__main__":
    main()
