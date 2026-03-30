from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.genome import compute_gc_content
from src.utils.io import dump_json, ensure_dir


def repeat_to_length(seed: str, length: int) -> str:
    repeated = (seed * ((length // len(seed)) + 1))[:length]
    return repeated


def build_sequence(motif: str, variant: str, length: int) -> str:
    base = repeat_to_length(motif + variant, length)
    return base[: length // 2] + repeat_to_length(variant + motif, length // 2)


def encode_variant(index: int, length: int = 12) -> str:
    alphabet = "ACGT"
    chars: list[str] = []
    value = index
    for _ in range(length):
        chars.append(alphabet[value % len(alphabet)])
        value //= len(alphabet)
    return "".join(chars)


def make_manifest(path: Path, examples_per_class: int) -> pd.DataFrame:
    class_specs = {
        "promoter_like": ("TATA", [1, 1, 0, 0]),
        "enhancer_like": ("CACGTG", [0, 1, 1, 0]),
        "ctcf_like": ("CCCTC", [0, 0, 1, 1]),
        "open_chromatin_like": ("GGAA", [1, 0, 1, 1]),
    }
    split_chroms = {
        "train": ["chr3", "chr4", "chr5", "chr6"],
        "val": ["chr2", "chr7"],
        "test": ["chr1", "chr8"],
    }

    rows: list[dict[str, object]] = []
    row_idx = 0
    for split, chroms in split_chroms.items():
        for class_idx, (class_name, (motif, activity)) in enumerate(class_specs.items()):
            for example_idx in range(examples_per_class):
                chrom = chroms[example_idx % len(chroms)]
                start = 1000 + (row_idx * 200)
                end = start + 150
                midpoint = (start + end) // 2
                variant = encode_variant((class_idx * examples_per_class) + example_idx, length=12)
                sequence_512 = build_sequence(motif, variant, 512)
                sequence_1024 = build_sequence(motif, variant + motif[:2], 1024)
                sequence_2048 = build_sequence(motif, variant + motif[:2], 2048)
                rows.append(
                    {
                        "ccre_id": f"{split}_{class_name}_{example_idx}_{row_idx}",
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "assembly": "hg38",
                        "ccre_class": class_name,
                        "midpoint": midpoint,
                        "sequence_512": sequence_512,
                        "sequence_1024": sequence_1024,
                        "sequence_2048": sequence_2048,
                        "gc_content": compute_gc_content(sequence_1024),
                        "activity_vector": json.dumps(activity),
                        "activity_count": sum(activity),
                        "biosample_group": f"group_{class_name}",
                        "length": end - start,
                        "split": split,
                    }
                )
                row_idx += 1

    manifest = pd.DataFrame(rows)
    ensure_dir(path.parent)
    manifest.to_csv(path, index=False)
    return manifest


def run_step(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a larger synthetic Stage A pipeline")
    parser.add_argument("--tag", default="synthetic_medium")
    parser.add_argument("--examples-per-class", type=int, default=64)
    parser.add_argument("--negatives-per-query", type=int, default=15)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pair-path", default=None)
    parser.add_argument("--train-config", default="configs/train/stage_a_synthetic_medium.yaml")
    parser.add_argument("--eval-config", default="configs/train/eval_synthetic_medium.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    tag = args.tag
    manifest_path = Path(args.manifest) if args.manifest else ROOT / f"data/interim/manifests/{tag}_manifest.csv"
    pair_path = Path(args.pair_path) if args.pair_path else ROOT / f"data/processed/stage_a/{tag}_pairs.csv"
    checkpoint_path = ROOT / f"outputs/{tag}/checkpoints/{tag}_stage_a_best.pt"
    metrics_dir = ROOT / f"outputs/{tag}/metrics"
    model_metrics_path = metrics_dir / f"eval_model_{tag}.json"
    kmer_metrics_path = metrics_dir / f"eval_kmer_{tag}.json"
    random_metrics_path = metrics_dir / f"eval_random_{tag}.json"
    summary_path = metrics_dir / f"{tag}_summary.json"

    manifest = make_manifest(manifest_path, examples_per_class=args.examples_per_class)
    run_step(
        [
            sys.executable,
            "-m",
            "src.data.build_stage_a_pairs",
            "--manifest",
            str(manifest_path),
            "--output",
            str(pair_path),
            "--negatives-per-query",
            str(args.negatives_per_query),
        ]
    )

    if not args.skip_train:
        run_step([sys.executable, "-m", "src.train.train_stage_a", "--config", args.train_config])

    if not args.skip_eval:
        run_step(
            [
                sys.executable,
                "-m",
                "src.eval.eval_screen",
                "--config",
                args.eval_config,
                "--baseline",
                "model",
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(model_metrics_path),
            ]
        )
        run_step(
            [
                sys.executable,
                "-m",
                "src.eval.eval_screen",
                "--config",
                args.eval_config,
                "--baseline",
                "kmer",
                "--output",
                str(kmer_metrics_path),
            ]
        )
        run_step(
            [
                sys.executable,
                "-m",
                "src.eval.eval_screen",
                "--config",
                args.eval_config,
                "--baseline",
                "random",
                "--output",
                str(random_metrics_path),
            ]
        )

    summary = {
        "tag": tag,
        "manifest_path": str(manifest_path),
        "pair_path": str(pair_path),
        "checkpoint_path": str(checkpoint_path),
        "examples_per_class": int(args.examples_per_class),
        "num_manifest_rows": int(len(manifest)),
    }
    if model_metrics_path.exists():
        summary["model_metrics"] = json.loads(model_metrics_path.read_text())
    if kmer_metrics_path.exists():
        summary["kmer_metrics"] = json.loads(kmer_metrics_path.read_text())
    if random_metrics_path.exists():
        summary["random_metrics"] = json.loads(random_metrics_path.read_text())
    dump_json(summary, summary_path)


if __name__ == "__main__":
    main()
