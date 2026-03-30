from __future__ import annotations

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


def make_manifest(path: Path) -> None:
    class_specs = {
        "promoter_like": ("TATA", [1, 1, 0, 0]),
        "enhancer_like": ("CACGTG", [0, 1, 1, 0]),
        "ctcf_like": ("CCCTC", [0, 0, 1, 1]),
    }
    split_chroms = {
        "train": ["chr3", "chr4"],
        "val": ["chr2"],
        "test": ["chr1"],
    }

    rows: list[dict[str, object]] = []
    row_idx = 0
    for split, chroms in split_chroms.items():
        for chrom in chroms:
            for class_name, (motif, activity) in class_specs.items():
                for variant_idx in range(4):
                    start = 1000 + (row_idx * 200)
                    end = start + 150
                    midpoint = (start + end) // 2
                    variant = "ACGT"[(variant_idx % 4)] * 8
                    sequence_512 = build_sequence(motif, variant, 512)
                    sequence_1024 = build_sequence(motif, variant + motif[:2], 1024)
                    sequence_2048 = build_sequence(motif, variant + motif[:2], 2048)
                    rows.append(
                        {
                            "ccre_id": f"{split}_{class_name}_{variant_idx}_{row_idx}",
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


def run_step(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    manifest_path = ROOT / "data/interim/manifests/smoke_manifest.csv"
    pair_path = ROOT / "data/processed/stage_a/smoke_pairs.csv"
    checkpoint_path = ROOT / "outputs/smoke/checkpoints/smoke_stage_a_best.pt"
    model_metrics_path = ROOT / "outputs/smoke/metrics/eval_model_smoke.json"
    kmer_metrics_path = ROOT / "outputs/smoke/metrics/eval_kmer_smoke.json"
    random_metrics_path = ROOT / "outputs/smoke/metrics/eval_random_smoke.json"

    make_manifest(manifest_path)
    run_step([sys.executable, "-m", "src.data.split_by_chromosome", "--manifest", str(manifest_path)])
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
            "6",
        ]
    )
    run_step([sys.executable, "-m", "src.train.train_stage_a", "--config", "configs/train/stage_a_smoke.yaml"])
    run_step(
        [
            sys.executable,
            "-m",
            "src.eval.eval_screen",
            "--config",
            "configs/train/eval_smoke.yaml",
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
            "configs/train/eval_smoke.yaml",
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
            "configs/train/eval_smoke.yaml",
            "--baseline",
            "random",
            "--output",
            str(random_metrics_path),
        ]
    )

    summary = {
        "manifest_path": str(manifest_path),
        "pair_path": str(pair_path),
        "checkpoint_path": str(checkpoint_path),
        "model_metrics": json.loads(model_metrics_path.read_text()),
        "kmer_metrics": json.loads(kmer_metrics_path.read_text()),
        "random_metrics": json.loads(random_metrics_path.read_text()),
    }
    dump_json(summary, ROOT / "outputs/smoke/metrics/smoke_test_summary.json")


if __name__ == "__main__":
    main()
