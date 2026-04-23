#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'USAGE'
Usage: ./run.sh [command] [args...]

Commands:
  smoke         Run the lightweight MVP smoke test pipeline
  synthetic     Run a larger synthetic Stage A pipeline
  stage-a       Train the Stage A retrieval model with the default config
  pfam-colbert  Train Pfam ColBERT (single process; uses gpu_ids from config for DataParallel)
  eval          Evaluate the retrieval pipeline with the default config
  train         Run the root diffusion training script
  infer         Run the root diffusion inference script
  help          Show this message

Env:
  PFAM_CONFIG   Config for pfam-colbert (default: configs/train/stage_a_pfam.yaml)
  NPROC         If >1, launch pfam-colbert via torchrun (future DDP; default: 1)

Examples:
  ./run.sh
  ./run.sh smoke
  ./run.sh synthetic
  ./run.sh synthetic --examples-per-class 96
  ./run.sh stage-a --config configs/train/stage_a.yaml
  ./run.sh pfam-colbert
  NPROC=2 PFAM_CONFIG=configs/train/stage_a_pfam_meanpool.yaml ./run.sh pfam-colbert
  ./run.sh eval --config configs/train/eval.yaml --baseline kmer
  ./run.sh train --debug
  ./run.sh infer --gene MYC --mode guided --target-expr 2.5
USAGE
}

COMMAND="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$COMMAND" in
  smoke)
    exec "$PYTHON_BIN" scripts/smoke_test.py "$@"
    ;;
  synthetic)
    exec "$PYTHON_BIN" scripts/run_synthetic_stage_a.py "$@"
    ;;
  stage-a)
    exec "$PYTHON_BIN" -m src.train.train_stage_a --config configs/train/stage_a.yaml "$@"
    ;;
  pfam-colbert)
    PFAM_CONFIG="${PFAM_CONFIG:-configs/train/stage_a_pfam.yaml}"
    # Multi-GPU uses DataParallel via gpu_ids in the config (single process).
    # Set NPROC>1 to launch under torchrun for future DDP use — currently
    # only the rank-0 broadcasts/logging are DDP-aware, the model itself is not
    # DDP-wrapped, so NPROC=1 is the correct default.
    NPROC="${NPROC:-1}"
    if [[ "$NPROC" -gt 1 ]]; then
      exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC" --standalone \
        -m src.train.train_stage_a --config "$PFAM_CONFIG" "$@"
    fi
    exec "$PYTHON_BIN" -m src.train.train_stage_a --config "$PFAM_CONFIG" "$@"
    ;;
  eval)
    exec "$PYTHON_BIN" -m src.eval.eval_screen --config configs/train/eval.yaml "$@"
    ;;
  train)
    exec "$PYTHON_BIN" train.py "$@"
    ;;
  infer)
    exec "$PYTHON_BIN" infer.py "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
