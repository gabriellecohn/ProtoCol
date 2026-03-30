#!/bin/bash
#SBATCH -p mit_preemptable
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=32GB
#SBATCH -t 4:00:00
#SBATCH --job-name=compbio_smoke
#SBATCH --output=/home/rgumaste/bio_proj/compbio/slurm_outputs/slurm-%j.out
#SBATCH --signal=B:USR1@120
#SBATCH --requeue

set -euo pipefail
module load miniforge
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH

PROJECT_DIR="${PROJECT_DIR:-/home/rgumaste/bio_proj/compbio}"
RUN_TARGET="${RUN_TARGET:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PROJECT_DIR"

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

mkdir -p "$PROJECT_DIR/slurm_outputs"

if [[ $# -gt 0 ]]; then
  RUN_ARGS=("$@")
else
  RUN_ARGS=()
fi

echo "=== compbio ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Project: $PROJECT_DIR"
echo "Run target: $RUN_TARGET"
echo "Python: $PYTHON_BIN"
echo "SLURM_RESTART_COUNT: ${SLURM_RESTART_COUNT:-0}"
echo ""

cd "$PROJECT_DIR"
"$PROJECT_DIR/run.sh" "$RUN_TARGET" "${RUN_ARGS[@]}"

echo ""
echo "=== Done ==="
echo "Date: $(date)"
