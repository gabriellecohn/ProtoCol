#!/usr/bin/env bash

set -euo pipefail

python -m src.eval.eval_screen --config configs/train/eval.yaml "$@"
