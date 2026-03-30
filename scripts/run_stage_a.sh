#!/usr/bin/env bash

set -euo pipefail

python -m src.train.train_stage_a --config configs/train/stage_a.yaml "$@"
