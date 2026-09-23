#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CACHE [DEVICE]" >&2
  exit 2
fi

CACHE=$1
DEVICE=${2:-cuda}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for SEED in 42 43 44; do
  PYTHONPATH="$ROOT/src" python3 "$ROOT/src/dsgr_net.py" train \
    --cache "$CACHE" \
    --output-dir "$ROOT/outputs/dsgr_v16_seed${SEED}" \
    --device "$DEVICE" \
    --backbone dnn \
    --seed "$SEED" \
    --hidden 32 \
    --batch-size 1024 \
    --epochs 30 \
    --patience 8 \
    --learning-rate 0.0005 \
    --correction-weight 0.001 \
    --et-correction-limit 0.2 \
    --rue-correction-limit 0.2 \
    --use-nutrients \
    --fixed-yield-coefficient 0.42305775 \
    --maturity-gdd 2600 \
    --train-model all \
    --skip-gradient-scan
done
