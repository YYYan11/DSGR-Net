#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 CACHE [DEVICE]" >&2
  exit 2
fi

CACHE=$1
DEVICE=${2:-cuda}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

run_variant() {
  local NAME=$1 ENTRY=$2 EPOCHS=$3 PATIENCE=$4 LR=$5 TRAIN_MODEL=$6
  for SEED in 42 43 44; do
    PYTHONPATH="$ROOT/src" python3 "$ROOT/$ENTRY" train \
      --cache "$CACHE" --output-dir "$ROOT/outputs/${NAME}_seed${SEED}" \
      --device "$DEVICE" --backbone dnn --seed "$SEED" --hidden 32 \
      --batch-size 1024 --epochs "$EPOCHS" --patience "$PATIENCE" \
      --learning-rate "$LR" --correction-weight 0.001 \
      --et-correction-limit 0.2 --rue-correction-limit 0.2 \
      --use-nutrients --fixed-yield-coefficient 0.42305775 \
      --maturity-gdd 2600 --train-model "$TRAIN_MODEL" --skip-gradient-scan
  done
}

run_variant strong_environment src/ablation/strong_environment.py 15 4 0.001 process_nn
run_variant direct_irrigation src/ablation/direct_irrigation.py 15 4 0.001 process_nn
run_variant process_gate_v13 src/ablation/process_gate_v13.py 30 8 0.0005 all
run_variant dsgr_v16 src/dsgr_net.py 30 8 0.0005 all
