#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 CACHE CHECKPOINT_DIR [DEVICE]" >&2
  exit 2
fi

CACHE=$1
CHECKPOINT_DIR=$2
DEVICE=${3:-cpu}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

for SEED in 42 43 44; do
  PYTHONPATH="$ROOT/src" python3 "$ROOT/src/dsgr_net.py" gradient-check \
    --cache "$CACHE" \
    --checkpoint "$CHECKPOINT_DIR/dsgr_v16_seed${SEED}/process_nn.pt" \
    --output "$CHECKPOINT_DIR/dsgr_v16_seed${SEED}/gradient_audit.json" \
    --device "$DEVICE"
done
