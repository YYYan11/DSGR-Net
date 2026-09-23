#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python3 "$ROOT/scripts/summarize_results.py" \
  --input "$ROOT/results/baselines" \
  --output "$ROOT/results/paper_table_summary.csv"
