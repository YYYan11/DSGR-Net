#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 SOURCE_JSON OUTPUT_CACHE [EXPECTED_RECORDS]" >&2
  exit 2
fi

SOURCE_JSON=$1
OUTPUT_CACHE=$2
EXPECTED_RECORDS=${3:-51480}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

PYTHONPATH="$ROOT/src" python3 "$ROOT/src/dsgr_net.py" cache \
  --source "$SOURCE_JSON" \
  --cache "$OUTPUT_CACHE" \
  --expected-records "$EXPECTED_RECORDS" \
  --start-day 90 \
  --end-day 300
