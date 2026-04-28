#!/usr/bin/env bash
# Detached launcher for the 500-doc validation. Uses setsid so the python
# process survives when the invoking terminal is torn down.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs

# Kill any leftover runs first.
pkill -f "scripts/quick_extract_quality_check.py" 2>/dev/null || true

# Launch with nohup + disown + own session so the python process survives
# the VSCode terminal shutdown that happens between tool calls.
N=500 CONCURRENCY=40 OUT_SUBDIR=prod500 DB=STJ PYTHONPATH=. \
  nohup .venv/bin/python scripts/quick_extract_quality_check.py \
  > data/logs/prod500.log 2>&1 < /dev/null &
PID=$!
disown
echo "launched pid=$PID"
