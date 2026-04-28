#!/usr/bin/env bash
# Launch the 500-doc validation against STJ raw docs and log to data/logs/prod500.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/logs
export N=500
export CONCURRENCY=40
export OUT_SUBDIR=prod500
export DB=STJ
export PYTHONPATH=.
exec .venv/bin/python scripts/quick_extract_quality_check.py
