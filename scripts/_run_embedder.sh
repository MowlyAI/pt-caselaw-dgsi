#!/bin/bash
set -e
cd /Users/franciscocosta/repos/pt-caselaw-dgsi
echo "Starting embedder at $(date)" >> data/logs/embed_full.log
exec .venv/bin/python -u -m embedder.runner --concurrency 100 >> data/logs/embed_full.log 2>&1
