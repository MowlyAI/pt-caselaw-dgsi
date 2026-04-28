#!/bin/bash
cd /Users/franciscocosta/repos/pt-caselaw-dgsi
PATTERN='is_jurisprudence_unification": true'
for db in STJ STA TCON TC TRP TRL TRC TRG TRE TCAS TCAN JP; do
  total=0
  for f in data/enhanced/$db/chunk_*.jsonl; do
    if [ -f "$f" ]; then
      c=$(grep -c "$PATTERN" "$f" 2>/dev/null || echo 0)
      total=$((total + c))
    fi
  done
  echo "$db: $total AUJs"
done
