#!/bin/bash
export PYTHONPATH=/Users/franciscocosta/repos/pt-caselaw-dgsi
python3 scripts/_analyze_legislation_cardinality.py > /tmp/leg_analysis.txt 2>&1
echo "EXIT CODE: $?" >> /tmp/leg_analysis.txt
