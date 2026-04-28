#!/bin/bash
cd /Users/franciscocosta/repos/pt-caselaw-dgsi
echo "=== PART 1 ==="
.venv/bin/python scripts/_quality_check.py
echo "=== PART 2 ==="
.venv/bin/python scripts/_quality_check_p2.py
echo "=== PART 3 ==="
.venv/bin/python scripts/_quality_check_p3.py
echo "=== DONE ==="
