#!/bin/bash
# The body below the "SHARED PIPELINE" marker must be byte-identical across
# every sweep_*.py. Run from anywhere; exits non-zero on drift.
set -u
cd "$(dirname "$0")"

extract() { sed -n '/^# ==== SHARED PIPELINE/,$p' "$1"; }

ok=1
for f in sweep_*.py; do
  [ "$f" = "sweep_ULB.py" ] && continue
  if diff -q <(extract sweep_ULB.py) <(extract "$f") > /dev/null; then
    echo "OK    sweep_ULB.py == $f  (shared body)"
  else
    echo "DRIFT sweep_ULB.py != $f  — regenerate with generate_from_notebook.py"
    ok=0
  fi
done
exit $((1 - ok))
