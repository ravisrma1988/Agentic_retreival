#!/usr/bin/env bash
# Final dialogue evaluation on the held-out persona set (seed 7). QA, retrieval and intent results are unaffected
# by the last routing change, so only dialogues are re-run here.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export MPLBACKEND=Agg
mkdir -p runs_dev2
[ -d runs/dialogues ] && mv runs/dialogues runs_dev2/ || true
[ -d runs/dialogues_latency ] && mv runs/dialogues_latency runs_dev2/ || true
[ -f data/personas_dev2.jsonl ] || mv data/personas.jsonl data/personas_dev2.jsonl
$PY -m lq.datagen personas
$PY -m lq.evaluate dialogues agentic_rag agentic_rag_no_sticky agentic_rag_no_interrupt agentic_rag_no_confirm react --workers 8
$PY -m lq.evaluate dialogues agentic_rag react --workers 1 --tag _seq
$PY -m lq.evaluate report > /dev/null
echo FINAL DONE
