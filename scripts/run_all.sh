#!/usr/bin/env bash
# Full experiment pipeline. Human annotation (step marked HUMAN) happens between stage 1 and stage 2.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
stage=${1:-1}

if [ "$stage" = 1 ]; then
  bash scripts/serve.sh
  [ -f data/personas.jsonl ] || $PY -m lq.datagen personas
  [ -f data/intent_pool.csv ] || $PY -m lq.datagen intent   # -> data/annotation/intent_sheet_{A,B}.csv (HUMAN)
  $PY -m lq.evaluate retrieval
  $PY -m lq.evaluate qa llm_only full_context naive_rag react agentic_rag   # sequential: clean latency
  $PY -m lq.evaluate judge llm_only full_context naive_rag react agentic_rag
  $PY -m lq.evaluate annotation-export         # -> data/annotation/qa_sheet_{A,B}.csv       (HUMAN)
  $PY -m lq.evaluate dialogues agentic_rag agentic_rag_no_sticky agentic_rag_no_interrupt agentic_rag_no_confirm react --workers 8
  # sequential runs: clean per-turn latency and a check that results do not depend on request batching
  $PY -m lq.evaluate dialogues agentic_rag react --workers 1 --tag _seq
  $PY -m lq.evaluate intent --allow-unverified  # provisional, replaced after annotation
  $PY -m lq.evaluate report
fi

if [ "$stage" = 2 ]; then  # after both annotators have filled their sheets
  $PY -m lq.evaluate finalize-intent
  $PY -m lq.evaluate intent
  $PY -m lq.evaluate intent --no-context
  $PY -m lq.evaluate agreement
  $PY -m lq.evaluate report
fi
