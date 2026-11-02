#!/usr/bin/env bash
# Start both model servers on one 45 GB MIG slice (B200 1g.45gb).
#   aux:   Phi-4 (14B), FP8 weight quantisation, ~0.48 of the slice (simulator, data generation, judge)
#   agent: Qwen2.5-7B-Instruct, bf16, ~0.42 of the slice, with tool calling for the ReAct baseline
# Phi-4 starts first: online FP8 quantisation briefly holds the full bf16 weights (~29 GB) while loading.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
VLLM=.venv/bin/vllm

wait_up() {  # port pid log
  until curl -sf "localhost:$1/v1/models" >/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "server on :$1 died, see $3"; tail -20 "$3"; exit 1; }
    sleep 10
  done
  echo "server on :$1 up"
}

if ! curl -sf localhost:8001/v1/models >/dev/null; then
  setsid nohup $VLLM serve microsoft/phi-4 --port 8001 --seed 42 --quantization fp8 \
    --gpu-memory-utilization 0.48 --max-model-len 8192 > logs/vllm_aux.log 2>&1 < /dev/null &
  wait_up 8001 $! logs/vllm_aux.log
fi

if ! curl -sf localhost:8000/v1/models >/dev/null; then
  setsid nohup $VLLM serve Qwen/Qwen2.5-7B-Instruct --port 8000 --seed 42 \
    --gpu-memory-utilization 0.42 --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes > logs/vllm_agent.log 2>&1 < /dev/null &
  wait_up 8000 $! logs/vllm_agent.log
fi
