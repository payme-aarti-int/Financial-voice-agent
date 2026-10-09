#!/bin/bash
# Start llama-server with optimized settings.
# All llama.cpp logs go to /tmp/llama-server.log (not the terminal).

MODEL="${LLAMACPP_MODEL_PATH:-$HOME/models/qwen2.5-7b/Qwen2.5-7B-Instruct-Q4_K_M.gguf}"
PORT="${LLAMACPP_PORT:-8080}"
LOG=/tmp/llama-server.log

if [ ! -f "$MODEL" ]; then
  echo "[LLM] Model not found: $MODEL" >&2
  echo "[LLM] Set LLAMACPP_MODEL_PATH or download it (see README)" >&2
  exit 1
fi

echo "[LLM] Starting llama-server..."
echo "[LLM] Model: $MODEL"
echo "[LLM] Logs:  $LOG"

pkill -f llama-server 2>/dev/null
sleep 2

llama-server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  > "$LOG" 2>&1 &

# Model load time depends on disk cache; poll instead of a fixed sleep.
for _ in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "[LLM] Ready on http://localhost:$PORT"
    exit 0
  fi
  sleep 2
done

echo "[LLM] Failed - check $LOG" >&2
tail -5 "$LOG" >&2
exit 1
