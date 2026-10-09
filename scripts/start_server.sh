#!/bin/bash
# Start the Financial Voice Agent server with clean logs.
# Run scripts/start_llm.sh first if LLM_PROVIDER=llamacpp.

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate

PORT="${PORT:-5000}"

echo "[SERVER] Starting Financial Voice Agent on http://localhost:$PORT ..."
exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT"
