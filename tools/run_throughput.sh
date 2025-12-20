#!/usr/bin/env bash
set -euo pipefail

MODEL_FP16="${1:-./models/qwen2.5-coder-0.5b-instruct-fp16.gguf}"
MODEL_Q8="${2:-./models/qwen2.5-coder-0.5b-instruct-q8_0.gguf}"
PROMPT="${3:-create a short js fibonacci function}"
MAX_TOKENS="${MAX_TOKENS:-256}"

echo "[metallic] MAX_TOKENS=${MAX_TOKENS}"
echo "[metallic] prompt=${PROMPT}"
echo

run_one() {
  local name="$1"
  local model="$2"
  local jsonl_path="/tmp/metallic-throughput-${name}.jsonl"

  echo "== ${name} =="
  rm -f "${jsonl_path}"
  METALLIC_ENABLE_PROFILING=0 \
    METALLIC_RECORD_CB_GPU_TIMING=1 \
    METALLIC_METRICS_JSONL_PATH="${jsonl_path}" \
    cargo run -q --message-format=short --release -- "${model}" "${PROMPT}" --output-format=text --max-tokens="${MAX_TOKENS}"
  echo
}

run_one "fp16" "${MODEL_FP16}"
run_one "q8" "${MODEL_Q8}" ### Disable when we're working on FP16 to speed up iterating
