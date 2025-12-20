#!/usr/bin/env bash
set -euo pipefail

MODEL_FP16="${1:-./models/qwen2.5-coder-0.5b-instruct-fp16.gguf}"
MODEL_Q8="${2:-./models/qwen2.5-coder-0.5b-instruct-q8_0.gguf}"
PROMPT="${3:-create a short js fibonacci function}"
MAX_TOKENS="${MAX_TOKENS:-50}"
ITERATIONS="${ITERATIONS:-5}"

echo "[metallic] MAX_TOKENS=${MAX_TOKENS}"
echo "[metallic] prompt=${PROMPT}"
echo "[metallic] iterations=${ITERATIONS}"
echo

calculate_stats() {
    local label="$1"
    shift
    local values=("$@")
    
    if [ ${#values[@]} -eq 0 ]; then
        return
    fi

    local min=${values[0]}
    local max=${values[0]}
    local sum=0
    
    for val in "${values[@]}"; do
        if (( $(echo "$val < $min" | bc -l) )); then min=$val; fi
        if (( $(echo "$val > $max" | bc -l) )); then max=$val; fi
        sum=$(echo "$sum + $val" | bc -l)
    done
    
    local avg=$(echo "$sum / ${#values[@]}" | bc -l)
    printf "[stats] %-12s | min: %6.2f | avg: %6.2f | max: %6.2f\n" "$label" "$min" "$avg" "$max"
}

run_benchmark() {
    local name="$1"
    local model="$2"
    local jsonl_path="metrics-throughput-${name}.jsonl"
    
    echo "== ${name} =="
    
    local tps_totals=()
    local tps_decodes=()
    
    for ((i=1; i<=ITERATIONS; i++)); do
        echo "  Run $i/$ITERATIONS..."
        rm -f "${jsonl_path}"
        
        # Run and capture stderr (where metrics are printed)
        local output
        output=$(METALLIC_ENABLE_PROFILING=0 \
          METALLIC_RECORD_CB_GPU_TIMING=1 \
          METALLIC_METRICS_JSONL_PATH="${jsonl_path}" \
          cargo run -q --message-format=short --release -- "${model}" "${PROMPT}" --seed 42 --output-format=none --max-tokens="${MAX_TOKENS}" 2>&1)
        
        local tps_total=$(echo "$output" | grep "tps_total=" | sed -E 's/.*tps_total=([0-9.]+).*/\1/')
        local tps_decode=$(echo "$output" | grep "tps_decode=" | sed -E 's/.*tps_decode=([0-9.]+).*/\1/')
        
        if [ -n "$tps_total" ]; then
            tps_totals+=("$tps_total")
            echo "    TPS Total: $tps_total"
        fi
        if [ -n "$tps_decode" ]; then
            tps_decodes+=("$tps_decode")
            echo "    TPS Decode: $tps_decode"
        fi
    done
    
    echo "  Results for ${name}:"
    calculate_stats "TPS Total" "${tps_totals[@]}"
    calculate_stats "TPS Decode" "${tps_decodes[@]}"
    echo
}

run_benchmark "fp16" "${MODEL_FP16}"
run_benchmark "q8" "${MODEL_Q8}"
