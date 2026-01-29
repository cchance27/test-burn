#!/usr/bin/env bash
set -euo pipefail

# Default values
MODEL_FP16="./models/qwen2.5-coder-0.5b-instruct-fp16.gguf"
MODEL_Q8="./models/qwen2.5-coder-0.5b-instruct-q8_0.gguf"
PROMPT="create a short js fibonacci function"
MAX_TOKENS=50
ITERATIONS=5
ENABLE_PROFILING=0
ENGINE="context"

RUN_FP16=0
RUN_Q8=0

show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --fp16             Run FP16 benchmark"
    echo "  --q8               Run Q8 benchmark"
    echo "  --fp16_path <path> Path to FP16 model (default: $MODEL_FP16)"
    echo "  --q8_path <path>   Path to Q8 model (default: $MODEL_Q8)"
    echo "  --prompt <string>  Prompt to use (default: '$PROMPT')"
    echo "  --max-tokens <n>   Max tokens to generate (default: $MAX_TOKENS)"
    echo "  --iterations <n>   Number of iterations per model (default: $ITERATIONS)"
    echo "  --profiles         Enable profiling (METALLIC_ENABLE_PROFILING=1)"
    echo "  --engine <name>    Engine to use (context, foundry) (default: $ENGINE)"
    echo "  --help             Show this help"
    echo ""
    echo "Note: If neither --fp16 nor --q8 is specified, both will run."
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fp16) RUN_FP16=1; shift ;;
        --q8) RUN_Q8=1; shift ;;
        --fp16_path) MODEL_FP16="$2"; shift 2 ;;
        --q8_path) MODEL_Q8="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --iterations) ITERATIONS="$2"; shift 2 ;;
        --profiles) ENABLE_PROFILING=1; shift ;;
        --engine) ENGINE="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Default to both if none specified
if [[ $RUN_FP16 -eq 0 && $RUN_Q8 -eq 0 ]]; then
    RUN_FP16=1
    RUN_Q8=1
fi

echo "[metallic] MAX_TOKENS=${MAX_TOKENS}"
echo "[metallic] prompt=${PROMPT}"
echo "[metallic] iterations=${ITERATIONS}"
echo "[metallic] engine=${ENGINE}"
echo "[metallic] profiling=${ENABLE_PROFILING}"
echo

calculate_stats() {
    local label="$1"
    shift
    local values=("$@")
    
    if [ ${#values[@]} -eq 0 ]; then
        return
    fi

    local min=""
    local max=""
    local sum=0
    local count=0
    
    for val in "${values[@]}"; do
        # Defensive: only accept plain numeric values.
        if [[ ! "$val" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            continue
        fi
        if [ -z "$min" ]; then
            min="$val"
            max="$val"
        fi
        if (( $(echo "$val < $min" | bc -l) )); then min=$val; fi
        if (( $(echo "$val > $max" | bc -l) )); then max=$val; fi
        sum=$(echo "$sum + $val" | bc -l)
        count=$((count + 1))
    done
    
    if [ "$count" -eq 0 ]; then
        return
    fi

    local avg=$(echo "$sum / $count" | bc -l)
    printf "  [stats] %-20s | min: %8.2f | avg: %8.2f | max: %8.2f\n" "$label" "$min" "$avg" "$max"
}

run_benchmark() {
    local name="$1"
    local model="$2"
    local jsonl_path="metrics-throughput-${name}.jsonl"
    local bin_path="target/release/metallic_cli"
    local foundry_per_kernel_profiling=0
    if [[ "${ENABLE_PROFILING}" -eq 1 && "${ENGINE}" == "foundry" ]]; then
        foundry_per_kernel_profiling=1
    fi
    
    echo "== ${name} =="

    # Build once up-front so we don't depend on a stale binary after code changes.
    cargo build -q --message-format=short --release
    
    local load_times=()
    local tok_times=()
    local tok_tps=()
    local pp_times=()
    local pp_tps=()
    local setup_times=()
    local decode_times=()
    local decode_tps=()
    local e2e_times=()
    local e2e_tps=()
    local total_times=()
    local total_tps=()
    
    for ((i=1; i<=ITERATIONS; i++)); do
        echo "  Run $i/$ITERATIONS..."
        rm -f "${jsonl_path}"
        
        # Run and capture stderr (where metrics are printed)
        local output
        output=$(METALLIC_ENABLE_PROFILING=${ENABLE_PROFILING} \
          METALLIC_FOUNDRY_PER_KERNEL_PROFILING=${foundry_per_kernel_profiling} \
          METALLIC_PERF_OUTPUT=1 \
          METALLIC_RECORD_CB_GPU_TIMING=1 \
          METALLIC_METRICS_JSONL_PATH="${jsonl_path}" \
          "${bin_path}" "${model}" "${PROMPT}" \
          --seed 42 --output-format=none --max-tokens="${MAX_TOKENS}" --engine="${ENGINE}" 2>&1)
        
        # Extraction logic
        # Anchor on the perf breakdown lines to avoid accidental matches in unrelated logs.
        local l_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Model Load:' | sed -E 's/.*Model Load:[[:space:]]+([0-9.]+)s.*/\1/')
        local t_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Tokenization:' | sed -E 's/.*Tokenization:[[:space:]]+([0-9.]+)s.*/\1/')
        local t_tps=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Tokenization:' | sed -E 's|.*[(]([0-9.]+) tokens/s[)].*|\1|')
        local p_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Prompt Processing:' | sed -E 's/.*Prompt Processing:[[:space:]]+([0-9.]+)s.*/\1/')
        local p_tps=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Prompt Processing:' | sed -E 's|.*[(]([0-9.]+) tok/s[)].*|\1|')
        local d_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Decode:' | sed -E 's/.*Decode:[[:space:]]+([0-9.]+)s.*/\1/')
        local d_tps=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Decode:' | sed -E 's|.*[(]([0-9.]+) tokens/s[)].*|\1|')
        local e2e_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+End-to-End:' | sed -E 's/.*End-to-End:[[:space:]]+([0-9.]+)s.*/\1/')
        local e2e_tps=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+End-to-End:' | sed -E 's|.*[(]([0-9.]+) tokens/s[)].*|\1|')
        local tot_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Total:' | sed -E 's/.*Total:[[:space:]]+([0-9.]+)s.*/\1/')
        local tot_tps=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Total:' | sed -E 's|.*[(]([0-9.]+) tokens/s[)].*|\1|')

        local setup_time=$(echo "$output" | grep -m1 -E '^\[metallic\][[:space:]]+Setup:' | sed -E 's/.*Setup:[[:space:]]+([0-9.]+)s.*/\1/' || true)

        [[ -n "$l_time" ]] && load_times+=("$l_time")
        [[ -n "$t_time" ]] && tok_times+=("$t_time")
        [[ -n "$t_tps" ]] && tok_tps+=("$t_tps")
        [[ -n "$p_time" ]] && pp_times+=("$p_time")
        [[ -n "$p_tps" ]] && pp_tps+=("$p_tps")
        [[ -n "$setup_time" ]] && setup_times+=("$setup_time")
        [[ -n "$d_time" ]] && decode_times+=("$d_time")
        [[ -n "$d_tps" ]] && decode_tps+=("$d_tps")
        [[ -n "$e2e_time" ]] && e2e_times+=("$e2e_time")
        [[ -n "$e2e_tps" ]] && e2e_tps+=("$e2e_tps")
        [[ -n "$tot_time" ]] && total_times+=("$tot_time")
        [[ -n "$tot_tps" ]] && total_tps+=("$tot_tps")

        echo "    Model Load: ${l_time:-N/A}s | Prefill: ${p_tps:-N/A} tok/s | Decode: ${d_tps:-N/A} tok/s | Setup: ${setup_time:-N/A}s | E2E: ${e2e_tps:-N/A} tok/s"
    done
    
    echo ""
    echo "== Results for ${name} (${ENGINE}) =="
    # Use ${array[@]+"${array[@]}"} to avoid unbound variable errors on empty arrays in older bash versions
    calculate_stats "Model Load (s)" ${load_times[@]+"${load_times[@]}"}
    calculate_stats "Tokenization (s)" ${tok_times[@]+"${tok_times[@]}"}
    calculate_stats "Tokenization (tps)" ${tok_tps[@]+"${tok_tps[@]}"}
    calculate_stats "Prefill (s)" ${pp_times[@]+"${pp_times[@]}"}
    calculate_stats "Prefill (tps)" ${pp_tps[@]+"${pp_tps[@]}"}
    calculate_stats "Setup (s)" ${setup_times[@]+"${setup_times[@]}"}
    calculate_stats "Decode (s)" ${decode_times[@]+"${decode_times[@]}"}
    calculate_stats "Decode (tps)" ${decode_tps[@]+"${decode_tps[@]}"}
    calculate_stats "End-to-End (s)" ${e2e_times[@]+"${e2e_times[@]}"}
    calculate_stats "End-to-End (tps)" ${e2e_tps[@]+"${e2e_tps[@]}"}
    calculate_stats "Total (s)" ${total_times[@]+"${total_times[@]}"}
    calculate_stats "Total (tps)" ${total_tps[@]+"${total_tps[@]}"}

    if [[ "${ENABLE_PROFILING}" -eq 1 && "${ENGINE}" == "foundry" ]]; then
        echo
        echo "== Top GPU ops (from ${jsonl_path}) =="
        bash ./tools/summarize_metrics_jsonl.sh "${jsonl_path}" 15 || true
    fi
    echo
}

# Run FP16 if selected
if [[ $RUN_FP16 -eq 1 ]]; then
    if [ ! -f "$MODEL_FP16" ]; then 
        echo "Error: FP16 model not found at $MODEL_FP16"
        exit 1
    fi
    run_benchmark "fp16" "${MODEL_FP16}"
fi

# Run Q8 if selected
if [[ $RUN_Q8 -eq 1 ]]; then
    if [ ! -f "$MODEL_Q8" ]; then 
        echo "Error: Q8 model not found at $MODEL_Q8"
        exit 1
    fi
    run_benchmark "q8" "${MODEL_Q8}"
fi
