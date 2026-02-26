#!/usr/bin/env bash
set -euo pipefail

# METALLIC_DEBUG_DECODE_STAGE_TIMING=1 \
# RUST_LOG=metallic_foundry::model::executor::forward=info \
# cargo run --release -- models/qwen2.5-0.5b-instruct-q6_k.gguf \
# "create a short js fibonacci function" \
# --temperature 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --repeat-last-n 64 --max-tokens 64 --output-format text 2>&1 | tee /tmp/fa_decode_timing.log
#
# tools/summarize_fa_decode_timing.sh /tmp/fa_decode_timing.log 25

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <log_file> [top_n]"
  echo "Example: $0 /tmp/fa_decode_timing.log 20"
  exit 1
fi

LOG_FILE="$1"
TOP_N="${2:-20}"

if [[ ! -f "$LOG_FILE" ]]; then
  echo "error: log file not found: $LOG_FILE" >&2
  exit 1
fi

if ! [[ "$TOP_N" =~ ^[0-9]+$ ]] || [[ "$TOP_N" -lt 1 ]]; then
  echo "error: top_n must be a positive integer (got '$TOP_N')" >&2
  exit 1
fi

if ! command -v rg >/dev/null 2>&1; then
  echo "error: rg is required but not found in PATH" >&2
  exit 1
fi

FORWARD_TMP="$(mktemp -t fa_decode_forward.XXXXXX)"
SANITIZED_TMP="$(mktemp -t fa_decode_sanitized.XXXXXX)"
trap 'rm -f "$FORWARD_TMP" "$SANITIZED_TMP"' EXIT

# Strip ANSI color/control escapes that can appear in captured logs.
perl -pe 's/\e\[[0-9;]*[[:alpha:]]//g' "$LOG_FILE" >"$SANITIZED_TMP"

# Only keep decode timing lines emitted by model::executor::forward.
# This intentionally excludes generated model text and all non-forward logs.
# Use fixed-string matching first to avoid regex edge-cases on long mixed-content logs.
rg -F "metallic_foundry::model::executor::forward: Decode " "$SANITIZED_TMP" \
  | rg -e "Decode stage timing \(m=1\):" -e "Decode step hot\[" >"$FORWARD_TMP" || true

if [[ ! -s "$FORWARD_TMP" ]]; then
  echo "No decode timing lines found for metallic_foundry::model::executor::forward in: $LOG_FILE"
  exit 0
fi

echo "Decode Timing Report"
echo "log_file=$LOG_FILE"
echo "top_n=$TOP_N"
echo

echo "[stage-summary]"
grep "Decode stage timing (m=1):" "$FORWARD_TMP" \
  | sed -E 's/.*total=([0-9.]+) us \| qkv=([0-9.]+) us \([^)]*\) \| rope=[0-9.]+ us \([^)]*\) \| fa=([0-9.]+) us \([^)]*\) \| oproj=([0-9.]+) us \([^)]*\) \| other=([0-9.]+) us.*/\1 \2 \3 \4 \5/' \
  | awk '
      { t += $1; q += $2; fa += $3; o += $4; ot += $5; n += 1 }
      END {
        if (n == 0 || t == 0) {
          print "samples=0"
          exit
        }
        printf("samples=%d\n", n)
        printf("avg_total_us=%.2f\n", t / n)
        printf("qkv=%.2f us (%.1f%%)\n", q / n, 100.0 * q / t)
        printf("fa=%.2f us (%.1f%%)\n", fa / n, 100.0 * fa / t)
        printf("oproj=%.2f us (%.1f%%)\n", o / n, 100.0 * o / t)
        printf("other=%.2f us (%.1f%%)\n", ot / n, 100.0 * ot / t)
      }
    '
echo

echo "[hot-steps]"
grep "Decode step hot\\[" "$FORWARD_TMP" \
  | sed -E 's/.*idx=([0-9]+) (.+): total=([0-9.]+) us.*/\1|\2|\3/' \
  | awk -F'|' '
      {
        key = $1 " | " $2
        sum[key] += $3
        cnt[key] += 1
      }
      END {
        for (k in sum) {
          avg = sum[k] / cnt[k]
          printf("%.2f us total | %.2f us avg | %d hits | %s\n", sum[k], avg, cnt[k], k)
        }
      }
    ' \
  | sort -nr \
  | head -n "$TOP_N"
echo

echo "[focused-idx=146]"
grep "Decode step hot\\[.*idx=146 " "$FORWARD_TMP" \
  | sed -E 's/.*total=([0-9.]+) us.*/\1/' \
  | awk '
      {
        s += $1
        if (n == 0 || $1 < mn) mn = $1
        if (n == 0 || $1 > mx) mx = $1
        n += 1
      }
      END {
        if (n == 0) {
          print "samples=0"
        } else {
          printf("samples=%d avg_us=%.2f min_us=%.2f max_us=%.2f\n", n, s / n, mn, mx)
        }
      }
    '
