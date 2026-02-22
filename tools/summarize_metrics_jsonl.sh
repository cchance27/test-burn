#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <metrics.jsonl> [top_n]" >&2
  exit 2
fi

FILE="$1"
TOP_N="${2:-20}"

if [[ ! -f "$FILE" ]]; then
  echo "error: file not found: $FILE" >&2
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required to summarize metrics jsonl" >&2
  exit 2
fi

# Print: total_us  count  op_name
# Note: we slurp the file; for very large traces, run with smaller max-tokens / iterations.
jq -r -s --argjson top "$TOP_N" '
  map(select(.event.type == "GpuOpCompleted"))
  | group_by(.event.data.op_name)
  | map({
      op: (.[0].event.data.op_name),
      count: length,
      total_us: (map(.event.data.duration_us) | add)
    })
  | sort_by(.total_us)
  | reverse
  | .[0:$top]
  | .[]
  | "\(.total_us)\t\(.count)\t\(.op)"
' "$FILE" \
  | awk -F'\t' 'BEGIN{printf("%-12s %-8s %s\n","total_us","count","op_name")} {printf("%-12s %-8s %s\n",$1,$2,$3)}'
