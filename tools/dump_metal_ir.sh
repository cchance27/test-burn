#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <input.metal> <output.ll> [include_dir]" >&2
  exit 2
fi

IN="$1"
OUT="$2"
INC_DIR="${3:-}"

if [[ ! -f "$IN" ]]; then
  echo "error: input not found: $IN" >&2
  exit 2
fi

ARGS=()
if [[ -n "$INC_DIR" ]]; then
  ARGS+=("-I" "$INC_DIR")
fi

# Notes:
# - We use -O to approximate runtime compilation optimizations.
# - -emit-llvm -S emits textual LLVM IR.
if (( ${#ARGS[@]} )); then
  xcrun metal -O -emit-llvm -S "${ARGS[@]}" -o "$OUT" "$IN"
else
  xcrun metal -O -emit-llvm -S -o "$OUT" "$IN"
fi

echo "wrote: $OUT"
