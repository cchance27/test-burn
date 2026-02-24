#!/usr/bin/env bash
set -euo pipefail

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

HOT_FILES=(
  "crates/metallic-foundry/src/metals/flashattention/decode_kernels.metal"
  "crates/metallic-foundry/src/metals/flashattention/prefill_online.metal"
  "crates/metallic-foundry/src/metals/flashattention/prefill_splitk.metal"
  "crates/metallic-foundry/src/metals/gemv/dot.metal"
  "crates/metallic-foundry/src/metals/gemv/vectorized_stage.metal"
  "crates/metallic-foundry/src/metals/qkv/qkv_project.metal"
  "crates/metallic-foundry/src/metals/swiglu/ffn_stages.metal"
  "crates/metallic-foundry/src/metals/kv_prep/kv_prep_fused.metal"
)

if ! command -v rg >/dev/null 2>&1; then
  echo "error: rg is required" >&2
  exit 2
fi

echo "[index-check] scanning hot kernels for raw (ulong) casts"
violations=0
for f in "${HOT_FILES[@]}"; do
  [[ -f "$f" ]] || continue
  hits=$(rg -n "\\(ulong\\)" "$f" | rg -v "INDEX64_OK" || true)
  if [[ -n "$hits" ]]; then
    echo
    echo "[index-check] $f"
    echo "$hits"
    violations=$((violations + 1))
  fi
done

if [[ "$violations" -eq 0 ]]; then
  echo "[index-check] pass: no raw (ulong) casts in tracked hot files"
  exit 0
fi

if [[ "$STRICT" -eq 1 ]]; then
  echo "[index-check] fail: found $violations file(s) with raw (ulong) casts" >&2
  exit 1
fi

echo "[index-check] warning: found $violations file(s) with raw (ulong) casts"
echo "[index-check] tip: annotate intentional uses with INDEX64_OK and keep out of inner loops"
exit 0
