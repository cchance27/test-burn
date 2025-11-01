#!/usr/bin/env bash
set -euo pipefail

SDK=${SDK:-macosx}
METAL_STD=${METAL_STD:-metal4.0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MATMUL_DIR="${SCRIPT_DIR}/matmul"
BUILD_DIR="${SCRIPT_DIR}/.build/matmul"

mkdir -p "${MATMUL_DIR}"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

declare -a ORIGINAL_SOURCES=(
  "crates/metallic/src/kernels/matmul_mlx/mlx.metal:original_mlx.metal"
  "crates/metallic/src/kernels/matmul_gemv/kernel.metal:original_gemv.metal"
  "crates/metallic/src/kernels/matmul_gemv_smalln/kernel.metal:original_gemv_smalln.metal"
  "crates/metallic/src/kernels/matmul_gemm_tiled/kernel.metal:original_gemm_tiled.metal"
)

for mapping in "${ORIGINAL_SOURCES[@]}"; do
  src_rel="${mapping%%:*}"
  dst_name="${mapping##*:}"
  src_path="${REPO_ROOT}/${src_rel}"
  dst_path="${MATMUL_DIR}/${dst_name}"
  if [[ ! -f "${src_path}" ]]; then
    echo "Expected source kernel missing: ${src_path}" >&2
    exit 1
  fi
  cp -f "${src_path}" "${dst_path}"
done

shopt -s nullglob
sources=("${MATMUL_DIR}"/*.metal)
shopt -u nullglob

if [[ ${#sources[@]} -eq 0 ]]; then
  echo "No .metal probe files found in ${MATMUL_DIR}" >&2
  exit 1
fi

for src in "${sources[@]}"; do
  base="$(basename "${src}" .metal)"
  air="${BUILD_DIR}/${base}.air"
  metallib="${BUILD_DIR}/${base}.metallib"
  echo "Compiling ${base}.metal → ${base}.metallib"
  xcrun -sdk "${SDK}" metal -c -std="${METAL_STD}" "${src}" -o "${air}"
  xcrun -sdk "${SDK}" metallib "${air}" -o "${metallib}"
done

echo "Matmul probe compilation succeeded."

if [[ "${METAL_RUN:-0}" == "1" ]]; then
  echo "Executing matmul probe harness..."
  xcrun swift "${SCRIPT_DIR}/run_matmul_probes.swift" "${BUILD_DIR}" "${MATMUL_DIR}" "${SCRIPT_DIR}/MATMUL_QWEN25_SIZES.md"
fi
