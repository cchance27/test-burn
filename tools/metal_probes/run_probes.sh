#!/usr/bin/env bash
set -euo pipefail

SDK=${SDK:-macosx}
METAL_STD=${METAL_STD:-metal4.0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/.build"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

shopt -s nullglob
sources=("${SCRIPT_DIR}"/*.metal)
shopt -u nullglob

if [[ ${#sources[@]} -eq 0 ]]; then
  echo "No .metal probe files found in ${SCRIPT_DIR}" >&2
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

echo "Metal probe compilation succeeded."

if [[ "${METAL_RUN:-0}" == "1" ]]; then
  echo "Executing probe harness..."
  xcrun swift "${SCRIPT_DIR}/run_probes.swift" "${BUILD_DIR}"
fi
