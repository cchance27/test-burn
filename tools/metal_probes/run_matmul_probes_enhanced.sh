#!/usr/bin/env bash
set -euo pipefail

SDK=${SDK:-macosx}
METAL_STD=${METAL_STD:-metal4.0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ensure we run from the script directory so relative paths and caches land here
cd "${SCRIPT_DIR}"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MATMUL_DIR="${SCRIPT_DIR}/matmul"
BUILD_DIR="${SCRIPT_DIR}/.build/matmul"

mkdir -p "${MATMUL_DIR}"
mkdir -p "${BUILD_DIR}"
MODULE_CACHE_DIR="${BUILD_DIR}/module_cache"
FAKE_HOME="${BUILD_DIR}/home"
mkdir -p "${MODULE_CACHE_DIR}" "${FAKE_HOME}"
export METAL_MODULE_CACHE_DIR="${MODULE_CACHE_DIR}"
export MTL_MODULE_CACHE_DIR="${MODULE_CACHE_DIR}"
export CLANG_MODULE_CACHE_PATH="${MODULE_CACHE_DIR}"
export HOME="${FAKE_HOME}"

declare -a ORIGINAL_SOURCES=(
  "crates/metallic/src/kernels/matmul_mlx/mlx.metal:original_mlx.metal"
  "crates/metallic/src/kernels/matmul_gemv/kernel.metal:original_gemv.metal"
  "crates/metallic/src/kernels/matmul_gemm_tiled/kernel.metal:original_gemm_tiled.metal"
)

# Copy original sources (from the main repository)
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

# Note: The enhanced Swift code uses variants_enhanced.json directly,
# so we don't overwrite the original variants.json to preserve legacy compatibility
if [[ -f "${MATMUL_DIR}/variants_enhanced.json" ]]; then
  echo "Enhanced variants configuration available, enhanced Swift code will use it"
else
  echo "Warning: Enhanced variants file not found, enhanced harness may fall back to defaults"
fi

# Also copy any additional .metal files in the matmul directory
shopt -s nullglob
sources=("${MATMUL_DIR}"/*.metal)
shopt -u nullglob

if [[ ${#sources[@]} -eq 0 ]]; then
  echo "No .metal probe files found in ${MATMUL_DIR}" >&2
  exit 1
fi

for src in "${sources[@]}"; do
  base="$(basename "${src}" .metal)"
  hash=$(shasum -a 256 "${src}" | awk '{print $1}')
  
  # Check for existing compiled files for this base name
  existing_libs=("${BUILD_DIR}/${base}."*.metallib)
  found_match=false
  
  if [[ -e "${existing_libs[0]}" ]]; then
    for existing_lib in "${existing_libs[@]}"; do
      existing_hash=$(basename "${existing_lib}" .metallib | cut -d. -f2-)
      if [[ "${existing_hash}" == "${hash}" ]]; then
        echo "Skipping compilation for ${base}.metal (hash match)"
        found_match=true
        break
      else
        echo "Removing outdated compiled files for ${base}.metal"
        rm -f "${BUILD_DIR}/${base}.${existing_hash}.air" "${BUILD_DIR}/${base}.${existing_hash}.metallib"
      fi
    done
  fi

  if [[ "${found_match}" == "false" ]]; then
    air="${BUILD_DIR}/${base}.${hash}.air"
    metallib="${BUILD_DIR}/${base}.${hash}.metallib"
    echo "Compiling ${base}.metal → ${base}.${hash}.metallib"
    xcrun -sdk "${SDK}" metal -c -std="${METAL_STD}" "${src}" -o "${air}"
    xcrun -sdk "${SDK}" metallib "${air}" -o "${metallib}"
  fi
done

echo "Enhanced matmul probe compilation succeeded."

if [[ "${METAL_RUN:-0}" == "1" ]]; then
  echo "Executing enhanced matmul probe harness..."
  # Create temporary Swift file that imports all modules
  temp_swift="${BUILD_DIR}/enhanced_matmul_probes.swift"
  
  # Combine all Swift modules into a single file for execution
  {
    echo 'import Foundation'
    echo 'import Metal'
    echo 'import MetalPerformanceShaders'
    echo 'import Darwin'
    echo ''
  } > "${temp_swift}"

  # Append contents of all Swift files in the correct order
  for file in matmul_types.swift variant_manager.swift base_backend_runner.swift generic_kernel_runner.swift parser.swift main_harness.swift utils_and_main.swift; do
    if [[ -f "${SCRIPT_DIR}/${file}" ]]; then
      echo "// MARK: - Content from ${file}" >> "${temp_swift}"
      cat "${SCRIPT_DIR}/${file}" >> "${temp_swift}"
      echo "" >> "${temp_swift}"  # Add a blank line between files
    else
      echo "Error: Swift file not found: ${file}" >&2
      exit 1
    fi
  done

  # Forward any CLI flags to the Swift harness (e.g., --bestvsbaseline)
  xcrun swift "${temp_swift}" "${BUILD_DIR}" "${MATMUL_DIR}" "${SCRIPT_DIR}/MATMUL_QWEN25_SIZES.md" "$@"
fi
