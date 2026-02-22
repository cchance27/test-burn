#ifndef GEMV_V2_COMMON_METAL_H
#define GEMV_V2_COMMON_METAL_H

#include <metal_stdlib>
using namespace metal;

// NOTE: ALWAYS_INLINE is provided by policies/base.metal included via policy_f16.metal.

// Dispatch constants (use ifndef to avoid redefinition if already set by stages).
#ifndef ELEMS_PER_THREAD
#define ELEMS_PER_THREAD 8
#endif

#ifndef SIMD_WIDTH
#define SIMD_WIDTH 32
#endif

#ifndef K_CHUNK_SIZE
#define K_CHUNK_SIZE (SIMD_WIDTH * ELEMS_PER_THREAD)
#endif

#ifndef WARPS_PER_TG
#define WARPS_PER_TG 8
#endif

#if defined(METALLIC_POLICY_HAS_AFFINE) && METALLIC_POLICY_HAS_AFFINE
#define METALLIC_AFFINE_LOAD(scale_bytes, idx) ((float)Policy::load_affine((scale_bytes), (idx)))
#define METALLIC_AFFINE_XSUM(xv_lo, xv_hi) (dot((xv_lo), float4(1.0f)) + dot((xv_hi), float4(1.0f)))
#else
#define METALLIC_AFFINE_LOAD(scale_bytes, idx) (0.0f)
#define METALLIC_AFFINE_XSUM(xv_lo, xv_hi) (0.0f)
#endif

#endif // GEMV_V2_COMMON_METAL_H
