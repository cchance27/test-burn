#ifndef METALLIC_RUNTIME_TYPES_METAL_H
#define METALLIC_RUNTIME_TYPES_METAL_H

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// Storage types (defaults preserve current behavior).
#ifndef METALLIC_DTYPE_TENSOR_STORAGE
#define METALLIC_DTYPE_TENSOR_STORAGE half
#endif

#ifndef METALLIC_DTYPE_OUTPUT_STORAGE
#define METALLIC_DTYPE_OUTPUT_STORAGE METALLIC_DTYPE_TENSOR_STORAGE
#endif

#ifndef METALLIC_DTYPE_BIAS_STORAGE
#define METALLIC_DTYPE_BIAS_STORAGE METALLIC_DTYPE_TENSOR_STORAGE
#endif

#ifndef METALLIC_DTYPE_RESIDUAL_STORAGE
#define METALLIC_DTYPE_RESIDUAL_STORAGE METALLIC_DTYPE_TENSOR_STORAGE
#endif

#ifndef METALLIC_DTYPE_GAMMA_STORAGE
#define METALLIC_DTYPE_GAMMA_STORAGE METALLIC_DTYPE_TENSOR_STORAGE
#endif

#ifndef METALLIC_DTYPE_INPUT_STORAGE
#define METALLIC_DTYPE_INPUT_STORAGE METALLIC_DTYPE_TENSOR_STORAGE
#endif

// Math types.
#ifndef METALLIC_DTYPE_COMPUTE
#define METALLIC_DTYPE_COMPUTE float
#endif

#ifndef METALLIC_DTYPE_ACCUM
#define METALLIC_DTYPE_ACCUM float
#endif

// Fast vector lane types used by hot kernel internals.
// Default maps to FP16 vector lanes; callers can override via compile-time defines.
#ifndef METALLIC_DTYPE_FAST_SCALAR
#define METALLIC_DTYPE_FAST_SCALAR half
#endif

#ifndef METALLIC_DTYPE_FAST_VEC2
#define METALLIC_DTYPE_FAST_VEC2 half2
#endif

#ifndef METALLIC_DTYPE_FAST_VEC4
#define METALLIC_DTYPE_FAST_VEC4 half4
#endif

// Index width specialization. 32-bit is default for hot paths.
#ifndef METALLIC_INDEX_WIDTH
#define METALLIC_INDEX_WIDTH 32
#endif

#if METALLIC_INDEX_WIDTH == 64
typedef ulong IndexT;
#define METALLIC_INDEX_IS_64 1
#else
typedef uint IndexT;
#define METALLIC_INDEX_IS_64 0
#endif

// Compile-time storage classification helpers.
#define METALLIC_TYPE_IS_HALF_half 1
#define METALLIC_TYPE_IS_HALF_float 0
#define METALLIC_TYPE_IS_HALF_bfloat 0
#define METALLIC_TYPE_IS_HALF_uchar 0
#define METALLIC_TYPE_IS_HALF_uint 0
#define METALLIC_TYPE_IS_HALF_ulong 0
#define METALLIC_TYPE_IS_HALF_IMPL(x) METALLIC_TYPE_IS_HALF_##x
#define METALLIC_TYPE_IS_HALF(x) METALLIC_TYPE_IS_HALF_IMPL(x)

// Canonical fast-path predicates to keep kernels DRY.
#define METALLIC_FASTPATH_INPUT_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_INPUT_STORAGE))
#define METALLIC_FASTPATH_OUTPUT_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_OUTPUT_STORAGE))
#define METALLIC_FASTPATH_TENSOR_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_TENSOR_STORAGE))
#define METALLIC_FASTPATH_GAMMA_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_GAMMA_STORAGE))
#define METALLIC_FASTPATH_BIAS_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_BIAS_STORAGE))
#define METALLIC_FASTPATH_RESIDUAL_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_RESIDUAL_STORAGE))
#define METALLIC_FASTPATH_COMPUTE_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_COMPUTE))
#define METALLIC_FASTPATH_ACCUM_HALF (METALLIC_TYPE_IS_HALF(METALLIC_DTYPE_ACCUM))
#define METALLIC_FASTPATH_F16_IO (METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_OUTPUT_HALF)
#define METALLIC_FASTPATH_F16_ALL \
    (METALLIC_FASTPATH_INPUT_HALF && METALLIC_FASTPATH_OUTPUT_HALF && METALLIC_FASTPATH_TENSOR_HALF && \
     METALLIC_FASTPATH_BIAS_HALF && METALLIC_FASTPATH_RESIDUAL_HALF && METALLIC_FASTPATH_GAMMA_HALF)

typedef METALLIC_DTYPE_TENSOR_STORAGE TensorStorageT;
typedef METALLIC_DTYPE_INPUT_STORAGE InputStorageT;
typedef METALLIC_DTYPE_OUTPUT_STORAGE OutputStorageT;
typedef METALLIC_DTYPE_BIAS_STORAGE BiasStorageT;
typedef METALLIC_DTYPE_RESIDUAL_STORAGE ResidualStorageT;
typedef METALLIC_DTYPE_GAMMA_STORAGE GammaStorageT;
typedef METALLIC_DTYPE_COMPUTE ComputeT;
typedef METALLIC_DTYPE_ACCUM AccumT;
typedef METALLIC_DTYPE_FAST_SCALAR FastScalarT;
typedef METALLIC_DTYPE_FAST_VEC2 FastVec2T;
typedef METALLIC_DTYPE_FAST_VEC4 FastVec4T;

ALWAYS_INLINE IndexT metallic_idx1(const IndexT base, const IndexT i) {
    return base + i;
}

ALWAYS_INLINE IndexT metallic_idx2(const IndexT i, const IndexT stride_i, const IndexT j) {
    return i * stride_i + j;
}

ALWAYS_INLINE IndexT metallic_idx3(const IndexT i, const IndexT stride_i, const IndexT j, const IndexT stride_j, const IndexT k) {
    return i * stride_i + j * stride_j + k;
}

template<typename T>
ALWAYS_INLINE ComputeT metallic_to_compute(const T value) {
    return (ComputeT)value;
}

template<typename T>
ALWAYS_INLINE AccumT metallic_to_accum(const T value) {
    return (AccumT)value;
}

ALWAYS_INLINE OutputStorageT metallic_to_output(const AccumT value) {
    return (OutputStorageT)value;
}

ALWAYS_INLINE ComputeT metallic_load_input(const device InputStorageT* ptr, const IndexT idx) {
    return metallic_to_compute(ptr[idx]);
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE ComputeT metallic_load_input(const device InputStorageT* ptr, const ulong idx) {
    return metallic_load_input(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE ComputeT metallic_load_tensor(const device TensorStorageT* ptr, const IndexT idx) {
    return metallic_to_compute(ptr[idx]);
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE ComputeT metallic_load_tensor(const device TensorStorageT* ptr, const ulong idx) {
    return metallic_load_tensor(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE ComputeT metallic_load_bias(const device BiasStorageT* ptr, const IndexT idx) {
    return metallic_to_compute(ptr[idx]);
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE ComputeT metallic_load_bias(const device BiasStorageT* ptr, const ulong idx) {
    return metallic_load_bias(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE ComputeT metallic_load_residual(const device ResidualStorageT* ptr, const IndexT idx) {
    return metallic_to_compute(ptr[idx]);
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE ComputeT metallic_load_residual(const device ResidualStorageT* ptr, const ulong idx) {
    return metallic_load_residual(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE ComputeT metallic_load_gamma(const device GammaStorageT* ptr, const IndexT idx) {
    return metallic_to_compute(ptr[idx]);
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE ComputeT metallic_load_gamma(const device GammaStorageT* ptr, const ulong idx) {
    return metallic_load_gamma(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE float2 metallic_load_input_vec2f(const device InputStorageT* ptr, const IndexT idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return float2(((const device half2*)((const device half*)ptr + idx))[0]);
#else
    return float2(
        (float)metallic_load_input(ptr, idx + (IndexT)0),
        (float)metallic_load_input(ptr, idx + (IndexT)1)
    );
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE float2 metallic_load_input_vec2f(const device InputStorageT* ptr, const ulong idx) {
    return metallic_load_input_vec2f(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE float4 metallic_load_input_vec4f(const device InputStorageT* ptr, const IndexT idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return float4(((const device half4*)((const device half*)ptr + idx))[0]);
#else
    return float4(
        (float)metallic_load_input(ptr, idx + (IndexT)0),
        (float)metallic_load_input(ptr, idx + (IndexT)1),
        (float)metallic_load_input(ptr, idx + (IndexT)2),
        (float)metallic_load_input(ptr, idx + (IndexT)3)
    );
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE float4 metallic_load_input_vec4f(const device InputStorageT* ptr, const ulong idx) {
    return metallic_load_input_vec4f(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE float2 metallic_load_tensor_vec2f(const device TensorStorageT* ptr, const IndexT idx) {
#if METALLIC_FASTPATH_TENSOR_HALF
    return float2(((const device half2*)((const device half*)ptr + idx))[0]);
#else
    return float2(
        (float)metallic_load_tensor(ptr, idx + (IndexT)0),
        (float)metallic_load_tensor(ptr, idx + (IndexT)1)
    );
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE float2 metallic_load_tensor_vec2f(const device TensorStorageT* ptr, const ulong idx) {
    return metallic_load_tensor_vec2f(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE float4 metallic_load_tensor_vec4f(const device TensorStorageT* ptr, const IndexT idx) {
#if METALLIC_FASTPATH_TENSOR_HALF
    return float4(((const device half4*)((const device half*)ptr + idx))[0]);
#else
    return float4(
        (float)metallic_load_tensor(ptr, idx + (IndexT)0),
        (float)metallic_load_tensor(ptr, idx + (IndexT)1),
        (float)metallic_load_tensor(ptr, idx + (IndexT)2),
        (float)metallic_load_tensor(ptr, idx + (IndexT)3)
    );
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE float4 metallic_load_tensor_vec4f(const device TensorStorageT* ptr, const ulong idx) {
    return metallic_load_tensor_vec4f(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE float2 metallic_load_gamma_vec2f(const device GammaStorageT* ptr, const IndexT idx) {
#if METALLIC_FASTPATH_GAMMA_HALF
    return float2(((const device half2*)((const device half*)ptr + idx))[0]);
#else
    return float2(
        (float)metallic_load_gamma(ptr, idx + (IndexT)0),
        (float)metallic_load_gamma(ptr, idx + (IndexT)1)
    );
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE float2 metallic_load_gamma_vec2f(const device GammaStorageT* ptr, const ulong idx) {
    return metallic_load_gamma_vec2f(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE float4 metallic_load_gamma_vec4f(const device GammaStorageT* ptr, const IndexT idx) {
#if METALLIC_FASTPATH_GAMMA_HALF
    return float4(((const device half4*)((const device half*)ptr + idx))[0]);
#else
    return float4(
        (float)metallic_load_gamma(ptr, idx + (IndexT)0),
        (float)metallic_load_gamma(ptr, idx + (IndexT)1),
        (float)metallic_load_gamma(ptr, idx + (IndexT)2),
        (float)metallic_load_gamma(ptr, idx + (IndexT)3)
    );
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE float4 metallic_load_gamma_vec4f(const device GammaStorageT* ptr, const ulong idx) {
    return metallic_load_gamma_vec4f(ptr, (IndexT)idx);
}
#endif

ALWAYS_INLINE void metallic_store_output(device OutputStorageT* ptr, const IndexT idx, const AccumT value) {
    // Conversion-elision fast path when accum/output are both FP16 lanes.
#if METALLIC_FASTPATH_OUTPUT_HALF && METALLIC_FASTPATH_ACCUM_HALF
    ptr[idx] = (OutputStorageT)value;
#else
    ptr[idx] = metallic_to_output(value);
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE void metallic_store_output(device OutputStorageT* ptr, const ulong idx, const AccumT value) {
    metallic_store_output(ptr, (IndexT)idx, value);
}
#endif

// Forward declarations for auto-routing in indexed helpers.
ALWAYS_INLINE void metallic_store_output2_contig(device OutputStorageT* ptr, const IndexT idx, const float2 value);
ALWAYS_INLINE void metallic_store_output4_contig(device OutputStorageT* ptr, const IndexT idx, const float4 value);

ALWAYS_INLINE void metallic_store_output2(device OutputStorageT* ptr, const IndexT idx0, const IndexT idx1, const float2 value) {
    if (idx1 == idx0 + (IndexT)1) {
        metallic_store_output2_contig(ptr, idx0, value);
        return;
    }
    metallic_store_output(ptr, idx0, metallic_to_accum(value[0]));
    metallic_store_output(ptr, idx1, metallic_to_accum(value[1]));
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE void metallic_store_output2(device OutputStorageT* ptr, const ulong idx0, const ulong idx1, const float2 value) {
    metallic_store_output2(ptr, (IndexT)idx0, (IndexT)idx1, value);
}
#endif

ALWAYS_INLINE void metallic_store_output4(
    device OutputStorageT* ptr,
    const IndexT idx0,
    const IndexT idx1,
    const IndexT idx2,
    const IndexT idx3,
    const float4 value
) {
    if (idx1 == idx0 + (IndexT)1 && idx2 == idx1 + (IndexT)1 && idx3 == idx2 + (IndexT)1) {
        metallic_store_output4_contig(ptr, idx0, value);
        return;
    }
    metallic_store_output(ptr, idx0, metallic_to_accum(value[0]));
    metallic_store_output(ptr, idx1, metallic_to_accum(value[1]));
    metallic_store_output(ptr, idx2, metallic_to_accum(value[2]));
    metallic_store_output(ptr, idx3, metallic_to_accum(value[3]));
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE void metallic_store_output4(
    device OutputStorageT* ptr,
    const ulong idx0,
    const ulong idx1,
    const ulong idx2,
    const ulong idx3,
    const float4 value
) {
    metallic_store_output4(ptr, (IndexT)idx0, (IndexT)idx1, (IndexT)idx2, (IndexT)idx3, value);
}
#endif

// Contiguous vector stores (idx+1 / idx+2 / idx+3 layout) for hot paths.
ALWAYS_INLINE void metallic_store_output2_contig(device OutputStorageT* ptr, const IndexT idx, const float2 value) {
#if METALLIC_FASTPATH_OUTPUT_HALF
    ((device half2*)((device half*)ptr + idx))[0] = half2((half)value[0], (half)value[1]);
#else
    metallic_store_output(ptr, idx + (IndexT)0, metallic_to_accum(value[0]));
    metallic_store_output(ptr, idx + (IndexT)1, metallic_to_accum(value[1]));
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE void metallic_store_output2_contig(device OutputStorageT* ptr, const ulong idx, const float2 value) {
    metallic_store_output2_contig(ptr, (IndexT)idx, value);
}
#endif

ALWAYS_INLINE void metallic_store_output4_contig(device OutputStorageT* ptr, const IndexT idx, const float4 value) {
#if METALLIC_FASTPATH_OUTPUT_HALF
    ((device half4*)((device half*)ptr + idx))[0] = half4((half)value[0], (half)value[1], (half)value[2], (half)value[3]);
#else
    metallic_store_output(ptr, idx + (IndexT)0, metallic_to_accum(value[0]));
    metallic_store_output(ptr, idx + (IndexT)1, metallic_to_accum(value[1]));
    metallic_store_output(ptr, idx + (IndexT)2, metallic_to_accum(value[2]));
    metallic_store_output(ptr, idx + (IndexT)3, metallic_to_accum(value[3]));
#endif
}
#if !METALLIC_INDEX_IS_64
ALWAYS_INLINE void metallic_store_output4_contig(device OutputStorageT* ptr, const ulong idx, const float4 value) {
    metallic_store_output4_contig(ptr, (IndexT)idx, value);
}
#endif

#endif // METALLIC_RUNTIME_TYPES_METAL_H
