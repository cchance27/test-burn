#include <metal_stdlib>

using namespace metal;

// Keep vector/tg aliases centralized so storage widening is localized.
#if METALLIC_FASTPATH_INPUT_HALF
typedef half FlashDecodeScalarT;
typedef half2 FlashDecodeVec2T;
typedef half4 FlashDecodeVec4T;
#else
typedef float FlashDecodeScalarT;
typedef float2 FlashDecodeVec2T;
typedef float4 FlashDecodeVec4T;
#endif

template<bool TG_OUT_HALF>
struct FlashTgOut2;
template<>
struct FlashTgOut2<false> {
    using type = float2;
};
template<>
struct FlashTgOut2<true> {
    using type = FlashDecodeVec2T;
};

template<bool TG_OUT_HALF>
struct FlashTgOut4;
template<>
struct FlashTgOut4<false> {
    using type = float4;
};
template<>
struct FlashTgOut4<true> {
    using type = FlashDecodeVec4T;
};

template<bool TG_OUT_HALF>
METAL_FUNC typename FlashTgOut2<TG_OUT_HALF>::type flash_pack_out2(float2 v);
template<>
METAL_FUNC float2 flash_pack_out2<false>(float2 v) {
    return v;
}
template<>
METAL_FUNC FlashDecodeVec2T flash_pack_out2<true>(float2 v) {
    return FlashDecodeVec2T((FlashDecodeScalarT)v[0], (FlashDecodeScalarT)v[1]);
}

template<bool TG_OUT_HALF>
METAL_FUNC float2 flash_unpack_out2(typename FlashTgOut2<TG_OUT_HALF>::type v);
template<>
METAL_FUNC float2 flash_unpack_out2<false>(float2 v) {
    return v;
}
template<>
METAL_FUNC float2 flash_unpack_out2<true>(FlashDecodeVec2T v) {
    return float2((float)v[0], (float)v[1]);
}

template<bool TG_OUT_HALF>
METAL_FUNC typename FlashTgOut4<TG_OUT_HALF>::type flash_pack_out4(float4 v);
template<>
METAL_FUNC float4 flash_pack_out4<false>(float4 v) {
    return v;
}
template<>
METAL_FUNC FlashDecodeVec4T flash_pack_out4<true>(float4 v) {
    return FlashDecodeVec4T((FlashDecodeScalarT)v[0], (FlashDecodeScalarT)v[1], (FlashDecodeScalarT)v[2], (FlashDecodeScalarT)v[3]);
}

template<bool TG_OUT_HALF>
METAL_FUNC float4 flash_unpack_out4(typename FlashTgOut4<TG_OUT_HALF>::type v);
template<>
METAL_FUNC float4 flash_unpack_out4<false>(float4 v) {
    return v;
}
template<>
METAL_FUNC float4 flash_unpack_out4<true>(FlashDecodeVec4T v) {
    return float4((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
}

// =============================================================================
// FlashAttention-style decode (M=1) - warp-tiled over KV
// =============================================================================
//
// Design:
// - Use one threadgroup per (head, batch).
// - Use 8 warps (256 threads) per threadgroup.
// - Each warp processes one KV position at a time, so we process 8 keys per outer iteration.
// - Compute block max/logsumexp over those 8 scores, then update running (m, l) once per block.
// - Accumulate output in registers in warp 0 only.
//
// This is materially closer to FlashAttention than the previous per-position online loop:
// exp() and max/logsum updates are blockwise, and the V accumulation is fused.
//

// Reuses simd reductions in simd.metal (included by stage).

METAL_FUNC float2 half2_to_float2(FlashDecodeVec2T v) {
    return float2(v);
}

METAL_FUNC float4 half4_to_float4(FlashDecodeVec4T v) {
    return float4(v);
}

METAL_FUNC FlashDecodeScalarT flash_load_as_half(const device InputStorageT* ptr, ulong idx) {
    return (FlashDecodeScalarT)ptr[idx];
}

METAL_FUNC FlashDecodeVec2T flash_load_as_half2(const device InputStorageT* ptr, ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return ((const device FlashDecodeVec2T*)((const device FlashDecodeScalarT*)ptr + idx))[0];
#else
    return FlashDecodeVec2T(metallic_load_input_vec2f(ptr, idx));
#endif
}

METAL_FUNC FlashDecodeVec4T flash_load_as_half4(const device InputStorageT* ptr, ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return ((const device FlashDecodeVec4T*)((const device FlashDecodeScalarT*)ptr + idx))[0];
#else
    return FlashDecodeVec4T(metallic_load_input_vec4f(ptr, idx));
#endif
}

METAL_FUNC void flash_store_from_float2(device OutputStorageT* ptr, ulong idx, float2 value) {
    metallic_store_output2_contig(ptr, idx, value);
}

METAL_FUNC void flash_store_from_float4(device OutputStorageT* ptr, ulong idx, float4 value) {
    metallic_store_output4_contig(ptr, idx, value);
}
