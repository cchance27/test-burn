#include <metal_stdlib>

using namespace metal;

template<bool TG_OUT_HALF>
struct FlashTgOut2;
template<>
struct FlashTgOut2<false> {
    using type = float2;
};
template<>
struct FlashTgOut2<true> {
    using type = half2;
};

template<bool TG_OUT_HALF>
struct FlashTgOut4;
template<>
struct FlashTgOut4<false> {
    using type = float4;
};
template<>
struct FlashTgOut4<true> {
    using type = half4;
};

template<bool TG_OUT_HALF>
METAL_FUNC typename FlashTgOut2<TG_OUT_HALF>::type flash_pack_out2(float2 v);
template<>
METAL_FUNC float2 flash_pack_out2<false>(float2 v) {
    return v;
}
template<>
METAL_FUNC half2 flash_pack_out2<true>(float2 v) {
    return half2((half)v[0], (half)v[1]);
}

template<bool TG_OUT_HALF>
METAL_FUNC float2 flash_unpack_out2(typename FlashTgOut2<TG_OUT_HALF>::type v);
template<>
METAL_FUNC float2 flash_unpack_out2<false>(float2 v) {
    return v;
}
template<>
METAL_FUNC float2 flash_unpack_out2<true>(half2 v) {
    return float2((float)v[0], (float)v[1]);
}

template<bool TG_OUT_HALF>
METAL_FUNC typename FlashTgOut4<TG_OUT_HALF>::type flash_pack_out4(float4 v);
template<>
METAL_FUNC float4 flash_pack_out4<false>(float4 v) {
    return v;
}
template<>
METAL_FUNC half4 flash_pack_out4<true>(float4 v) {
    return half4((half)v[0], (half)v[1], (half)v[2], (half)v[3]);
}

template<bool TG_OUT_HALF>
METAL_FUNC float4 flash_unpack_out4(typename FlashTgOut4<TG_OUT_HALF>::type v);
template<>
METAL_FUNC float4 flash_unpack_out4<false>(float4 v) {
    return v;
}
template<>
METAL_FUNC float4 flash_unpack_out4<true>(half4 v) {
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

METAL_FUNC float2 half2_to_float2(half2 v) {
    return float2((float)v[0], (float)v[1]);
}

METAL_FUNC float4 half4_to_float4(half4 v) {
    return float4((float)v[0], (float)v[1], (float)v[2], (float)v[3]);
}
