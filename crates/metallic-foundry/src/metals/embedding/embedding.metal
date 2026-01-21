#include <metal_stdlib>
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif
using namespace metal;

// NOTE: EmbeddingParamsResolved is injected by Foundry via struct_defs()

/** 
 * Embedding Lookup
 *
 * We provide two specializations:
 * - F16 table: direct gather of half values
 * - Q8_0 table: int8 weights + per-32 scale (fp16 stored as 2 bytes)
 */
ALWAYS_INLINE void run_embedding_core_f16(
    const device uchar* table_bytes,
    const device uint* indices,
    device half* out,
    constant EmbeddingParamsResolved* params,
    uint gid
) {
    uint d_model = params->d_model;
    uint total_elements = params->total_elements;
    uint vocab_size = params->vocab_size;

    if (gid >= total_elements) return;

    uint pos = gid / d_model;
    uint feat = gid % d_model;

    uint tok = indices[pos];
    if (tok >= vocab_size) {
        out[gid] = (half)0;
        return;
    }

    const device half* table = reinterpret_cast<const device half*>(table_bytes);
    out[gid] = table[tok * d_model + feat];
}

ALWAYS_INLINE half load_q8_scale(const device uchar* scale_bytes, uint scale_idx) {
    const device uchar* s_ptr = scale_bytes + (ulong)scale_idx * 2ul;
    ushort bits = (ushort)s_ptr[0] | ((ushort)s_ptr[1] << 8);
    return as_type<half>(bits);
}

ALWAYS_INLINE void run_embedding_core_q8(
    const device uchar* table_bytes,
    const device uchar* scale_bytes,
    const device uint* indices,
    device half* out,
    constant EmbeddingParamsResolved* params,
    uint gid
) {
    uint d_model = params->d_model;
    uint total_elements = params->total_elements;
    uint vocab_size = params->vocab_size;

    if (gid >= total_elements) return;

    uint pos = gid / d_model;
    uint feat = gid % d_model;

    uint tok = indices[pos];
    if (tok >= vocab_size) {
        out[gid] = (half)0;
        return;
    }

    const uint weights_per_block = 32u;
    uint blocks_per_k = (d_model + weights_per_block - 1u) / weights_per_block;
    uint block = feat / weights_per_block;
    uint scale_idx = (tok * blocks_per_k + block);

    const device char* w = reinterpret_cast<const device char*>(table_bytes);
    float v = (float)w[tok * d_model + feat];
    half scale = load_q8_scale(scale_bytes, scale_idx);
    out[gid] = (half)(v * (float)scale);
}

kernel void embedding_lookup_f16(
    const device uchar* table [[buffer(0)]],
    const device uint* indices [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParamsResolved* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    run_embedding_core_f16(table, indices, output, params, gid);
}

kernel void embedding_lookup_q8(
    const device uchar* table [[buffer(0)]],
    const device uchar* scale_bytes [[buffer(1)]],
    const device uint* indices [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant EmbeddingParamsResolved* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    run_embedding_core_q8(table, scale_bytes, indices, output, params, gid);
}
