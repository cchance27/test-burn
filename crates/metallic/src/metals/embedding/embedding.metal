#include <metal_stdlib>
using namespace metal;

// NOTE: EmbeddingParamsResolved is injected by Foundry via struct_defs()

/// Embedding lookup kernel for half precision.
///
/// Each thread copies one output element from the embedding table.
/// Inputs:
///   - table: [vocab_size, d_model] row-major embedding matrix
///   - indices: [total_tokens] uint token ids
///   - out: [total_tokens, d_model] flattened output
///   - params: { d_model, total_elements, vocab_size }
kernel void embedding_lookup_f16(
    const device half* table [[buffer(0)]],
    const device uint* indices [[buffer(1)]],
    device half* out [[buffer(2)]],
    constant EmbeddingParamsResolved* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint d_model = params->d_model;
    uint total_elements = params->total_elements;
    uint vocab_size = params->vocab_size;
    
    if (gid >= total_elements) return;
    
    // Compute position and feature index
    uint pos = gid / d_model;      // Which token position
    uint feat = gid % d_model;     // Which feature dimension
    
    // Look up token id
    uint tok = indices[pos];
    
    // Bounds check - output zero for out-of-vocab tokens
    if (tok >= vocab_size) {
        out[gid] = (half)0;
        return;
    }
    
    // Copy from table[tok, feat] to out[gid]
    uint src = tok * d_model + feat;
    out[gid] = table[src];
}

/// Embedding lookup kernel for Q8_0 (split int8 weights + fp16 block scales).
///
/// Layout assumptions:
/// - `table` is stored as a dense [vocab_size, d_model] int8 array (1 byte per weight).
/// - `scale_bytes` stores fp16 scales per 32-weight block, row-major:
///     scale_idx = tok * blocks_per_k + (feat / 32)
kernel void embedding_lookup_q8(
    const device uchar* table [[buffer(0)]],
    const device uchar* scale_bytes [[buffer(1)]],
    const device uint* indices [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant EmbeddingParamsResolved* params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
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

    // One fp16 scale per 32-weight block along feature dimension.
    const uint weights_per_block = 32u;
    uint blocks_per_k = (d_model + weights_per_block - 1u) / weights_per_block;
    uint block = feat / weights_per_block;
    uint scale_idx = (tok * blocks_per_k + block) * 2u;

    // Load fp16 scale (little-endian)
    ushort bits = (ushort)scale_bytes[scale_idx] | ((ushort)scale_bytes[scale_idx + 1u] << 8);
    half scale = as_type<half>(bits);

    // Load int8 weight and dequantize.
    uint src = tok * d_model + feat;
    char q = (char)table[src];
    out[gid] = (half)((float)q * (float)scale);
}
