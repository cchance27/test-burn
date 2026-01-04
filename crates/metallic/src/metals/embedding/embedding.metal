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
