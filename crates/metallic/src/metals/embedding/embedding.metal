#include <metal_stdlib>
using namespace metal;

// NOTE: EmbeddingParamsResolved is injected by Foundry via struct_defs()

/** 
 * Generic Embedding Lookup 
 */
template<typename Policy>
void run_embedding_core(
    const device uchar* table,
    const device uchar* scale_bytes,
    const device uint* indices,
    device half* out,
    constant EmbeddingParamsResolved* params,
    uint3 gid
) {
    uint d_model = params->d_model;
    uint total_elements = params->total_elements;
    uint vocab_size = params->vocab_size;

    if (gid.x >= total_elements) return;

    uint pos = gid.x / d_model;
    uint feat = gid.x % d_model;

    uint tok = indices[pos];
    if (tok >= vocab_size) {
        out[gid.x] = (half)0;
        return;
    }

    const uint weights_per_block = 32u;
    uint blocks_per_k = (d_model + weights_per_block - 1u) / weights_per_block;
    uint block = feat / weights_per_block;
    uint scale_idx = (tok * blocks_per_k + block);

    float val[1];
    Policy::template load_weights<1>(table, (ulong)(tok * d_model + feat), val);
    half scale = Policy::load_scale(scale_bytes, (ulong)scale_idx);

    out[gid.x] = (half)(val[0] * (float)scale);
}

