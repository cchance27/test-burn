#include <metal_stdlib>
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif
using namespace metal;

// NOTE: EmbeddingParams is injected by Foundry via struct_defs()
// NOTE: Policy types (PolicyF16, PolicyQ8) are included via CompoundKernel.includes()

/**
 * Unified Embedding Lookup Core (Template)
 *
 * Uses Policy trait to handle F16 vs Q8 loading and scaling transparency.
 * Instantiated by CompoundKernel at runtime based on tensor dtype.
 */
template<typename Policy>
ALWAYS_INLINE void run_embedding_core(
    const device uchar* table_bytes,
    const device uchar* scale_bytes,
    const device uint* indices,
    device OutputStorageT* out,
    constant EmbeddingParams* params,
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
        metallic_store_output(out, gid, metallic_to_accum(0.0f));
        return;
    }

    // Policy-driven loading
    ulong offset = (ulong)tok * d_model + feat;

    // Load value (unscaled)
    float val[1];
    Policy::template load_weights<1>(table_bytes, offset, val);

    // Load scale (if applicable) - Q8 uses per-32 block scales, F16 returns 1.0
    float scale = 1.0f;
    float affine = 0.0f;
    if (Policy::HAS_SCALE) {
        const uint weights_per_block = Policy::WEIGHTS_PER_BLOCK;
        uint blocks_per_row = (d_model + weights_per_block - 1u) / weights_per_block;
        uint block_in_row = feat / weights_per_block;
        ulong scale_idx = (ulong)tok * blocks_per_row + block_in_row;
        scale = (float)Policy::load_scale(scale_bytes, scale_idx);
#if defined(METALLIC_POLICY_HAS_AFFINE) && METALLIC_POLICY_HAS_AFFINE
        affine = (float)Policy::load_affine(scale_bytes, scale_idx);
#endif
    }

    metallic_store_output(out, gid, metallic_to_accum(val[0] * scale + affine));
}
