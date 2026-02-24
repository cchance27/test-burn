#ifndef GEMV_V2_SCALAR_OUTPUT_METAL_H
#define GEMV_V2_SCALAR_OUTPUT_METAL_H

/// Thread-per-row scalar dot product helper (used by ScalarDotStage).
template<typename Policy, uint UNROLL>
ALWAYS_INLINE float run_gemv_scalar_dot_stage(
    const device uchar *weights,
    const device uchar *scale_bytes,
    const device InputStorageT *input,
    const uint row_idx,
    const uint k_dim,
    const uint n_dim,
    const uint weights_per_block
) {
    float acc = 0.0f;
    uint k = 0u;
    const uint blocks_per_k_dim = (k_dim + weights_per_block - 1u) / weights_per_block;

    // Main loop with unrolling.
    while (k + UNROLL <= k_dim) {
        #pragma unroll
        for (uint i = 0u; i < UNROLL; ++i) {
            uint curr_k = k + i;

            float val_x = (float)metallic_load_input(input, curr_k);

            ulong idx = WEIGHT_INDEX(row_idx, curr_k, k_dim, n_dim);
            float w;
            Policy::template load_weights<1>(weights, idx, &w);

            ulong scale_idx = (ulong)row_idx * blocks_per_k_dim + (curr_k / weights_per_block);
            ComputeT scale = (ComputeT)Policy::load_scale(scale_bytes, scale_idx);
            float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;

            acc += val_x * (w * (ComputeT)scale + affine);
        }
        k += UNROLL;
    }

    // Tail loop.
    while (k < k_dim) {
        float val_x = (float)metallic_load_input(input, k);
        ulong idx = WEIGHT_INDEX(row_idx, k, k_dim, n_dim);
        float w;
        Policy::template load_weights<1>(weights, idx, &w);

        ulong scale_idx = (ulong)row_idx * blocks_per_k_dim + (k / weights_per_block);
        ComputeT scale = (ComputeT)Policy::load_scale(scale_bytes, scale_idx);
        float affine = Policy::HAS_AFFINE ? (float)Policy::load_affine(scale_bytes, scale_idx) : 0.0f;

        acc += val_x * (w * (ComputeT)scale + affine);
        ++k;
    }

    return acc;
}

/// Apply bias and write output.
ALWAYS_INLINE void gemv_write_output(
    device OutputStorageT *output,
    const device BiasStorageT *bias,
    const uint row_idx,
    const float value,
    const bool has_bias
) {
    AccumT result = metallic_to_accum(value);
    if (has_bias) {
        result += metallic_to_accum(metallic_load_bias(bias, row_idx));
    }
    metallic_store_output(output, row_idx, result);
}

#endif // GEMV_V2_SCALAR_OUTPUT_METAL_H
