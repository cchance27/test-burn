#include <metal_stdlib>
using namespace metal;

struct EpilogueNone {
    template<uint HEADS>
    static void apply(
        float acc[HEADS],
        uint lane_id,
        uint logical_col,
        const uint N[HEADS],
        const device half *bias[HEADS],
        const uint has_bias_flags[HEADS],
        const float alpha,
        const float beta,
        const device half *residual,
        device half *result_y[HEADS]
    ) {
        // Perform standard reduction across thread group lanes.
        
        for (uint h = 0; h < HEADS; ++h) {
            if (logical_col >= N[h]) continue;

            float val = acc[h];
            val += simd_shuffle_xor(val, 16u);
            val += simd_shuffle_xor(val, 8u);
            val += simd_shuffle_xor(val, 4u);
            val += simd_shuffle_xor(val, 2u);
            val += simd_shuffle_xor(val, 1u);
            
            if (lane_id == 0) {
                 // Basic Write
                 result_y[h][logical_col] = (half)val;
            }
        }
    }
};
