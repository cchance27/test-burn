#ifndef METALLIC_SIMD_GEMV_CANONICAL_METAL
#define METALLIC_SIMD_GEMV_CANONICAL_METAL

#include <metal_stdlib>
using namespace metal;

// =================================================================================================
// Canonical GEMV Strategy (Generic over QuantPolicy)
// =================================================================================================
//
// Handles the loop structure, barrier synchronization, and register unrolling for
// SIMD GEMV kernels. It delegates data loading and dot product math to the `Quant` policy.

template<typename Quant>
struct CanonicalStrategy {
    typedef typename Quant::Params Params;

    // Strategy State
    Quant quant;
    
    // Register State (4-way unroll per loop step)
    float4 xv0, xv1; 
    
    // Loop State
    uint block_idx_0, block_idx_1, total_blocks;
    
    // Constants (exposed for driver)
    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    template<uint HEADS>
    void init(typename Quant::Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        // Init Quant (handles strides, pointers, etc.)
        uint lane_id = lid.x & 31u;
        uint block_in_group = lane_id / 8u;
        uint sub_lane = lane_id % 8u;
        uint sub_offset = sub_lane * 4u; // 4 elements per sub-lane
        
        quant.template init<HEADS>(p, lid, logical_col, K, N, gp, batch_idx, block_in_group, sub_offset);
        
        // Init Block indices
        uint weights_per_block = quant.weights_per_block;
        total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    }
    
    void load_x_fast(const device half *vector_x, uint k_base) {
        // Strategy decides: we load 2 chunks of 4 elements (8 elements total) for unroll
        uint weights_per_block = quant.weights_per_block;
        uint block_in_group = quant.block_in_group;
        uint sub_offset = quant.sub_offset;
        
        uint k_idx_0 = k_base + block_in_group * weights_per_block + sub_offset;
        uint step = 4u * weights_per_block;
        uint k_idx_1 = k_idx_0 + step;
        
        // Delegate loading logic (and gamma application) to Quant
        xv0 = quant.load_input(vector_x, k_idx_0);
        xv1 = quant.load_input(vector_x, k_idx_1);
        
        block_idx_0 = (k_base / weights_per_block) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
    }
    
    void load_x_safe(const device half *vector_x, uint k_base, uint K) {
        uint weights_per_block = quant.weights_per_block;
        uint block_in_group = quant.block_in_group;
        uint sub_offset = quant.sub_offset;
        
        uint k_idx_0 = k_base + block_in_group * weights_per_block + sub_offset;
        uint step = 4u * weights_per_block;
        uint k_idx_1 = k_idx_0 + step;

        xv0 = quant.load_input_safe(vector_x, k_idx_0, K);
        xv1 = quant.load_input_safe(vector_x, k_idx_1, K);

        block_idx_0 = (k_base / weights_per_block) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
        total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    }
    
    float compute_dot(uint h, bool fast_mode) {
        float partial = 0.0f;
        
        if (fast_mode || block_idx_0 < total_blocks) {
             partial += quant.dot_part(h, xv0, false);
        }
        if (fast_mode || block_idx_1 < total_blocks) {
             // In canonical policy, block_idx_1 corresponds to the 'second block' in dot_part logic
             // Effectively: offset = 4 * stride_w
             partial += quant.dot_part(h, xv1, true);
        }
        
        // Reduction
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);
        
        uint sub_lane = quant.sub_offset / 4u;
        if (sub_lane == 0u) return partial;
        return 0.0f;
    }
    
    template<uint HEADS>
    void advance_pointers(uint k_step) {
        quant.template advance<HEADS>(k_step);
    }
};

#endif // METALLIC_SIMD_GEMV_CANONICAL_METAL
