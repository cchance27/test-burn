#ifndef SOFTMAX_METAL_H
#define SOFTMAX_METAL_H

#include <metal_stdlib>
using namespace metal;

/// Helper to find the maximum value in a row partition.
inline float find_row_max(
    const device InputStorageT* matrix,
    uint row_idx,
    uint tid,
    uint seq_k,
    uint causal,
    uint query_offset
) {
    // Per-row causal mask index (matches Context kernel behavior)
    uint mask_pos = row_idx + query_offset;
    
    float local_max = -INFINITY;
    for (uint i = tid; i < seq_k; i += 256) {
        uint input_idx = row_idx * seq_k + i;
        float val = metallic_load_input(matrix, input_idx);
        
        if (causal != 0 && i > mask_pos) {
             val = -INFINITY;
        }
        if (val > local_max) {
             local_max = val;
        }
    }
    return local_max;
}

/// Helper to compute the sum of exponentials (val - max).
inline float compute_exp_sum(
    const device InputStorageT* matrix,
    float row_max,
    uint row_idx,
    uint tid,
    uint seq_k,
    uint causal,
    uint query_offset
) {
    // Per-row causal mask index (matches Context kernel behavior)
    uint mask_pos = row_idx + query_offset;
    
    float local_sum = 0.0f;
    for (uint i = tid; i < seq_k; i += 256) {
        uint input_idx = row_idx * seq_k + i;
        float val = metallic_load_input(matrix, input_idx);
        
        if (causal == 0 || i <= mask_pos) {
            float diff = val - row_max;
            local_sum += metal::fast::exp(diff);
        }
    }
    return local_sum;
}

/// Helper to normalize and write results.
inline void normalize_and_write(
    device OutputStorageT* output,
    const device InputStorageT* matrix,
    float row_max,
    float row_sum,
    uint row_idx,
    uint tid,
    uint seq_k,
    uint causal,
    uint query_offset
) {
    // Per-row causal mask index (matches Context kernel behavior)
    uint mask_pos = row_idx + query_offset;
    
    for (uint i = tid; i < seq_k; i += 256) {
        uint input_idx = row_idx * seq_k + i;
        float val = metallic_load_input(matrix, input_idx);
        
        if (causal != 0 && i > mask_pos) {
            metallic_store_output(output, input_idx, metallic_to_accum(0.0f));
        } else {
            float diff = val - row_max;
            float prob = metal::fast::exp(diff) / row_sum;
            metallic_store_output(output, input_idx, metallic_to_accum(prob));
        }
    }
}

#endif // SOFTMAX_METAL_H
