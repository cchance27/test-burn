#include <metal_stdlib>
using namespace metal;

// Block-softmax kernel for very long sequences using segmented reductions
// Each threadgroup handles a segment of the sequence, then a final reduction
// across segments computes the global max/sum for proper normalization

kernel void block_softmax_f32(
    device const float* input [[buffer(0)]],  // [batch, seq_q, seq_k]
    device float* output [[buffer(1)]],      // [batch, seq_q, seq_k]
    constant uint& batch [[buffer(2)]],
    constant uint& seq_q [[buffer(3)]],
    constant uint& seq_k [[buffer(4)]],
    constant uint& segment_size [[buffer(5)]],
    constant uint& causal [[buffer(6)]],
    constant uint& query_offset [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint SEGMENT_THREADS = 256; // Fixed threadgroup size for block softmax
    const uint MAX_SEGMENTS_PER_TG = 4; // Maximum segments per threadgroup

    // Calculate which row and segment this threadgroup handles
    uint row_idx = tgid.x;
    uint segment_idx = tgid.y;

    if (row_idx >= batch * seq_q) return;

    // Calculate segment boundaries
    uint segment_start = segment_idx * segment_size;
    uint segment_end = min(segment_start + segment_size, seq_k);

    if (segment_start >= seq_k) return;

    uint actual_segment_size = segment_end - segment_start;

    // Calculate query position for causal masking
    uint query_pos = query_offset + (row_idx % seq_q);

    // Shared memory for local max/sum per segment
    threadgroup float local_max_vals[MAX_SEGMENTS_PER_TG];
    threadgroup float local_sum_vals[MAX_SEGMENTS_PER_TG];

    // Each thread processes elements in the segment
    float local_max = -INFINITY;
    float local_sum = 0.0;

    // Process the segment assigned to this thread
    // First pass: find local max with causal masking
    for (uint i = tid; i < actual_segment_size; i += SEGMENT_THREADS) {
        uint col_idx = segment_start + i;
        if (col_idx < seq_k) {
            uint input_idx = row_idx * seq_k + col_idx;
            float val = input[input_idx];

            // Apply causal mask if needed
            if (causal != 0u && col_idx > query_pos) {
                val = -INFINITY; // Mask out future tokens
            }

            // Update local max (only if not masked to -INFINITY)
            if (val > local_max) {
                local_max = val;
            }
        }
    }

    // Reduce local max across threads in threadgroup
    threadgroup float shared_max[256];
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find threadgroup max
    for (uint stride = SEGMENT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float threadgroup_max = shared_max[0];

    // Second pass: compute exponentials and local sum
    float local_sum_exp = 0.0f;
    for (uint i = tid; i < actual_segment_size; i += SEGMENT_THREADS) {
        uint col_idx = segment_start + i;
        if (col_idx < seq_k) {
            uint input_idx = row_idx * seq_k + col_idx;
            float val = input[input_idx];

            // Only contribute to sum if not causally masked
            if (causal == 0u || col_idx <= query_pos) {
                // Calculate exp(val - threadgroup_max) 
                float diff = val - threadgroup_max;
                float exp_val = (val != -INFINITY) ? exp(diff) : 0.0f;
                local_sum_exp += exp_val;
            }
            // Causally masked values contribute 0 to the sum
        }
    }

    // Reduce local sum across threads in threadgroup
    threadgroup float shared_sum[256];
    shared_sum[tid] = local_sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find threadgroup sum
    for (uint stride = SEGMENT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float final_sum = shared_sum[0];

    // Write results to threadgroup memory for final reduction across segments
    if (tid == 0) {
        local_max_vals[segment_idx] = threadgroup_max;
        local_sum_vals[segment_idx] = final_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Wait for the final reductions to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across segments (only thread 0 does this) - write global values
    if (tid == 0 && segment_idx == 0) {
        float final_global_max = -INFINITY;
        float final_global_sum = 0.0f;

        // Find global max across all segments of this row
        uint num_segments = (seq_k + segment_size - 1) / segment_size; // Ceiling division
        for (uint s = 0; s < num_segments && s < MAX_SEGMENTS_PER_TG; ++s) {
            if (local_max_vals[s] > final_global_max) {
                final_global_max = local_max_vals[s];
            }
        }

        // Compute global sum by adjusting each segment's sum to the global max
        for (uint s = 0; s < num_segments && s < MAX_SEGMENTS_PER_TG; ++s) {
            float segment_max = local_max_vals[s];
            float segment_sum = local_sum_vals[s];
            if (segment_sum > 0.0f) {
                // Adjust sum from segment_max to global_max
                float adjusted_contribution = segment_sum * exp(segment_max - final_global_max);
                final_global_sum += adjusted_contribution;
            }
        }

        // Store the computed global values in the first two positions for all threads to use
        local_max_vals[0] = final_global_max;  // global max
        local_sum_vals[0] = final_global_sum;  // global sum
    }

    // Wait for global computation to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Get the global values computed by thread 0 of segment 0
    float final_global_max = local_max_vals[0];
    float final_global_sum = local_sum_vals[0];

    // Final normalization pass - compute final softmax values
    for (uint i = tid; i < actual_segment_size; i += SEGMENT_THREADS) {
        uint col_idx = segment_start + i;
        if (col_idx < seq_k) {
            uint input_idx = row_idx * seq_k + col_idx;
            float val = input[input_idx];

            if (causal != 0u && col_idx > query_pos) {
                // Causal masking: future positions get 0 probability
                output[input_idx] = 0.0f;
            } else {
                // Apply softmax normalization using the final global values
                float exp_val = (val != -INFINITY) ? exp(val - final_global_max) : 0.0f;
                float normalized_val = (final_global_sum > 0.0f) ? (exp_val / final_global_sum) : 0.0f;
                output[input_idx] = normalized_val;
            }
        }
    }
}

kernel void block_softmax_f16(
    device const half* input [[buffer(0)]],  // [batch, seq_q, seq_k]
    device half* output [[buffer(1)]],      // [batch, seq_q, seq_k]
    constant uint& batch [[buffer(2)]],
    constant uint& seq_q [[buffer(3)]],
    constant uint& seq_k [[buffer(4)]],
    constant uint& segment_size [[buffer(5)]],
    constant uint& causal [[buffer(6)]],
    constant uint& query_offset [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    const uint SEGMENT_THREADS = 256; // Fixed threadgroup size for block softmax
    const uint MAX_SEGMENTS_PER_TG = 4; // Maximum segments per threadgroup

    // Calculate which row and segment this threadgroup handles
    uint row_idx = tgid.x;
    uint segment_idx = tgid.y;

    if (row_idx >= batch * seq_q) return;

    // Calculate segment boundaries
    uint segment_start = segment_idx * segment_size;
    uint segment_end = min(segment_start + segment_size, seq_k);

    if (segment_start >= seq_k) return;

    uint actual_segment_size = segment_end - segment_start;

    // Calculate query position for causal masking
    uint query_pos = query_offset + (row_idx % seq_q);

    // Shared memory for local max/sum per segment
    threadgroup float local_max_vals[MAX_SEGMENTS_PER_TG];
    threadgroup float local_sum_vals[MAX_SEGMENTS_PER_TG];

    // Each thread processes elements in the segment
    float local_max = -INFINITY;
    float local_sum = 0.0;

    // Process the segment assigned to this thread
    // First pass: find local max with causal masking
    for (uint i = tid; i < actual_segment_size; i += SEGMENT_THREADS) {
        uint col_idx = segment_start + i;
        if (col_idx < seq_k) {
            uint input_idx = row_idx * seq_k + col_idx;
            float val = (float)input[input_idx];

            // Apply causal mask if needed
            if (causal != 0u && col_idx > query_pos) {
                val = -INFINITY; // Mask out future tokens
            }

            // Update local max (only if not masked to -INFINITY)
            if (val > local_max) {
                local_max = val;
            }
        }
    }

    // Reduce local max across threads in threadgroup
    threadgroup float shared_max[256];
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find threadgroup max
    for (uint stride = SEGMENT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float threadgroup_max = shared_max[0];

    // Second pass: compute exponentials and local sum
    float local_sum_exp = 0.0f;
    for (uint i = tid; i < actual_segment_size; i += SEGMENT_THREADS) {
        uint col_idx = segment_start + i;
        if (col_idx < seq_k) {
            uint input_idx = row_idx * seq_k + col_idx;
            float val = (float)input[input_idx];

            // Only contribute to sum if not causally masked
            if (causal == 0u || col_idx <= query_pos) {
                // Calculate exp(val - threadgroup_max) 
                float diff = val - threadgroup_max;
                float exp_val = (val != -INFINITY) ? exp(diff) : 0.0f;
                local_sum_exp += exp_val;
            }
            // Causally masked values contribute 0 to the sum
        }
    }

    // Reduce local sum across threads in threadgroup
    threadgroup float shared_sum[256];
    shared_sum[tid] = local_sum_exp;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction to find threadgroup sum
    for (uint stride = SEGMENT_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float final_sum = shared_sum[0];

    // Write results to threadgroup memory for final reduction across segments
    if (tid == 0) {
        local_max_vals[segment_idx] = threadgroup_max;
        local_sum_vals[segment_idx] = final_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Wait for the final reductions to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across segments (only thread 0 does this) - write global values
    if (tid == 0 && segment_idx == 0) {
        float final_global_max = -INFINITY;
        float final_global_sum = 0.0f;

        // Find global max across all segments of this row
        uint num_segments = (seq_k + segment_size - 1) / segment_size; // Ceiling division
        for (uint s = 0; s < num_segments && s < MAX_SEGMENTS_PER_TG; ++s) {
            if (local_max_vals[s] > final_global_max) {
                final_global_max = local_max_vals[s];
            }
        }

        // Compute global sum by adjusting each segment's sum to the global max
        for (uint s = 0; s < num_segments && s < MAX_SEGMENTS_PER_TG; ++s) {
            float segment_max = local_max_vals[s];
            float segment_sum = local_sum_vals[s];
            if (segment_sum > 0.0f) {
                // Adjust sum from segment_max to global_max
                float adjusted_contribution = segment_sum * exp(segment_max - final_global_max);
                final_global_sum += adjusted_contribution;
            }
        }

        // Store the computed global values in the first two positions for all threads to use
        local_max_vals[0] = final_global_max;  // global max
        local_sum_vals[0] = final_global_sum;  // global sum
    }

    // Wait for global computation to complete
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Get the global values computed by thread 0 of segment 0
    float final_global_max = local_max_vals[0];
    float final_global_sum = local_sum_vals[0];

    // Final normalization pass - compute final softmax values
    for (uint i = tid; i < actual_segment_size; i += SEGMENT_THREADS) {
        uint col_idx = segment_start + i;
        if (col_idx < seq_k) {
            uint input_idx = row_idx * seq_k + col_idx;
            float val = (float)input[input_idx];

            if (causal != 0u && col_idx > query_pos) {
                // Causal masking: future positions get 0 probability
                output[input_idx] = 0.0h;
            } else {
                // Apply softmax normalization using the final global values
                float exp_val = (val != -INFINITY) ? exp(val - final_global_max) : 0.0f;
                float normalized_val = (final_global_sum > 0.0f) ? (exp_val / final_global_sum) : 0.0f;
                output[input_idx] = (half)normalized_val;
            }
        }
    }
}