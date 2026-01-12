#include <metal_stdlib>
using namespace metal;

// Vec-softmax kernel using simdgroup reductions for short-to-medium sequences
// Each threadgroup processes multiple rows using vectorized operations and simdgroup reductions

kernel void vec_softmax_f32(
    device const float* input [[buffer(0)]],  // [batch * seq_q, seq_k]
    device float* output [[buffer(1)]],      // [batch * seq_q, seq_k]
    constant uint& seq_q [[buffer(2)]],
    constant uint& seq_k [[buffer(3)]],
    constant uint& causal_flag [[buffer(4)]],
    constant uint& query_offset [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    // Each threadgroup processes one row along Y
    uint row_idx = tg_pos.y;

    // Use 32 threads per row for simdgroup operations
    const uint THREADS_PER_ROW = tptg.x; // match legacy stride

    // Parallel reduction setup
    uint lane_id = tid3.x;
    uint stride = THREADS_PER_ROW;
    uint base = row_idx * seq_k;
    uint i_q = query_offset + (row_idx % seq_q);

    // Shared memory for reductions
    threadgroup float shared_data[256];
    threadgroup uint shared_indices[256];

    // Phase 1: parallel max reduction with index tracking
    float local_max = -INFINITY;
    uint local_max_index = 0u;
    for (uint c = lane_id; c < seq_k; c += stride) {
        float xv = input[base + c];
        if (causal_flag != 0u && c > i_q) {
            xv = -INFINITY;
        }
        if (xv > local_max) {
            local_max = xv;
            local_max_index = c;
        }
    }
    shared_data[lane_id] = local_max;
    shared_indices[lane_id] = local_max_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
        if (lane_id < offset) {
            float cand = shared_data[lane_id + offset];
            uint cand_idx = shared_indices[lane_id + offset];
            if (cand > shared_data[lane_id]) {
                shared_data[lane_id] = cand;
                shared_indices[lane_id] = cand_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    uint row_max_index = shared_indices[0];

    // Phase 2: write exp(x - max) to output and reduce sum in parallel
    float local_sum = 0.0f;
    for (uint c = lane_id; c < seq_k; c += stride) {
        float xv = input[base + c];
        if (causal_flag != 0u && c > i_q) {
            xv = -INFINITY;
        }
        float e = 0.0f;
        if (isinf(row_max) && row_max > 0.0f) {
            if (isinf(xv) && xv > 0.0f) {
                e = 1.0f;
            }
        } else if (xv != -INFINITY) {
            float diff = xv - row_max;
            if (diff < -80.0f) {
                e = 0.0f;
            } else if (diff > 80.0f) {
                e = exp(80.0f);
            } else {
                e = exp(diff);
            }
        }
        output[base + c] = e;
        local_sum += e;
    }
    shared_data[lane_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
        if (lane_id < offset) {
            shared_data[lane_id] += shared_data[lane_id + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_data[0];

    // Phase 3: normalize
    for (uint c = lane_id; c < seq_k; c += stride) {
        bool masked = (causal_flag != 0u && c > i_q);
        if (isnan(row_sum)) {
            output[base + c] = row_sum;
        } else if (row_sum > 0.0f && !isinf(row_sum)) {
            float e = output[base + c];
            output[base + c] = e / row_sum;
        } else {
            if (masked) {
                output[base + c] = 0.0f;
            } else if (c == row_max_index) {
                output[base + c] = 1.0f;
            } else {
                output[base + c] = 0.0f;
            }
        }
    }
}

kernel void vec_softmax_f16(
    device const half* input [[buffer(0)]],  // [batch * seq_q, seq_k]
    device half* output [[buffer(1)]],      // [batch * seq_q, seq_k]
    constant uint& seq_q [[buffer(2)]],
    constant uint& seq_k [[buffer(3)]],
    constant uint& causal_flag [[buffer(4)]],
    constant uint& query_offset [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    // Each threadgroup processes one row along Y
    uint row_idx = tg_pos.y;

    // Use 32 threads per row for simdgroup operations
    const uint THREADS_PER_ROW = tptg.x; // match legacy stride

    // Parallel reduction setup
    uint lane_id = tid3.x;
    uint stride = THREADS_PER_ROW;
    uint base = row_idx * seq_k;
    uint i_q = query_offset + (row_idx % seq_q);

    // Shared memory for reductions
    threadgroup float shared_data[256];
    threadgroup uint shared_indices[256];

    // Phase 1: parallel max reduction with index tracking
    float local_max = -INFINITY;
    uint local_max_index = 0u;
    for (uint c = lane_id; c < seq_k; c += stride) {
        float xv = (float)input[base + c];
        if (causal_flag != 0u && c > i_q) {
            xv = -INFINITY;
        }
        if (xv > local_max) {
            local_max = xv;
            local_max_index = c;
        }
    }
    shared_data[lane_id] = local_max;
    shared_indices[lane_id] = local_max_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
        if (lane_id < offset) {
            float cand = shared_data[lane_id + offset];
            uint cand_idx = shared_indices[lane_id + offset];
            if (cand > shared_data[lane_id]) {
                shared_data[lane_id] = cand;
                shared_indices[lane_id] = cand_idx;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_data[0];
    uint row_max_index = shared_indices[0];

    // Phase 2: write exp(x - max) to output and reduce sum in parallel
    float local_sum = 0.0f;
    for (uint c = lane_id; c < seq_k; c += stride) {
        float xv = (float)input[base + c];
        if (causal_flag != 0u && c > i_q) {
            xv = -INFINITY;
        }
        float e = 0.0f;
        if (isinf(row_max) && row_max > 0.0f) {
            if (isinf(xv) && xv > 0.0f) {
                e = 1.0f;
            }
        } else if (xv != -INFINITY) {
            float diff = xv - row_max;
            if (diff < -80.0f) {
                e = 0.0f;
            } else if (diff > 80.0f) {
                e = exp(80.0f);
            } else {
                e = exp(diff);
            }
        }
        output[base + c] = half(e);
        local_sum += e;
    }
    shared_data[lane_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = stride / 2u; offset > 0u; offset /= 2u) {
        if (lane_id < offset) {
            shared_data[lane_id] += shared_data[lane_id + offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_data[0];

    // Phase 3: normalize
    for (uint c = lane_id; c < seq_k; c += stride) {
        bool masked = (causal_flag != 0u && c > i_q);
        if (isnan(row_sum)) {
            output[base + c] = half(row_sum);
        } else if (row_sum > 0.0f && !isinf(row_sum)) {
            float e = (float)output[base + c];
            output[base + c] = half(e / row_sum);
        } else {
            if (masked) {
                output[base + c] = half(0.0f);
            } else if (c == row_max_index) {
                output[base + c] = half(1.0f);
            } else {
                output[base + c] = half(0.0f);
            }
        }
    }
}