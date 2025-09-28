#include <metal_stdlib>
using namespace metal;

kernel void repeat_kv_heads_kernel(device const float* input [[buffer(0)]],
                                   device float* output [[buffer(1)]],
                                   constant uint& group_size [[buffer(2)]],
                                   constant uint& batch [[buffer(3)]],
                                   constant uint& n_kv_heads [[buffer(4)]],
                                   constant uint& n_heads [[buffer(5)]],
                                   constant uint& seq [[buffer(6)]],
                                   constant uint& head_dim [[buffer(7)]],
                                   constant uint& total_elements [[buffer(8)]],
                                   uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) {
        return;
    }

    uint dim_idx = gid % head_dim;
    uint tmp = gid / head_dim;
    uint seq_idx = tmp % seq;
    uint batch_head_idx = tmp / seq;

    uint b = batch_head_idx / n_heads;
    uint h = batch_head_idx % n_heads;
    uint kv_head = h / group_size;

    uint input_batch_head = b * n_kv_heads + kv_head;
    uint total_batch_heads = batch * n_kv_heads;
    uint input_index = ((seq_idx * total_batch_heads) + input_batch_head) * head_dim + dim_idx;

    output[gid] = input[input_index];
}

kernel void repeat_kv_heads_step_kernel(device const float* input [[buffer(0)]],
                                        device float* output [[buffer(1)]],
                                        constant uint& step [[buffer(2)]],
                                        constant uint& group_size [[buffer(3)]],
                                        constant uint& n_kv_heads [[buffer(4)]],
                                        constant uint& n_heads [[buffer(5)]],
                                        constant uint& total_batch_kv_heads [[buffer(6)]],
                                        constant uint& total_batch_heads [[buffer(7)]],
                                        constant uint& seq_len [[buffer(8)]],
                                        constant uint& head_dim [[buffer(9)]],
                                        uint gid [[thread_position_in_grid]]) {
    uint total_elements = total_batch_heads * head_dim;
    if (gid >= total_elements) {
        return;
    }

    uint dim_idx = gid % head_dim;
    uint batch_head_idx = gid / head_dim;

    uint b = batch_head_idx / n_heads;
    uint h = batch_head_idx % n_heads;
    uint kv_head = h / group_size;

    uint input_batch_head = b * n_kv_heads + kv_head;

    uint input_index = ((step * total_batch_kv_heads) + input_batch_head) * head_dim + dim_idx;
    uint output_index = ((batch_head_idx * seq_len) + step) * head_dim + dim_idx;

    output[output_index] = input[input_index];
}
