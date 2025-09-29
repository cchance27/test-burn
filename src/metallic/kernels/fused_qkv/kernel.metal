#include <metal_stdlib>
using namespace metal;

kernel void fused_qkv_bias_split(
    device const float* qkv_linear [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* q_out [[buffer(2)]],
    device float* k_out [[buffer(3)]],
    device float* v_out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& q_cols [[buffer(6)]],
    constant uint& kv_cols [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
    uint total_cols = q_cols + 2 * kv_cols;
    uint total = rows * total_cols;
    if (gid >= total) {
        return;
    }

    uint row = gid / total_cols;
    uint col = gid % total_cols;

    float value = qkv_linear[gid] + bias[col];

    if (col < q_cols) {
        uint out_idx = row * q_cols + col;
        q_out[out_idx] = value;
    } else if (col < q_cols + kv_cols) {
        uint offset = col - q_cols;
        uint out_idx = row * kv_cols + offset;
        k_out[out_idx] = value;
    } else {
        uint offset = col - q_cols - kv_cols;
        uint out_idx = row * kv_cols + offset;
        v_out[out_idx] = value;
    }
}
