#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-6f;

kernel void fused_rmsnorm_qkv_projection_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    constant uint& feature_dim [[buffer(5)]],
    constant uint& total_out_dim [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) {
        return;
    }

    uint row_idx = gid;
    uint feature_offset = row_idx * feature_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < feature_dim; ++i) {
        float v = input[feature_offset + i];
        sum_sq += v * v;
    }

    float inv_rms = rsqrt(sum_sq / static_cast<float>(feature_dim) + EPS);

    for (uint out_idx = 0; out_idx < total_out_dim; ++out_idx) {
        float acc = 0.0f;
        uint weight_offset = out_idx;
        for (uint i = 0; i < feature_dim; ++i) {
            float x = input[feature_offset + i];
            float gamma_val = gamma[i];
            float normed = x * inv_rms * gamma_val;
            float w = weight[i * total_out_dim + weight_offset];
            acc += normed * w;
        }
        output[row_idx * total_out_dim + out_idx] = acc + bias[out_idx];
    }
}

kernel void fused_rmsnorm_qkv_projection_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    device const half* gamma [[buffer(2)]],
    device const half* weight [[buffer(3)]],
    device const half* bias [[buffer(4)]],
    constant uint& feature_dim [[buffer(5)]],
    constant uint& total_out_dim [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= rows) {
        return;
    }

    uint row_idx = gid;
    uint feature_offset = row_idx * feature_dim;
    float sum_sq = 0.0f;
    for (uint i = 0; i < feature_dim; ++i) {
        float v = static_cast<float>(input[feature_offset + i]);
        sum_sq += v * v;
    }

    float inv_rms = rsqrt(sum_sq / static_cast<float>(feature_dim) + EPS);

    for (uint out_idx = 0; out_idx < total_out_dim; ++out_idx) {
        float acc = 0.0f;
        uint weight_offset = out_idx;
        for (uint i = 0; i < feature_dim; ++i) {
            float x = static_cast<float>(input[feature_offset + i]);
            float gamma_val = static_cast<float>(gamma[i]);
            float scaled = x * inv_rms * gamma_val;
            // Match the standalone RMSNorm kernel by quantizing the scaled value to half
            // precision before it feeds the projection matmuls. This mirrors the previous
            // two-kernel pipeline where the normalized activations were materialized as
            // F16 tensors prior to the fused QKV matmul, keeping test parity.
            half scaled_half = static_cast<half>(scaled);
            float normed = static_cast<float>(scaled_half);
            float w = static_cast<float>(weight[i * total_out_dim + weight_offset]);
            acc += normed * w;
        }
        half matmul_half = static_cast<half>(acc);
        float matmul_val = static_cast<float>(matmul_half);
        float biased = matmul_val + static_cast<float>(bias[out_idx]);
        output[row_idx * total_out_dim + out_idx] = static_cast<half>(biased);
    }
}
