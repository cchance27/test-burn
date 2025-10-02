#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-6f;
constant uint THREADGROUP_SIZE = 256;

kernel void fused_rmsnorm_qkv_projection_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* weight [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    constant uint& feature_dim [[buffer(5)]],
    constant uint& total_out_dim [[buffer(6)]],
    constant uint& rows [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_dim [[threads_per_threadgroup]]) {
    uint row_idx = tg_pos.x;
    if (row_idx >= rows) {
        return;
    }

    uint threadgroup_size = tg_dim.x;
    uint feature_offset = row_idx * feature_dim;
    uint output_offset = row_idx * total_out_dim;

    threadgroup float partial_sums[THREADGROUP_SIZE];
    threadgroup float shared_inv_rms;

    float sum_sq = 0.0f;
    for (uint i = tid; i < feature_dim; i += threadgroup_size) {
        float v = input[feature_offset + i];
        sum_sq += v * v;
    }

    partial_sums[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean_sq = partial_sums[0] / static_cast<float>(feature_dim);
        shared_inv_rms = rsqrt(mean_sq + EPS);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_inv_rms;

    for (uint out_idx = tid; out_idx < total_out_dim; out_idx += threadgroup_size) {
        float acc = 0.0f;
        for (uint i = 0; i < feature_dim; ++i) {
            float normed = input[feature_offset + i] * inv_rms * gamma[i];
            float w = weight[i * total_out_dim + out_idx];
            acc += normed * w;
        }
        output[output_offset + out_idx] = acc + bias[out_idx];
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
    uint tid [[thread_position_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_dim [[threads_per_threadgroup]]) {
    uint row_idx = tg_pos.x;
    if (row_idx >= rows) {
        return;
    }

    uint threadgroup_size = tg_dim.x;
    uint feature_offset = row_idx * feature_dim;
    uint output_offset = row_idx * total_out_dim;

    threadgroup float partial_sums[THREADGROUP_SIZE];
    threadgroup float shared_inv_rms;

    float sum_sq = 0.0f;
    for (uint i = tid; i < feature_dim; i += threadgroup_size) {
        float v = static_cast<float>(input[feature_offset + i]);
        sum_sq += v * v;
    }

    partial_sums[tid] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threadgroup_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean_sq = partial_sums[0] / static_cast<float>(feature_dim);
        shared_inv_rms = rsqrt(mean_sq + EPS);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared_inv_rms;

    for (uint out_idx = tid; out_idx < total_out_dim; out_idx += threadgroup_size) {
        float acc = 0.0f;
        for (uint i = 0; i < feature_dim; ++i) {
            float x = static_cast<float>(input[feature_offset + i]);
            float gamma_val = static_cast<float>(gamma[i]);
            float scaled = x * inv_rms * gamma_val;
            half scaled_half = static_cast<half>(scaled);
            float normed = static_cast<float>(scaled_half);
            float w = static_cast<float>(weight[i * total_out_dim + out_idx]);
            acc += normed * w;
        }
        half matmul_half = static_cast<half>(acc);
        float matmul_val = static_cast<float>(matmul_half);
        float biased = matmul_val + static_cast<float>(bias[out_idx]);
        output[output_offset + out_idx] = static_cast<half>(biased);
    }
}
