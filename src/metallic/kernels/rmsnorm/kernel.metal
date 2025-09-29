#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-6f;

kernel void rmsnorm_kernel_f32(device float* input [[buffer(0)]],
                               device float* output [[buffer(1)]],
                               device float* gamma [[buffer(2)]],
                               constant uint& feature_dim [[buffer(3)]],
                               constant uint& total_elements [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) {
        return;
    }

    uint feature_idx = gid % feature_dim;
    uint row_idx = gid / feature_dim;

    float sum_sq = 0.0f;
    float feature_dim_acc = static_cast<float>(feature_dim);
    for (uint f = 0; f < feature_dim; ++f) {
        float v = input[row_idx * feature_dim + f];
        sum_sq += v * v;
    }

    float rms = sqrt(sum_sq / feature_dim_acc + EPS);
    float x = input[gid];
    float gamma_val = gamma[feature_idx];
    output[gid] = (x / rms) * gamma_val;
}

kernel void rmsnorm_kernel_f16(device half* input [[buffer(0)]],
                               device half* output [[buffer(1)]],
                               device half* gamma [[buffer(2)]],
                               constant uint& feature_dim [[buffer(3)]],
                               constant uint& total_elements [[buffer(4)]],
                               uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) {
        return;
    }

    uint feature_idx = gid % feature_dim;
    uint row_idx = gid / feature_dim;

    float sum_sq = 0.0f;
    float feature_dim_acc = static_cast<float>(feature_dim);
    for (uint f = 0; f < feature_dim; ++f) {
        float v = static_cast<float>(input[row_idx * feature_dim + f]);
        sum_sq += v * v;
    }

    float rms = sqrt(sum_sq / feature_dim_acc + EPS);
    float x = static_cast<float>(input[gid]);
    float gamma_val = static_cast<float>(gamma[feature_idx]);
    output[gid] = static_cast<half>((x / rms) * gamma_val);
}

kernel void rmsnorm_kernel_bf16(device bfloat* input [[buffer(0)]],
                                device bfloat* output [[buffer(1)]],
                                device bfloat* gamma [[buffer(2)]],
                                constant uint& feature_dim [[buffer(3)]],
                                constant uint& total_elements [[buffer(4)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) {
        return;
    }

    uint feature_idx = gid % feature_dim;
    uint row_idx = gid / feature_dim;

    float sum_sq = 0.0f;
    float feature_dim_acc = static_cast<float>(feature_dim);
    for (uint f = 0; f < feature_dim; ++f) {
        float v = static_cast<float>(input[row_idx * feature_dim + f]);
        sum_sq += v * v;
    }

    float rms = sqrt(sum_sq / feature_dim_acc + EPS);
    float x = static_cast<float>(input[gid]);
    float gamma_val = static_cast<float>(gamma[feature_idx]);
    output[gid] = static_cast<bfloat>((x / rms) * gamma_val);
}
