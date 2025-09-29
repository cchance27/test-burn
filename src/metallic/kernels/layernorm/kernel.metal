#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-5f;

kernel void layernorm_kernel(device float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            device float* gamma [[buffer(2)]],
                            device float* beta [[buffer(3)]],
                            constant uint& feature_dim [[buffer(4)]],
                            constant uint& total_elements [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;

    uint feature_idx = gid % feature_dim;
    uint row_idx = gid / feature_dim;

    // Compute mean and variance for this row
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (uint f = 0; f < feature_dim; ++f) {
        float val = input[row_idx * feature_dim + f];
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / float(feature_dim);
    // Use numerically stable formula: var = (sum_sq - mean * sum) / n
    float var = (sum_sq - mean * sum) / float(feature_dim);

    // Normalize
    float x = input[gid];
    float normalized = (x - mean) / sqrt(var + EPS);

    // Apply affine transformation
    output[gid] = normalized * gamma[feature_idx] + beta[feature_idx];
}