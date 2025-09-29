#include <metal_stdlib>
using namespace metal;

constant float EPS = 1e-6f;

// RMSNorm: normalize by root-mean-square and apply per-feature scale (gamma)
kernel void rmsnorm_kernel(device float* input [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           device float* gamma [[buffer(2)]],
                           constant uint& feature_dim [[buffer(3)]],
                           constant uint& total_elements [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;

    uint feature_idx = gid % feature_dim;
    uint row_idx = gid / feature_dim;

    // Compute sum of squares for this row
    float sum_sq = 0.0f;
    for (uint f = 0; f < feature_dim; ++f) {
        float v = input[row_idx * feature_dim + f];
        sum_sq += v * v;
    }

    float rms = sqrt(sum_sq / float(feature_dim) + EPS);

    float x = input[gid];

    // Normalize by RMS and apply gamma scaling
    output[gid] = (x / rms) * gamma[feature_idx];
}