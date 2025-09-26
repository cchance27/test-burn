#include <metal_stdlib>
using namespace metal;

// More numerically stable GELU implementation
kernel void gelu_kernel(device float* input [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       constant uint& total_elements [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;

    float x = input[gid];

    // For very large positive values, GELU ≈ x
    if (x > 5.0f) {
        output[gid] = x;
        return;
    }

    // For very large negative values, GELU ≈ 0
    if (x < -5.0f) {
        output[gid] = 0.0f;
        return;
    }

    // Use more stable computation to avoid overflow
    // GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    // Approximation using tanh: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    // Compute x^3 more safely to avoid overflow
    float abs_x = fabs(x);
    float x3;
    if (abs_x < 1e-10f) {
        x3 = 0.0f;
    } else if (abs_x > 10.0f) {
        // For large |x|, use log-space computation
        x3 = copysign(exp(3.0f * log(abs_x)), x);
    } else {
        x3 = x * x * x;
    }

    float inner = sqrt(2.0f / 3.141592653589793f) * (x + 0.044715f * x3);
    float tanh_inner = tanh(inner);
    output[gid] = 0.5f * x * (1.0f + tanh_inner);
}