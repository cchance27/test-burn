#include <metal_stdlib>
using namespace metal;

// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
kernel void silu_kernel(device float* input [[buffer(0)]],
                        device float* output [[buffer(1)]],
                        constant uint& total_elements [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;
    float x = input[gid];
    // For better numerical stability, use more precise handling near clamping thresholds
    if (x > 20.0f) {
        // For large x values, SiLU(x) ≈ x, but use a smoother transition to avoid discontinuities
        if (x > 80.0f) {
            output[gid] = x;  // Clamp to avoid overflow in exp calculation
        } else {
            float sig = 1.0f / (1.0f + exp(-x));  // This will be very close to 1
            output[gid] = x * sig;
        }
    } else if (x < -20.0f) {
        // For very negative x values, SiLU(x) ≈ 0, but calculate more precisely for values closer to threshold
        if (x < -80.0f) {
            output[gid] = 0.0f;  // Clamp to avoid underflow in exp calculation
        } else {
            float sig = 1.0f / (1.0f + exp(-x));  // This will be very close to 0
            output[gid] = x * sig;
        }
    } else {
        // Use standard SiLU calculation for values in the stable range
        float sig = 1.0f / (1.0f + exp(-x));
        output[gid] = x * sig;
    }
}