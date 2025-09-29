#include <metal_stdlib>

using namespace metal;

// A simple hashing-based pseudo-random number generator for Metal shaders.
// It takes a 2D seed and produces a float in [0, 1).
float hash(uint2 seed) {
    // Constants for hashing
    const uint k1 = 0x456789abu;
    const uint k2 = 0x89abcdefu;
    const uint k3 = 0xabcdef01u;

    // Scramble the seed
    uint n = seed.x * k1 + seed.y * k2;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    n = (n >> 13) ^ n;

    // Convert to a float in [0, 1)
    return float(n & 0x0fffffffu) / float(0x10000000u);
}

// Kernel to fill a buffer with uniform random numbers in [min, min+scale).
// buffer(0) - device float *output_buffer
// buffer(1) - constant uint &seed
// buffer(2) - constant float &minv
// buffer(3) - constant float &scale
kernel void random_uniform(
    device float *output_buffer [[buffer(0)]],
    constant uint &seed [[buffer(1)]],
    constant float &minv [[buffer(2)]],
    constant float &scale [[buffer(3)]],
    uint thread_id [[thread_position_in_grid]]
) {
    uint2 random_seed = uint2(thread_id, seed);
    float u = hash(random_seed);
    output_buffer[thread_id] = minv + u * scale;
}

// Kernel to fill a buffer with values [0, 1, 2, 3, ...].
kernel void arange_kernel(
    device float *output_buffer [[buffer(0)]],
    uint thread_id [[thread_position_in_grid]]
) {
    output_buffer[thread_id] = thread_id;
}

// Vectorized ones kernel: each thread writes up to 4 elements.
// buffer(0) - device float *output_buffer
// buffer(1) - constant uint &total_elements
kernel void ones_kernel(
    device float *output_buffer [[buffer(0)]],
    constant uint &total_elements [[buffer(1)]],
    uint thread_id [[thread_position_in_grid]]
) {
    uint base = thread_id * 4u;
    // Write up to 4 elements, guarding against bounds
    for (uint i = 0u; i < 4u; ++i) {
        uint idx = base + i;
        if (idx < total_elements) {
            output_buffer[idx] = 1.0;
        }
    }
}

kernel void convert_f16_to_f32(
    device const half *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &total_elements [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_elements) {
        return;
    }
    dst[gid] = float(src[gid]);
}

kernel void convert_bf16_to_f32(
    device const bfloat *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant uint &total_elements [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_elements) {
        return;
    }
    dst[gid] = float(src[gid]);
}