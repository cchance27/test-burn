#include <metal_stdlib>
using namespace metal;

// RandomUniformParams struct is injected by Foundry via struct_defs()

// Simple hash-based PRNG
float hash(uint2 seed) {
    const uint k1 = 0x456789abu;
    const uint k2 = 0x89abcdefu;
    
    uint n = seed.x * k1 + seed.y * k2;
    n = (n << 13) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    n = (n >> 13) ^ n;
    
    return float(n & 0x0fffffffu) / float(0x10000000u);
}

/// Random Uniform kernel for half precision.
///
/// Fills output with random values in [min, min + scale).
kernel void random_uniform_kernel_f16(
    device half* output [[buffer(0)]],
    constant RandomUniformParams* params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    uint seed = params->seed;
    float minv = params->min_val;
    float scale = params->scale;
    
    uint2 random_seed = uint2(gid, seed);
    float u = hash(random_seed);
    float value = minv + u * scale;
    output[gid] = (half)value;
}
