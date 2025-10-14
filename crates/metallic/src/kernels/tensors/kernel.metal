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

#define FOR_EACH_FLOAT_TYPE(OP) \
    OP(float, float, f32) \
    OP(half, float, f16)

#define DEFINE_RANDOM_UNIFORM_KERNEL(SCALAR, ACCUM, SUFFIX) \
kernel void random_uniform_##SUFFIX( \
    device SCALAR *output_buffer [[buffer(0)]], \
    constant uint &seed [[buffer(1)]], \
    constant float &minv [[buffer(2)]], \
    constant float &scale [[buffer(3)]], \
    uint thread_id [[thread_position_in_grid]] \
) { \
    uint2 random_seed = uint2(thread_id, seed); \
    ACCUM u = static_cast<ACCUM>(hash(random_seed)); \
    ACCUM value = static_cast<ACCUM>(minv) + u * static_cast<ACCUM>(scale); \
    output_buffer[thread_id] = static_cast<SCALAR>(value); \
}

#define DEFINE_ARANGE_KERNEL(SCALAR, SUFFIX) \
kernel void arange_kernel_##SUFFIX( \
    device SCALAR *output_buffer [[buffer(0)]], \
    uint thread_id [[thread_position_in_grid]] \
) { \
    output_buffer[thread_id] = static_cast<SCALAR>(thread_id); \
}

#define DEFINE_ONES_KERNEL(SCALAR, SUFFIX) \
kernel void ones_kernel_##SUFFIX( \
    device SCALAR *output_buffer [[buffer(0)]], \
    constant uint &total_elements [[buffer(1)]], \
    uint thread_id [[thread_position_in_grid]] \
) { \
    uint base = thread_id * 4u; \
    for (uint i = 0u; i < 4u; ++i) { \
        uint idx = base + i; \
        if (idx < total_elements) { \
            output_buffer[idx] = static_cast<SCALAR>(1.0f); \
        } \
    } \
}

#define INSTANTIATE_TENSOR_KERNELS(SCALAR, ACCUM, SUFFIX) \
DEFINE_RANDOM_UNIFORM_KERNEL(SCALAR, ACCUM, SUFFIX) \
DEFINE_ARANGE_KERNEL(SCALAR, SUFFIX) \
DEFINE_ONES_KERNEL(SCALAR, SUFFIX)

FOR_EACH_FLOAT_TYPE(INSTANTIATE_TENSOR_KERNELS)

// Explicit NOOP kernels (avoid macro expansion issues)
kernel void noop_kernel_f32(
    uint thread_id [[thread_position_in_grid]]
) {
    // Intentionally empty: does no work.
}

kernel void noop_kernel_f16(
    uint thread_id [[thread_position_in_grid]]
) {
    // Intentionally empty: does no work.
}

#undef INSTANTIATE_TENSOR_KERNELS
#undef DEFINE_ONES_KERNEL
#undef DEFINE_ARANGE_KERNEL
#undef DEFINE_RANDOM_UNIFORM_KERNEL
#undef FOR_EACH_FLOAT_TYPE
