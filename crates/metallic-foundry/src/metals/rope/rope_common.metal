#ifndef ROPE_COMMON_METAL
#define ROPE_COMMON_METAL

#include <metal_stdlib>
using namespace metal;
#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

/// Apply RoPE rotation to a pair of values.
/// out_i = x_i * cos - x_j * sin
/// out_j = x_j * cos + x_i * sin
ALWAYS_INLINE void rope_rotate_half(
    thread float& out_i,
    thread float& out_j,
    float x_i,
    float x_j,
    float cos_v,
    float sin_v
) {
    out_i = x_i * cos_v - x_j * sin_v;
    out_j = x_j * cos_v + x_i * sin_v;
}

#endif // ROPE_COMMON_METAL
