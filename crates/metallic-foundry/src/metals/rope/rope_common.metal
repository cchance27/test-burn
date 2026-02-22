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

constant uint ROPE_MAX_HEAD_DIM_VEC = 64; // 256 / 4

ALWAYS_INLINE void run_rope_decode_stage(
    threadgroup half4* q_shared,
    const device half* q_ptr,
    const device half* cos_buf,
    const device half* sin_buf,
    constant RopeParams& params_rope,
    uint lane_id
) {
    const uint head_dim = params_rope.dim;
    const uint rope_vec_dim = head_dim / 4u;

    if (lane_id < rope_vec_dim) {
        const uint half_vec = rope_vec_dim / 2u;
        if (lane_id < half_vec) {
            const uint pos = params_rope.position_offset;

            const device half4* q_ptr_vec = (const device half4*)q_ptr;
            const device half4* cos_buf_vec = (const device half4*)cos_buf;
            const device half4* sin_buf_vec = (const device half4*)sin_buf;

            half4 x_low = q_ptr_vec[lane_id];
            half4 x_high = q_ptr_vec[lane_id + half_vec];
            half4 cos_v = cos_buf_vec[pos * half_vec + lane_id];
            half4 sin_v = sin_buf_vec[pos * half_vec + lane_id];

            float4 x_l = (float4)x_low;
            float4 x_h = (float4)x_high;
            float4 c_v = (float4)cos_v;
            float4 s_v = (float4)sin_v;

            q_shared[lane_id] = (half4)(x_l * c_v - x_h * s_v);
            q_shared[lane_id + half_vec] = (half4)(x_l * s_v + x_h * c_v);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#endif // ROPE_COMMON_METAL
