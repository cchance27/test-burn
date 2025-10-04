// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

using namespace metal;

struct GemvParams {
    uint K;
    uint N;
};

#define THREADGROUP_WIDTH 256u
#define TILE_N THREADGROUP_WIDTH
#define TILE_K 256u

template <typename MatrixT, typename VectorT, typename Scalar>
inline void gemv_kernel(
    const device MatrixT *matrix_a,
    const device VectorT *vector_x,
    device Scalar *result_y,
    const constant GemvParams *params,
    uint3 gid,
    uint3 lid,
    threadgroup float *shared_x_tile) {

    const uint N = params->N;
    const uint K = params->K;

    const uint out_idx = gid.x * TILE_N + lid.x;
    const bool is_active = out_idx < N;

    float sum = 0.0f;

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        // Stage a tile of the input vector in shared memory so threads reuse it.
        for (uint local = lid.x; local < TILE_K; local += THREADGROUP_WIDTH) {
            const uint global_k = tile_base + local;
            shared_x_tile[local] = global_k < K ? static_cast<float>(vector_x[global_k]) : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_active) {
            const uint tile_limit = min(TILE_K, K - tile_base);
            const device MatrixT *matrix_ptr = matrix_a + tile_base * N + out_idx;

            uint local = 0;
            for (; local + 3 < tile_limit; local += 4) {
                sum = fma(shared_x_tile[local], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
                sum = fma(shared_x_tile[local + 1], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
                sum = fma(shared_x_tile[local + 2], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
                sum = fma(shared_x_tile[local + 3], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
            }

            for (; local < tile_limit; ++local) {
                sum = fma(shared_x_tile[local], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (is_active) {
        result_y[out_idx] = static_cast<Scalar>(sum);
    }
}

[[kernel]] void gemv_f32(
    const device float *matrix_a [[buffer(0)]],
    const device float *vector_x [[buffer(1)]],
    device float *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float x_tile[TILE_K];
    gemv_kernel<float, float, float>(matrix_a, vector_x, result_y, params, gid, lid, x_tile);
}

[[kernel]] void gemv_f16(
    const device half *matrix_a [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float x_tile[TILE_K];
    gemv_kernel<half, half, half>(matrix_a, vector_x, result_y, params, gid, lid, x_tile);
}
