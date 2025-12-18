// Dense GEMV kernels

// Optimized SIMD-Parallel FP16 GEMV
// Ports the logic from Q8 (Vectorized 4-Block + 2x Unroll) to F16.
// Layout: Column-Major (standard).



// -----------------------------------------------------------------------------
// Small-N GEMV kernels for dense FP16 matmuls (N ∈ {1,2,4,8,16})
// -----------------------------------------------------------------------------

// Optimized N=8 GEMV kernel following GGML patterns
// Each thread handles one element of the output matrix C
kernel void gemv_n8_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 8
    device half*       C [[buffer(2)]], // M x 8
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 8u];
    gemv_f16_smalln_impl<8u, 8u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}

// N=4 GEMV kernel - optimized for small N=4 case
kernel void gemv_n4_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 4
    device half*       C [[buffer(2)]], // M x 4
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 4u];
    // This variant historically advanced rows along tgid.y; preserve that behavior (UseGridY=true).
    gemv_f16_smalln_impl<4u, 16u, GEMV_SMALLN_TILE_K, true>(A, B, C, M, K, tid, tgid, smemB);
}

// N=1 GEMV kernel - optimized for single column output
kernel void gemv_n1_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 1
    device half*       C [[buffer(2)]], // M x 1
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 1u];
    gemv_f16_smalln_impl<1u, 32u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}

// N=2 GEMV kernel - optimized for N=2 case
kernel void gemv_n2_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 2
    device half*       C [[buffer(2)]], // M x 2
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 2u];
    gemv_f16_smalln_impl<2u, 32u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}

// N=16 GEMV kernel - optimized for N=16 case
kernel void gemv_n16_f16(
    device const half* A [[buffer(0)]], // M x K
    device const half* B [[buffer(1)]], // K x 16
    device half*       C [[buffer(2)]], // M x 16
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    threadgroup half smemB[GEMV_SMALLN_TILE_K * 16u];
    gemv_f16_smalln_impl<16u, 4u, GEMV_SMALLN_TILE_K, false>(A, B, C, M, K, tid, tgid, smemB);
}


template <typename MatrixAccessor, typename VectorT, typename Scalar, bool HasBias>
inline void gemv_kernel(
    MatrixAccessor matrix_accessor,
    const device VectorT *vector_x,
    device Scalar *result_y,
    const constant GemvParams *params,
    const device Scalar *bias_ptr,
    uint3 gid,
    uint3 lid,
    threadgroup float *shared_x_tile) {

    const uint N = params->N;
    const uint K = params->K;

    const uint base_col = gid.x * TILE_N + lid.x * GEMV_COLS_PER_THREAD;
    float sum[GEMV_COLS_PER_THREAD];
    for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) { sum[c] = 0.0f; }

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        const uint tile_limit = min(TILE_K, K - tile_base);
        // Stage a tile of the input vector in shared memory so threads reuse it.
        if (lid.x < LOAD_LANES) {
            for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                const uint global_k = tile_base + local;
                shared_x_tile[local] = static_cast<float>(vector_x[global_k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Process up to GEMV_COLS_PER_THREAD columns per thread
        for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
            const uint col = base_col + c;
            if (col >= N) break;
            const device VectorT *matrix_base = matrix_accessor.matrix;
            const uint stride = matrix_accessor.stride;
            const device VectorT *matrix_ptr = matrix_base + tile_base * stride + col;
            uint local = 0u;
            for (; local + 3u < tile_limit; local += 4u) {
                sum[c] = fma(shared_x_tile[local + 0u], static_cast<float>(matrix_ptr[0]), sum[c]); matrix_ptr += stride;
                sum[c] = fma(shared_x_tile[local + 1u], static_cast<float>(matrix_ptr[0]), sum[c]); matrix_ptr += stride;
                sum[c] = fma(shared_x_tile[local + 2u], static_cast<float>(matrix_ptr[0]), sum[c]); matrix_ptr += stride;
                sum[c] = fma(shared_x_tile[local + 3u], static_cast<float>(matrix_ptr[0]), sum[c]); matrix_ptr += stride;
            }
            for (; local < tile_limit; ++local) {
                sum[c] = fma(shared_x_tile[local], static_cast<float>(matrix_ptr[0]), sum[c]);
                matrix_ptr += stride;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
        const uint col = base_col + c;
        if (col >= N) break;
        float bias_val = GemvBiasReader<HasBias>::template load<Scalar>(bias_ptr, col);
        result_y[col] = static_cast<Scalar>(sum[c] + bias_val);
    }
}

[[kernel]] void gemv_f32(
    const device float *matrix_a [[buffer(0)]],
    const device float *vector_x [[buffer(1)]],
    device float *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device float *bias [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float x_tile[TILE_K];
    MatrixPointerAccessor<float> accessor;
    accessor.init(matrix_a, params->N);
    if (has_bias) {
        gemv_kernel<MatrixPointerAccessor<float>, float, float, true>(
            accessor,
            vector_x,
            result_y,
            params,
            bias,
            gid,
            lid,
            x_tile);
    } else {
        gemv_kernel<MatrixPointerAccessor<float>, float, float, false>(
            accessor,
            vector_x,
            result_y,
            params,
            bias,
            gid,
            lid,
            x_tile);
    }
}

template <bool HasBias>
inline void run_gemv_dense(
    const device half *matrix_data,
    const device half *vector_x,
    device half *result_y,
    const constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    const float alpha,
    const float beta,
    uint3 gid,
    uint3 lid,
    threadgroup float *x_tile);


[[kernel]] void gemv_f16_dense(
    const device half *matrix_a [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float x_tile[TILE_K];
    run_gemv_dense<false>(matrix_a, vector_x, result_y, params, (const device half *)nullptr, (const device half *)nullptr, 1.0f, 0.0f, gid, lid, x_tile);
}

template <bool HasBias>
inline void run_gemv_dense(
    const device half *matrix_data,
    const device half *vector_x,
    device half *result_y,
    const constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    const float alpha,
    const float beta,
    uint3 gid,
    uint3 lid,
    threadgroup float *x_tile) {

    const uint N = params->N;
    const uint K = params->K;

    const uint base_col = gid.x * TILE_N + lid.x * GEMV_COLS_PER_THREAD;
    float sum[GEMV_COLS_PER_THREAD];
    for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) { sum[c] = 0.0f; }

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        const uint tile_limit = min(TILE_K, K - tile_base);
        const uint thread_linear = lid.x;
        gemv_stage_vector_tile(vector_x, tile_base, tile_limit, THREADGROUP_WIDTH, thread_linear, x_tile);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate for up to COLS_PER_THREAD columns per thread
        const uint tl = min(TILE_K, K - tile_base);
        for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
            const uint col = base_col + c;
            if (col >= N) break;
            const device half *matrix_ptr = matrix_data + tile_base * N + col;
            sum[c] += gemv_dot_shared_device<float>(x_tile, 1u, matrix_ptr, N, tl);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    gemv_store_results<HasBias>(sum, bias, residual, alpha, beta, base_col, N, result_y);
}
