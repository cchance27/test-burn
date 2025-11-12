// Quantized GEMV/GEMM implementations

template <bool HasBias, bool Debug>
void run_gemv_q8_canonical(
    const device uchar *data,
    const device uchar *scale_bytes,
    const device half *vector_x,
    device half *result_y,
    const constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    const float alpha,
    const float beta,
    const constant uint &diag_col,
    uint3 gid,
    uint3 lid,
    threadgroup float *x_tile) {

    const uint N = params->N;
    const uint K = params->K;
    const uint weights_per_block = params->weights_per_block;
    const uint blocks_per_k = params->blocks_per_k;
    const uint base_col = gid.x * TILE_N + lid.x * GEMV_COLS_PER_THREAD;
    // Do not return early on inactive lanes before threadgroup staging; they must
    // still participate in barriers and cooperative loads to avoid UB on the last tile.
    if (weights_per_block == 0u) {
        return;
    }

    float sum[GEMV_COLS_PER_THREAD];
    for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) { sum[c] = 0.0f; }
    // Precompute typed pointers and strides (column offset applied per column)
    const device ushort *scales_u16 = (const device ushort *)(scale_bytes);
    const device char   *data_char  = (const device char *)(data);
    const uint scale_block_stride_elems = N; // in u16 units
    const uint data_block_stride_bytes = N * weights_per_block; // in bytes

    if constexpr (Debug) {
        // If any owned column matches diag_col, compute its per-block contributions directly
        for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
            const uint out_idx = base_col + c;
            if (out_idx >= N) break;
            if (out_idx != diag_col) continue;
            for (uint block = 0; block < blocks_per_k; ++block) {
                const uint k_base = block * weights_per_block;
                if (k_base >= K) {
                    result_y[block] = static_cast<half>(0.0f);
                    continue;
                }
                const uint scale_idx_e = block * scale_block_stride_elems + out_idx;
                const ushort bits = scales_u16[scale_idx_e];
                const float scale = static_cast<float>(as_type<half>(bits));
                const uint data_idx = block * data_block_stride_bytes + out_idx * weights_per_block;
                const device char *qs = data_char + data_idx;
                float acc_block = 0.0f;
                const uint limit = min(weights_per_block, K - k_base);
                for (uint i = 0; i < limit; ++i) {
                    acc_block = fma((float)vector_x[k_base + i], (float)qs[i], acc_block);
                }
                result_y[block] = static_cast<half>(scale * acc_block);
            }
            return; // Debug path done
        }
    }

    // Shared multi-head accumulation path reused with fused kernels.
    const uint scale_stride_bytes = N * Q8_CANONICAL_SCALE_BYTES;
    const Q8HeadRef<GEMV_COLS_PER_THREAD> heads[1] = {
        {N, scale_stride_bytes, Q8_CANONICAL_SCALE_BYTES, data_block_stride_bytes, scale_bytes, data, sum},
    };
    const uint max_cols = N;
    const bool use_wide = q8_should_use_wide(K, max_cols);

    q8_run_fused_heads(
        heads,
        base_col,
        weights_per_block,
        K,
        use_wide,
        vector_x,
        lid,
        x_tile);

    if constexpr (!Debug) {
        gemv_store_results<HasBias>(sum, bias, residual, alpha, beta, base_col, N, result_y);
    }
}

// Fused QKV GEMV for canonical Q8 weights (supports distinct N per row).
// Computes three outputs (Q, K, V) in a single pass of K, sharing the staged x_tile.
[[kernel]] void gemv_q8_fused3_f16(
    const device uchar *data_q [[buffer(0)]],
    const device uchar *data_k [[buffer(1)]],
    const device uchar *data_v [[buffer(2)]],
    const device half *vector_x [[buffer(3)]],
    device half *out_q [[buffer(4)]],
    device half *out_k [[buffer(5)]],
    device half *out_v [[buffer(6)]],
    const constant QkvFusedParams *params [[buffer(7)]],
    const device uchar *scales_q [[buffer(8)]],
    const device uchar *scales_k [[buffer(9)]],
    const device uchar *scales_v [[buffer(10)]],
    const device half *bias_q [[buffer(11)]],
    const device half *bias_k [[buffer(12)]],
    const device half *bias_v [[buffer(13)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float x_tile[TILE_K];

    const uint NQ = params->Nq;
    const uint NK = params->Nk;
    const uint NV = params->Nv;
    const uint K = params->K;
    const uint weights_per_block = params->weights_per_block;
    const uint base_col = gid.x * TILE_N + lid.x * GEMV_COLS_PER_THREAD;
    if (weights_per_block == 0u) {
        return;
    }

    float sum_heads[3][GEMV_COLS_PER_THREAD];
    for (uint h = 0u; h < 3u; ++h) {
        for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
            sum_heads[h][c] = 0.0f;
        }
    }

    const uint scale_stride_q = NQ * Q8_CANONICAL_SCALE_BYTES;
    const uint data_stride_q  = NQ * weights_per_block;
    const uint scale_stride_k = NK * Q8_CANONICAL_SCALE_BYTES;
    const uint data_stride_k  = NK * weights_per_block;
    const uint scale_stride_v = NV * Q8_CANONICAL_SCALE_BYTES;
    const uint data_stride_v  = NV * weights_per_block;

    const Q8HeadRef<GEMV_COLS_PER_THREAD> heads[3] = {
        {NQ, scale_stride_q, Q8_CANONICAL_SCALE_BYTES, data_stride_q, scales_q, data_q, sum_heads[0]},
        {NK, scale_stride_k, Q8_CANONICAL_SCALE_BYTES, data_stride_k, scales_k, data_k, sum_heads[1]},
        {NV, scale_stride_v, Q8_CANONICAL_SCALE_BYTES, data_stride_v, scales_v, data_v, sum_heads[2]},
    };

    const Q8FusedHeadOut<GEMV_COLS_PER_THREAD> outputs[3] = {
        {out_q, bias_q, NQ, params->has_bias_q},
        {out_k, bias_k, NK, params->has_bias_k},
        {out_v, bias_v, NV, params->has_bias_v},
    };

    const uint max_cols = max(NQ, max(NK, NV));
    const bool use_wide = q8_should_use_wide(K, max_cols);

    q8_run_fused_heads(
        heads,
        base_col,
        weights_per_block,
        K,
        use_wide,
        vector_x,
        lid,
        x_tile);

    q8_write_fused_heads(sum_heads, outputs, base_col);
}

[[kernel]] void gemv_q8_fused2_f16(
    const device uchar *data0 [[buffer(0)]],
    const device uchar *data1 [[buffer(1)]],
    const device half *vector_x [[buffer(2)]],
    device half *out0 [[buffer(3)]],
    device half *out1 [[buffer(4)]],
    const constant Q2FusedParams *params [[buffer(5)]],
    const device uchar *scales0 [[buffer(6)]],
    const device uchar *scales1 [[buffer(7)]],
    const device half *bias0 [[buffer(8)]],
    const device half *bias1 [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float x_tile[TILE_K];

    const uint K = params->K;
    const uint N0 = params->N0;
    const uint N1 = params->N1;
    const uint weights_per_block = params->weights_per_block;
    const bool use_bias0 = (params->has_bias0 != 0u) && (bias0 != (const device half*)nullptr);
    const bool use_bias1 = (params->has_bias1 != 0u) && (bias1 != (const device half*)nullptr);

    const uint base_col = gid.x * TILE_N + lid.x * GEMV_COLS_PER_THREAD;
    if (weights_per_block == 0u) {
        return;
    }

    float sum_heads[2][GEMV_COLS_PER_THREAD];
    for (uint h = 0u; h < 2u; ++h) {
        for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
            sum_heads[h][c] = 0.0f;
        }
    }

    const uint scale_stride0 = N0 * Q8_CANONICAL_SCALE_BYTES;
    const uint data_stride0  = N0 * weights_per_block;
    const uint scale_stride1 = N1 * Q8_CANONICAL_SCALE_BYTES;
    const uint data_stride1  = N1 * weights_per_block;

    const Q8HeadRef<GEMV_COLS_PER_THREAD> heads[2] = {
        {N0, scale_stride0, Q8_CANONICAL_SCALE_BYTES, data_stride0, scales0, data0, sum_heads[0]},
        {N1, scale_stride1, Q8_CANONICAL_SCALE_BYTES, data_stride1, scales1, data1, sum_heads[1]},
    };

    const Q8FusedHeadOut<GEMV_COLS_PER_THREAD> outputs[2] = {
        {out0, bias0, N0, use_bias0 ? 1u : 0u},
        {out1, bias1, N1, use_bias1 ? 1u : 0u},
    };

    const uint max_cols = max(N0, N1);
    const bool use_wide = q8_should_use_wide(K, max_cols);

    q8_run_fused_heads(
        heads,
        base_col,
        weights_per_block,
        K,
        use_wide,
        vector_x,
        lid,
        x_tile);

    q8_write_fused_heads(sum_heads, outputs, base_col);
}

struct GemmQ8NtParams {
    uint m;
    uint n;
    uint k;
    uint lda;
    uint ldc;
    uint blocks_per_k;
    uint weights_per_block;
    uint has_bias;
};

// GEMM kernel for canonical Q8 weights with transpose_b = true (NT layout).
// Supports up to 4 output rows per threadgroup tile for decode workloads.
[[kernel]] void gemm_q8_nt_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint blocks_per_k = params->blocks_per_k;
    const uint weights_per_block = params->weights_per_block;

    // Correctness-first micro-path for all m=1 shapes: compute directly from global A.
    // Keeps parity tight for test shapes; decode uses GEMV path so perf unaffected.
    if (m == 1u) {
        const uint col = gid.x * 128u + tid.x;
        if (col >= n) {
            return;
        }
        float acc = 0.0f;
        for (uint block = 0; block < blocks_per_k; ++block) {
            const uint scale_idx = (block * n + col) * Q8_CANONICAL_SCALE_BYTES;
            const ushort bits = (ushort)scale_bytes[scale_idx] | ((ushort)scale_bytes[scale_idx + 1u] << 8);
            const float scale = static_cast<float>(as_type<half>(bits));
            const device char *qs = (const device char *)(matrix_data + (block * n + col) * weights_per_block);
            const uint base_k = block * weights_per_block;
            const uint limit = min(weights_per_block, k > base_k ? (k - base_k) : 0u);
            float block_sum = 0.0f;
            for (uint i = 0; i < limit; ++i) {
                const float a = static_cast<float>(matrix_a[0 * lda + (base_k + i)]);
                block_sum = fma(a, (float)qs[i], block_sum);
            }
            acc += scale * block_sum;
        }
        if (params->has_bias != 0u) {
            acc += static_cast<float>(bias[col]);
        }
        result_y[0 * ldc + col] = static_cast<half>(acc);
        return;
    }

    // Tile configuration tuned for decode (m small, n large)
    constexpr uint TILE_COLS = 128;
    constexpr uint ROWS_PER_TILE = 4;
    constexpr uint THREADS_PER_TG = TILE_COLS;

    threadgroup float a_tile[ROWS_PER_TILE * Q8_0_WEIGHTS_PER_BLOCK];

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) {
        return;
    }
    const uint rows_this_tile = min((uint)ROWS_PER_TILE, m - row_tile);

    const uint col = gid.x * TILE_COLS + tid.x;
    if (col >= n) {
        return;
    }

    float accum[ROWS_PER_TILE];
    for (uint r = 0; r < rows_this_tile; ++r) {
        accum[r] = 0.0f;
    }

    q8_for_each_block_k(
        blocks_per_k,
        weights_per_block,
        k,
        [&](uint block, uint block_k_start, uint valid) {
            // Stage the A tile for this K block
            for (uint idx = tid.x; idx < rows_this_tile * weights_per_block; idx += THREADS_PER_TG) {
                const uint local_row = idx / weights_per_block;
                const uint offset = idx % weights_per_block;
                const uint global_k = block_k_start + offset;
                float val = 0.0f;
                if (global_k < k) {
                    const uint global_row = row_tile + local_row;
                    val = static_cast<float>(matrix_a[global_row * lda + global_k]);
                }
                a_tile[local_row * weights_per_block + offset] = val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Load scale and quant data for this column/block
            const uint scale_idx = (block * n + col) * Q8_CANONICAL_SCALE_BYTES;
            const device uchar *sb = scale_bytes + scale_idx;
            const ushort bits = (ushort)sb[0] | ((ushort)sb[1] << 8);
            const float scale = static_cast<float>(as_type<half>(bits));
            const device char *qs = (const device char *)(matrix_data + (block * n + col) * weights_per_block);

            for (uint local_row = 0; local_row < rows_this_tile; ++local_row) {
                float block_sum = 0.0f;
                const threadgroup float *a_base = a_tile + local_row * weights_per_block;
                for (uint i = 0; i < valid; ++i) {
                    block_sum = fma(a_base[i], (float)qs[i], block_sum);
                }
                accum[local_row] = fma(scale, block_sum, accum[local_row]);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        });

    for (uint local_row = 0; local_row < rows_this_tile; ++local_row) {
        const uint out_row = row_tile + local_row;
        float value = accum[local_row];
        if (params->has_bias != 0u) {
            value += static_cast<float>(bias[col]);
        }
        result_y[out_row * ldc + col] = static_cast<half>(value);
    }
}

// Canonical large-N GEMM kernel for Q8 weights. Extends the NT variant by
// supporting additional output rows per tile (8 vs 4) and aggressively
// vectorizing the inner products for high-N decode blocks.
template <uint ROWS_PER_TILE>
inline void gemm_q8_canonical_large_n_impl(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    threadgroup float (&a_tile)[2][ROWS_PER_TILE * Q8_0_WEIGHTS_PER_BLOCK]) {
    constexpr uint TILE_COLS_TOTAL = 128u;
    constexpr uint TILE_COLS_PER_TG = TILE_COLS_TOTAL / 2u;
    constexpr uint COLS_PER_THREAD = 2u;
    constexpr uint TG_ROWS = 4u;
    constexpr uint TG_COL_LANES = TILE_COLS_PER_TG / COLS_PER_THREAD;
    constexpr uint THREADS_PER_TG = TG_COL_LANES * TG_ROWS;

    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint blocks_per_k = params->blocks_per_k;
    const uint weights_per_block = params->weights_per_block;

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) {
        return;
    }
    const uint rows_this_tile = min((uint)ROWS_PER_TILE, m - row_tile);

    const uint lane_x = tid3.x;
    const uint lane_y = tid3.y;
    const uint col_block = gid.x * TILE_COLS_PER_TG + lane_x * COLS_PER_THREAD;
    if (col_block >= n) {
        return;
    }

    float accum[ROWS_PER_TILE][COLS_PER_THREAD];
    for (uint r = 0; r < rows_this_tile; ++r) {
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            accum[r][c] = 0.0f;
        }
    }

    const uint scale_block_stride = n * Q8_CANONICAL_SCALE_BYTES;
    const uint data_block_stride = n * weights_per_block;
    const device uchar *scale_lane_ptrs[COLS_PER_THREAD];
    const device char *data_lane_ptrs[COLS_PER_THREAD];
    bool column_active[COLS_PER_THREAD];
    for (uint c = 0; c < COLS_PER_THREAD; ++c) {
        const uint col = col_block + c;
        if (col < n) {
            scale_lane_ptrs[c] = scale_bytes + col * Q8_CANONICAL_SCALE_BYTES;
            data_lane_ptrs[c] = (const device char *)(matrix_data + col * weights_per_block);
            column_active[c] = true;
        } else {
            scale_lane_ptrs[c] = scale_bytes;
            data_lane_ptrs[c] = (const device char *)(matrix_data);
            column_active[c] = false;
        }
    }

    const uint linear_tid = lane_y * TG_COL_LANES + lane_x;
    q8_for_each_block_k(
        blocks_per_k,
        weights_per_block,
        k,
        [&](uint block, uint block_k_start, uint valid) {
            const uint buffer_idx = block & 1u;
            for (uint idx = linear_tid; idx < rows_this_tile * weights_per_block; idx += THREADS_PER_TG) {
                const uint local_row = idx / weights_per_block;
                const uint offset = idx % weights_per_block;
                const uint global_k = block_k_start + offset;
                float val = 0.0f;
                if (global_k < k) {
                    const uint global_row = row_tile + local_row;
                    val = static_cast<float>(matrix_a[global_row * lda + global_k]);
                }
                a_tile[buffer_idx][local_row * weights_per_block + offset] = val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            float scale_vec[COLS_PER_THREAD];
            const device char *qs_ptrs[COLS_PER_THREAD];
            bool active_cols[COLS_PER_THREAD];
            for (uint c = 0; c < COLS_PER_THREAD; ++c) {
                if (column_active[c]) {
                    const ushort bits = (ushort)scale_lane_ptrs[c][0] | ((ushort)scale_lane_ptrs[c][1] << 8);
                    scale_vec[c] = static_cast<float>(as_type<half>(bits));
                    qs_ptrs[c] = data_lane_ptrs[c];
                    active_cols[c] = true;
                } else {
                    scale_vec[c] = 0.0f;
                    qs_ptrs[c] = (const device char *)(matrix_data);
                    active_cols[c] = false;
                }
            }

            const uint consume_buffer = buffer_idx;
            for (uint local_row = lane_y; local_row < rows_this_tile; local_row += TG_ROWS) {
                threadgroup float *a_base = &a_tile[consume_buffer][local_row * weights_per_block];
                uint processed = 0u;

                while (processed + 8u <= valid) {
                    const threadgroup float *x_ptr = a_base + processed;
                    const float4 x0 = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
                    const float4 x1 = float4(x_ptr[4], x_ptr[5], x_ptr[6], x_ptr[7]);

                    if (active_cols[0]) {
                        const device char4 *qv0 = (const device char4 *)(qs_ptrs[0] + processed + 0u);
                        const device char4 *qv1 = (const device char4 *)(qs_ptrs[0] + processed + 4u);
                        const char4 q0 = *qv0;
                        const char4 q1 = *qv1;
                        const float block_sum0 = dot(x0, float4(q0)) + dot(x1, float4(q1));
                        accum[local_row][0] = fma(scale_vec[0], block_sum0, accum[local_row][0]);
                    }
                    if (COLS_PER_THREAD > 1 && active_cols[1]) {
                        const device char4 *qv0 = (const device char4 *)(qs_ptrs[1] + processed + 0u);
                        const device char4 *qv1 = (const device char4 *)(qs_ptrs[1] + processed + 4u);
                        const char4 q0 = *qv0;
                        const char4 q1 = *qv1;
                        const float block_sum1 = dot(x0, float4(q0)) + dot(x1, float4(q1));
                        accum[local_row][1] = fma(scale_vec[1], block_sum1, accum[local_row][1]);
                    }

                    processed += 8u;
                }

                while (processed + 4u <= valid) {
                    const threadgroup float *x_ptr = a_base + processed;
                    const float4 x = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);

                    if (active_cols[0]) {
                        const device char4 *qv = (const device char4 *)(qs_ptrs[0] + processed);
                        const char4 q = *qv;
                        const float block_sum0 = dot(x, float4(q));
                        accum[local_row][0] = fma(scale_vec[0], block_sum0, accum[local_row][0]);
                    }
                    if (COLS_PER_THREAD > 1 && active_cols[1]) {
                        const device char4 *qv = (const device char4 *)(qs_ptrs[1] + processed);
                        const char4 q = *qv;
                        const float block_sum1 = dot(x, float4(q));
                        accum[local_row][1] = fma(scale_vec[1], block_sum1, accum[local_row][1]);
                    }

                    processed += 4u;
                }

                while (processed < valid) {
                    const float x_val = a_base[processed];
                    for (uint c = 0; c < COLS_PER_THREAD; ++c) {
                        if (!active_cols[c]) {
                            continue;
                        }
                        const float q_val = (float)qs_ptrs[c][processed];
                        accum[local_row][c] = fma(scale_vec[c], x_val * q_val, accum[local_row][c]);
                    }
                    processed += 1u;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint c = 0; c < COLS_PER_THREAD; ++c) {
                if (column_active[c]) {
                    scale_lane_ptrs[c] += scale_block_stride;
                    data_lane_ptrs[c] += data_block_stride;
                }
            }
        });

    for (uint local_row = lane_y; local_row < rows_this_tile; local_row += TG_ROWS) {
        const uint out_row = row_tile + local_row;
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            const uint col = col_block + c;
            if (col >= n) {
                continue;
            }
            float value = accum[local_row][c];
            if (params->has_bias != 0u) {
                value += static_cast<float>(bias[col]);
            }
            result_y[out_row * ldc + col] = static_cast<half>(value);
        }
    }
}

[[kernel]] void gemm_q8_canonical_large_n_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[2][32u * Q8_0_WEIGHTS_PER_BLOCK];
    gemm_q8_canonical_large_n_impl<32u>(
        matrix_data,
        scale_bytes,
        matrix_a,
        result_y,
        params,
        bias,
        gid,
        tid3,
        a_tile);
}

[[kernel]] void gemm_q8_canonical_large_n_rows16_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[2][16u * Q8_0_WEIGHTS_PER_BLOCK];
    gemm_q8_canonical_large_n_impl<16u>(
        matrix_data,
        scale_bytes,
        matrix_a,
        result_y,
        params,
        bias,
        gid,
        tid3,
        a_tile);
}

// Multi-row GEMV for canonical Q8 weights (computes up to 4 rows of Y per tile).
// Uses the same canonical addressing as GEMV: block-major across K, then columns N.
[[kernel]] void gemv_q8_rows_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint weights_per_block = params->weights_per_block;
    const uint blocks_per_k = params->blocks_per_k;
    const uint has_bias = params->has_bias;
    if (weights_per_block == 0u || blocks_per_k == 0u) {
        return;
    }

    constexpr uint ROWS_PER_TILE = 4;
    constexpr uint THREADS_PER_TG = THREADGROUP_WIDTH;
    threadgroup float x_rows[ROWS_PER_TILE * Q8_0_WEIGHTS_PER_BLOCK];

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) {
        return;
    }
    const uint rows_this_tile = min((uint)ROWS_PER_TILE, m - row_tile);

    const uint base_col = gid.x * TILE_N + tid.x * GEMV_COLS_PER_THREAD;
    float accum_rows[ROWS_PER_TILE][GEMV_COLS_PER_THREAD];
    for (uint r = 0; r < ROWS_PER_TILE; ++r) {
        for (uint c = 0; c < GEMV_COLS_PER_THREAD; ++c) {
            accum_rows[r][c] = 0.0f;
        }
    }

    q8_for_each_block_k(
        blocks_per_k,
        weights_per_block,
        k,
        [&](uint block, uint block_k_start, uint valid) {
            for (uint idx = tid.x; idx < rows_this_tile * weights_per_block; idx += THREADS_PER_TG) {
                const uint local_row = idx / weights_per_block;
                const uint offset = idx % weights_per_block;
                float val = 0.0f;
                if (local_row < rows_this_tile) {
                    const uint global_k = block_k_start + offset;
                    if (global_k < k) {
                        const uint global_row = row_tile + local_row;
                        val = static_cast<float>(matrix_a[global_row * lda + global_k]);
                    }
                }
                x_rows[local_row * weights_per_block + offset] = val;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint c = 0; c < GEMV_COLS_PER_THREAD; ++c) {
                const uint col = base_col + c;
                if (col >= n) {
                    continue;
                }
                const uint block_col_idx = block * n + col;
                const uint scale_idx = block_col_idx * Q8_CANONICAL_SCALE_BYTES;
                const ushort bits = (ushort)scale_bytes[scale_idx] | ((ushort)scale_bytes[scale_idx + 1u] << 8);
                const float scale = static_cast<float>(as_type<half>(bits));
                const device char *qs_base = (const device char *)(matrix_data + block_col_idx * weights_per_block);

                for (uint row = 0; row < rows_this_tile; ++row) {
                    threadgroup float *x_base = &x_rows[row * weights_per_block];
                    float block_sum = 0.0f;
                    uint processed = 0u;

                    while (processed + 8u <= valid) {
                        const threadgroup float *x_ptr = x_base + processed;
                        const float4 x0 = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
                        const float4 x1 = float4(x_ptr[4], x_ptr[5], x_ptr[6], x_ptr[7]);

                        const device char4 *qv0 = (const device char4 *)(qs_base + processed + 0u);
                        const device char4 *qv1 = (const device char4 *)(qs_base + processed + 4u);
                        const char4 q0 = *qv0;
                        const char4 q1 = *qv1;
                        block_sum += dot(x0, float4(q0)) + dot(x1, float4(q1));

                        processed += 8u;
                    }

                    while (processed + 4u <= valid) {
                        const threadgroup float *x_ptr = x_base + processed;
                        const float4 x = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
                        const device char4 *qv = (const device char4 *)(qs_base + processed);
                        const char4 q = *qv;
                        block_sum += dot(x, float4(q));
                        processed += 4u;
                    }

                    while (processed < valid) {
                        block_sum = fma(x_base[processed], (float)qs_base[processed], block_sum);
                        processed += 1u;
                    }

                    accum_rows[row][c] = fma(scale, block_sum, accum_rows[row][c]);
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        });

    for (uint row = 0; row < rows_this_tile; ++row) {
        for (uint c = 0; c < GEMV_COLS_PER_THREAD; ++c) {
            const uint col = base_col + c;
            if (col >= n) continue;
            float v = accum_rows[row][c];
            if (has_bias != 0u && bias != (const device half *)nullptr) {
                v += static_cast<float>(bias[col]);
            }
            result_y[(row_tile + row) * ldc + col] = static_cast<half>(v);
        }
    }
}
