// Dense GEMV kernels

// Optimized SIMD-Parallel FP16 GEMV
// Ports the logic from Q8 (Vectorized 4-Block + 2x Unroll) to F16.
// Layout: Column-Major (standard).


template <uint HEADS, bool HasBias>
void run_simd_f16_gemv(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid
) {
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    
    // Each Warp processes 1 Logical Output Column.
    const uint logical_col = gid.x * 4u + warp_id;
    
    bool head_active[HEADS];
    for (uint h = 0; h < HEADS; ++h) {
        head_active[h] = (logical_col < N[h]);
    }
    
    // Accumulators
    float acc[HEADS];
    for (uint h = 0; h < HEADS; ++h) acc[h] = 0.0f;

    // Pointers
    const device half *ptr_a[HEADS];
    for (uint h = 0; h < HEADS; ++h) {
        if (head_active[h]) {
            // [N, K] Layout -> A[col, 0].
            ptr_a[h] = matrix + (ulong)logical_col * K; 
        }
    }

    // Warp Parallelism:
    // We want to use 128-bit loads (float4 = 8 halves).
    // Each thread reads 8 elements.
    // Warp reads 8 * 32 = 256 K-elements per iter.
    
    const uint total_blocks = (K + 255u) / 256u; 
    
    // Align base pointers for vectorized access if possible? 
    // We assume K is somewhat aligned, or we handle tails.
    // Vector X is contiguous.
    
    // Thread offset in K
    const uint k_thread_offset = lane_id * 8u; // 0, 8, 16...
    
    // Unrolling 2x (Stride 512 halves) to balance latency hiding and register pressure.
    // Each thread loads 2 x float4 (2 x 8 halves = 16 halves).
    // Warp loads 32 x 16 = 512 halves per iteration.
    
    // Fast path loop for full 512-element chunks
    uint k_base = 0;
    for (; k_base + 512u <= K; k_base += 512u) {
        
        uint k_0 = k_base + k_thread_offset;
        uint k_1 = k_0 + 256u; 
        
        // Direct Vector Loads (No bounds check needed in fast loop)
        float4 xv_raw_0 = *(const device float4*)(vector_x + k_0);
        float4 xv_raw_1 = *(const device float4*)(vector_x + k_1);
        
        // Convert X
        half4 xv_lo_0 = as_type<half4>(xv_raw_0.xy); half4 xv_hi_0 = as_type<half4>(xv_raw_0.zw);
        half4 xv_lo_1 = as_type<half4>(xv_raw_1.xy); half4 xv_hi_1 = as_type<half4>(xv_raw_1.zw);
        
        float4 xv_lo_f32_0 = float4(xv_lo_0); float4 xv_hi_f32_0 = float4(xv_hi_0);
        float4 xv_lo_f32_1 = float4(xv_lo_1); float4 xv_hi_f32_1 = float4(xv_hi_1);

        for (uint h = 0; h < HEADS; ++h) {
            if (!head_active[h]) continue;
            
            float4 w_raw_0 = *(const device float4*)(ptr_a[h] + k_0);
            float4 w_raw_1 = *(const device float4*)(ptr_a[h] + k_1);
            
            half4 w_lo_0 = as_type<half4>(w_raw_0.xy); half4 w_hi_0 = as_type<half4>(w_raw_0.zw);
            half4 w_lo_1 = as_type<half4>(w_raw_1.xy); half4 w_hi_1 = as_type<half4>(w_raw_1.zw);
            
            float sum0 = dot(xv_lo_f32_0, float4(w_lo_0)) + dot(xv_hi_f32_0, float4(w_hi_0));
            float sum1 = dot(xv_lo_f32_1, float4(w_lo_1)) + dot(xv_hi_f32_1, float4(w_hi_1));
            
            acc[h] += (sum0 + sum1);
        }
    }
    
    // Tail handling for remaining elements (scalar/vector mixed)
    for (uint k = k_base + lane_id * 8u; k < K; k += 256u) { // Stride 256 (1 warp width)
         // Vector load if possible
         float4 xv_raw = float4(0.0f);
         if (k + 8 <= K) {
             xv_raw = *(const device float4*)(vector_x + k);
         } else {
             for (uint i=0; i<8 && k+i < K; ++i) ((thread half*)&xv_raw)[i] = vector_x[k+i];
         }
         
        half4 xv_lo = as_type<half4>(xv_raw.xy); half4 xv_hi = as_type<half4>(xv_raw.zw);
        float4 xv_lo_f32 = float4(xv_lo); float4 xv_hi_f32 = float4(xv_hi);

         for (uint h = 0; h < HEADS; ++h) {
            if (!head_active[h]) continue;
            
            float4 w_raw = float4(0.0f);
            if (k + 8 <= K) {
                w_raw = *(const device float4*)(ptr_a[h] + k);
            } else {
                for (uint i=0; i<8 && k+i < K; ++i) ((thread half*)&w_raw)[i] = ptr_a[h][k+i];
            }
            
            half4 w_lo = as_type<half4>(w_raw.xy); half4 w_hi = as_type<half4>(w_raw.zw);
            acc[h] += dot(xv_lo_f32, float4(w_lo)) + dot(xv_hi_f32, float4(w_hi));
         }
    }

    // Final Reduction (Warp Reduce)
    // Reduce the thread-local partial sums across the warp.
    for (uint h = 0; h < HEADS; ++h) {
        if (!head_active[h]) continue;
        
        float val = acc[h];
        
        val += simd_shuffle_xor(val, 16u);
        val += simd_shuffle_xor(val, 8u);
        val += simd_shuffle_xor(val, 4u);
        val += simd_shuffle_xor(val, 2u);
        val += simd_shuffle_xor(val, 1u);
        
        if (lane_id == 0) {
            if (has_bias_flags[h] && bias[h]) {
                val += (float)bias[h][logical_col];
            }
            
            if (HEADS == 1) { 
               if (residual) {
                   val = alpha * val + beta * (float)residual[logical_col];
               } else {
                   val = alpha * val;
               }
            } 
            result_y[h][logical_col] = (half)val;
        }
    }
}

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
