// Custom M=1 dot product kernel based on MLX SIMD patterns
// Optimized for M=1 shapes with minimal overhead
// Based on MLX patterns from dot_product_example_from_mlx.metal
// Using proper SIMD group 8x8 operations for optimal performance on Apple GPUs

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// Simple M=1 kernel - basic approach to compute C[0, :] = A[0, :] @ B
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v2_basic(
    device const half* A [[buffer(0)]],      // Shape: [M=1, K]
    device const half* B [[buffer(1)]],      // Shape: [K, N] 
    device half* C [[buffer(2)]],            // Shape: [M=1, N]
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    
    // Each thread processes multiple N values to distribute work evenly
    for (int n_idx = lid.x; n_idx < N; n_idx += 256) {
        float sum = 0.0f;
        
        // Compute dot product A[0, :] * B[:, n_idx]
        for (int k_idx = 0; k_idx < K; k_idx++) {
            sum += float(A[k_idx]) * float(B[k_idx * N + n_idx]);
        }
        
        C[n_idx] = half(sum);
    }
}

// Optimized version using MLX-inspired tiling approach for M=1
// This version uses a more efficient threading pattern and memory access
template <const short NBLK = 2>
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v2_tiled(
    device const half* A [[buffer(0)]],      // Shape: [M=1, K]
    device const half* B [[buffer(1)]],      // Shape: [K, N] 
    device half* C [[buffer(2)]],            // Shape: [M=1, N]
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    
    // Calculate which N values this threadgroup is responsible for
    short block_size = NBLK * 32;  // Process in blocks to improve cache usage
    int start_n = gid.x * block_size;
    int end_n = min(start_n + block_size, N);
    
    // Each thread within the threadgroup processes multiple N values
    for (int n_idx = start_n + lid.x; n_idx < end_n; n_idx += 256) {
        float sum = 0.0f;
        
        // Compute the dot product A[0, :] * B[:, n_idx]
        for (int k_idx = 0; k_idx < K; k_idx++) {
            sum += float(A[k_idx]) * float(B[k_idx * N + n_idx]);
        }
        
        C[n_idx] = half(sum);
    }
}

// Instantiate the template with different block sizes
template [[host_name("m1_dot_product_v2_tiled1")]] [[kernel]] void m1_dot_product_v2_tiled<1>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_tiled2")]] [[kernel]] void m1_dot_product_v2_tiled<2>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_tiled4")]] [[kernel]] void m1_dot_product_v2_tiled<4>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// SIMD group optimized version for M=1 case (similar to MLX sgemm_naive_simd pattern)
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v2_simd_naive(
    device const half* A [[buffer(0)]],      // Shape: [M=1, K]
    device const half* B [[buffer(1)]],      // Shape: [K, N] 
    device half* C [[buffer(2)]],            // Shape: [M=1, N]
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    
    // For M=1 case, assign threads to work on N dimension
    // Use simple linear assignment for better memory coalescing
    for (int n_idx = lid.x + gid.x * 256; n_idx < N; n_idx += 256 * 16) { // 16 threadgroups
        if (n_idx >= N) break;
        
        float sum = 0.0f;
        
        // Compute A[0, :] * B[:, n_idx]
        for (int k_idx = 0; k_idx < K; k_idx++) {
            sum += float(A[k_idx]) * float(B[k_idx * N + n_idx]);
        }
        
        C[n_idx] = half(sum);
    }
}

// Even more optimized version for small K values, using vectorized access
[[kernel, max_total_threads_per_threadgroup(32)]]
void m1_dot_product_v2_small_k(
    device const half* A [[buffer(0)]],      // Shape: [M=1, K]
    device const half* B [[buffer(1)]],      // Shape: [K, N] 
    device half* C [[buffer(2)]],            // Shape: [M=1, N]
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    
    // For small K, 32 threads (one SIMD group) is more efficient
    for (int n_idx = lid.x; n_idx < N; n_idx += 32) {
        float sum = 0.0f;
        
        // Compute the dot product A[0, :] * B[:, n_idx] for this element
        for (int k_idx = 0; k_idx < K; k_idx++) {
            sum += float(A[k_idx]) * float(B[k_idx * N + n_idx]);
        }
        
        C[n_idx] = half(sum);
    }
}

// ---------------------------------------------------------------------------
// NT-targeted kernels: per-thread column accumulation optimized for M=1
// ---------------------------------------------------------------------------

// Column-major (NT) variant with K-tiling and A staging into threadgroup memory.
// Threads within the TG cooperate to load A tiles once, then each thread streams B
// for its assigned output columns and accumulates into a private register.
// Optional double-buffer prefetching is used to reduce visible latency between tiles.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v2_nt_kernel_col(
    device const half* A [[buffer(0)]],      // Shape: [M=1, K]
    device const half* B [[buffer(1)]],      // Shape: [K, N] laid out column-major when transposeB=true
    device half* C [[buffer(2)]],            // Shape: [M=1, N]
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    if (M != 1) {
        return;  // Defensive: kernel is specialized for M == 1 only
    }

    const ushort lane = static_cast<ushort>(lid.x);
    const int group_base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    // Support multiple columns per thread when COLUMNS_PER_TG > THREADGROUP_WIDTH
    const int col0 = group_base_col + static_cast<int>(lane);
    const int col1 = group_base_col + static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH);
    const bool has1 = (static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH)) < static_cast<int>(COLUMNS_PER_TG);
    const bool active0 = (static_cast<int>(lane) < static_cast<int>(COLUMNS_PER_TG)) && (col0 < N);
    const bool active1 = has1 && (col1 < N);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const device half* __restrict column_ptr0 = active0 ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict column_ptr1 = active1 ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;

    // Stage A tiles into threadgroup memory once per TG, with double buffering.
    threadgroup half a_tile_buf[2][BK];

    // Preload first tile (tile 0) into buffer 0 cooperatively across all threads.
    int k_base = 0;
    int cur = 0;
    int next = 1;
    {
        const int tile_len0 = (K - k_base) > BK ? BK : max(0, K - k_base);
        for (int kk = lane; kk < tile_len0; kk += THREADGROUP_WIDTH) {
            a_tile_buf[cur][kk] = A[k_base + kk];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (k_base < K) {
        const int tile_len = (K - k_base) > BK ? BK : max(0, K - k_base);

        // Prefetch next tile into alternate buffer
        const int next_base = k_base + BK;
        if (next_base < K) {
            const int next_len = (K - next_base) > BK ? BK : max(0, K - next_base);
            for (int kk = lane; kk < next_len; kk += THREADGROUP_WIDTH) {
                a_tile_buf[next][kk] = A[next_base + kk];
            }
        }

        if (active0 || active1) {
            // Consume current tile from threadgroup memory, stream B from device memory
            int kk = 0;
            // Unroll by 4 for better ILP; guard tail with tile_len
            for (; kk + 3 < tile_len; kk += 4) {
                const float a0 = float(a_tile_buf[cur][kk    ]);
                const float a1 = float(a_tile_buf[cur][kk + 1]);
                const float a2 = float(a_tile_buf[cur][kk + 2]);
                const float a3 = float(a_tile_buf[cur][kk + 3]);
                const int k0 = k_base + kk;
                if (active0) {
                    acc0 = fma(a0, float(column_ptr0[k0    ]), acc0);
                    acc0 = fma(a1, float(column_ptr0[k0 + 1]), acc0);
                    acc0 = fma(a2, float(column_ptr0[k0 + 2]), acc0);
                    acc0 = fma(a3, float(column_ptr0[k0 + 3]), acc0);
                }
                if (active1) {
                    acc1 = fma(a0, float(column_ptr1[k0    ]), acc1);
                    acc1 = fma(a1, float(column_ptr1[k0 + 1]), acc1);
                    acc1 = fma(a2, float(column_ptr1[k0 + 2]), acc1);
                    acc1 = fma(a3, float(column_ptr1[k0 + 3]), acc1);
                }
            }
            for (; kk < tile_len; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) { acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0); }
                if (active1) { acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1); }
            }
        }

        // Ensure prefetch completed before swapping buffers for next iteration
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += BK;
        cur ^= 1;
        next ^= 1;
    }

    if (active0) { C[col0] = half(acc0); }
    if (active1) { C[col1] = half(acc1); }
}

// Column-major (NT) variant with A-tiling + vectorized B loads (half4) and deeper unroll.
// Reads A from threadgroup memory, streams B in half4 vectors when aligned.
// Each thread accumulates up to two columns (lane and lane+THREADGROUP_WIDTH).
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v2_nt_kernel_col_vec4(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    if (M != 1) { return; }

    const ushort lane = static_cast<ushort>(lid.x);
    const int group_base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    const int col0 = group_base_col + static_cast<int>(lane);
    const int col1 = group_base_col + static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH);
    const bool has1 = (static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH)) < static_cast<int>(COLUMNS_PER_TG);
    const bool active0 = (static_cast<int>(lane) < static_cast<int>(COLUMNS_PER_TG)) && (col0 < N);
    const bool active1 = has1 && (col1 < N);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const device half* __restrict column_ptr0 = active0 ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict column_ptr1 = active1 ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;

    threadgroup half a_tile_buf[2][BK];

    int k_base = 0;
    int cur = 0;
    int next = 1;
    {
        const int tile_len0 = (K - k_base) > BK ? BK : max(0, K - k_base);
        for (int kk = lane; kk < tile_len0; kk += THREADGROUP_WIDTH) {
            a_tile_buf[cur][kk] = A[k_base + kk];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (k_base < K) {
        const int tile_len = (K - k_base) > BK ? BK : max(0, K - k_base);

        // Prefetch next A tile
        const int next_base = k_base + BK;
        if (next_base < K) {
            const int next_len = (K - next_base) > BK ? BK : max(0, K - next_base);
            for (int kk = lane; kk < next_len; kk += THREADGROUP_WIDTH) {
                a_tile_buf[next][kk] = A[next_base + kk];
            }
        }

        if (active0 || active1) {
            int kk = 0;
            // Vectorized path requires 4-aligned k index
            // Process head to reach alignment
            for (; kk < tile_len && ((k_base + kk) & 3) != 0; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) { acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0); }
                if (active1) { acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1); }
            }

            // Vectorized core: unroll by 8 using two half4 loads
            for (; kk + 7 < tile_len; kk += 8) {
                // Load A scalars
                const float a0 = float(a_tile_buf[cur][kk    ]);
                const float a1 = float(a_tile_buf[cur][kk + 1]);
                const float a2 = float(a_tile_buf[cur][kk + 2]);
                const float a3 = float(a_tile_buf[cur][kk + 3]);
                const float a4 = float(a_tile_buf[cur][kk + 4]);
                const float a5 = float(a_tile_buf[cur][kk + 5]);
                const float a6 = float(a_tile_buf[cur][kk + 6]);
                const float a7 = float(a_tile_buf[cur][kk + 7]);

                const int k0 = k_base + kk;
                if (active0) {
                    const device half4* p0 = reinterpret_cast<const device half4*>(column_ptr0 + k0);
                    const half4 hb0 = *p0;
                    const half4 hb1 = *(p0 + 1);
                    const float4 fb0 = float4(hb0);
                    const float4 fb1 = float4(hb1);
                    acc0 = fma(a0, fb0.x, acc0);
                    acc0 = fma(a1, fb0.y, acc0);
                    acc0 = fma(a2, fb0.z, acc0);
                    acc0 = fma(a3, fb0.w, acc0);
                    acc0 = fma(a4, fb1.x, acc0);
                    acc0 = fma(a5, fb1.y, acc0);
                    acc0 = fma(a6, fb1.z, acc0);
                    acc0 = fma(a7, fb1.w, acc0);
                }
                if (active1) {
                    const device half4* p1 = reinterpret_cast<const device half4*>(column_ptr1 + k0);
                    const half4 hb0 = *p1;
                    const half4 hb1 = *(p1 + 1);
                    const float4 fb0 = float4(hb0);
                    const float4 fb1 = float4(hb1);
                    acc1 = fma(a0, fb0.x, acc1);
                    acc1 = fma(a1, fb0.y, acc1);
                    acc1 = fma(a2, fb0.z, acc1);
                    acc1 = fma(a3, fb0.w, acc1);
                    acc1 = fma(a4, fb1.x, acc1);
                    acc1 = fma(a5, fb1.y, acc1);
                    acc1 = fma(a6, fb1.z, acc1);
                    acc1 = fma(a7, fb1.w, acc1);
                }
            }

            // Tail of current tile
            for (; kk < tile_len; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) { acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0); }
                if (active1) { acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1); }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += BK;
        cur ^= 1;
        next ^= 1;
    }

    if (active0) { C[col0] = half(acc0); }
    if (active1) { C[col1] = half(acc1); }
}

// Column-major variant with both A-tiling (double-buffered) and B-tiling (single-buffered).
// Stages A tiles into threadgroup memory similarly to the col kernel, and additionally stages
// a BK x COLUMNS_PER_TG slice of B for each tile to reduce global memory latency.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v2_nt_kernel_col_bt(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    if (M != 1) { return; }

    const ushort lane = static_cast<ushort>(lid.x);
    const int group_base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    const int col0 = group_base_col + static_cast<int>(lane);
    const int col1 = group_base_col + static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH);
    const bool has1 = (static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH)) < static_cast<int>(COLUMNS_PER_TG);
    const bool active0 = (static_cast<int>(lane) < static_cast<int>(COLUMNS_PER_TG)) && (col0 < N);
    const bool active1 = has1 && (col1 < N);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    // A double-buffered tile and B single-buffered tile in threadgroup memory
    threadgroup half a_tile_buf[2][BK];
    threadgroup half b_tile_buf[BK][COLUMNS_PER_TG];

    // Preload first A tile
    int k_base = 0;
    int cur = 0;
    int next = 1;
    {
        const int tile_len0 = (K - k_base) > BK ? BK : max(0, K - k_base);
        for (int kk = lane; kk < tile_len0; kk += THREADGROUP_WIDTH) {
            a_tile_buf[cur][kk] = A[k_base + kk];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (k_base < K) {
        const int tile_len = (K - k_base) > BK ? BK : max(0, K - k_base);

        // Prefetch next A tile (if any) into alternate buffer
        const int next_base = k_base + BK;
        if (next_base < K) {
            const int next_len = (K - next_base) > BK ? BK : max(0, K - next_base);
            for (int kk = lane; kk < next_len; kk += THREADGROUP_WIDTH) {
                a_tile_buf[next][kk] = A[next_base + kk];
            }
        }

        // Stage B tile (BK x COLUMNS_PER_TG). Lanes load columns in strides.
        for (int col_off = lane; col_off < COLUMNS_PER_TG; col_off += THREADGROUP_WIDTH) {
            const int col = group_base_col + col_off;
            const bool col_valid = (col < N);
            const device half* __restrict col_ptr = col_valid
                ? (B + static_cast<size_t>(col) * static_cast<size_t>(K))
                : nullptr;
            for (int kk = 0; kk < tile_len; ++kk) {
                b_tile_buf[kk][col_off] = col_valid ? col_ptr[k_base + kk] : half(0.0h);
            }
        }

        // Ensure B tile is visible before compute
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active0 || active1) {
            // Compute using staged A and B tiles
            int kk = 0;
            for (; kk + 3 < tile_len; kk += 4) {
                const float a0 = float(a_tile_buf[cur][kk    ]);
                const float a1 = float(a_tile_buf[cur][kk + 1]);
                const float a2 = float(a_tile_buf[cur][kk + 2]);
                const float a3 = float(a_tile_buf[cur][kk + 3]);
                if (active0) {
                    acc0 = fma(a0, float(b_tile_buf[kk    ][lane]), acc0);
                    acc0 = fma(a1, float(b_tile_buf[kk + 1][lane]), acc0);
                    acc0 = fma(a2, float(b_tile_buf[kk + 2][lane]), acc0);
                    acc0 = fma(a3, float(b_tile_buf[kk + 3][lane]), acc0);
                }
                if (active1) {
                    const int lane1 = static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH);
                    acc1 = fma(a0, float(b_tile_buf[kk    ][lane1]), acc1);
                    acc1 = fma(a1, float(b_tile_buf[kk + 1][lane1]), acc1);
                    acc1 = fma(a2, float(b_tile_buf[kk + 2][lane1]), acc1);
                    acc1 = fma(a3, float(b_tile_buf[kk + 3][lane1]), acc1);
                }
            }
            for (; kk < tile_len; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) { acc0 = fma(av, float(b_tile_buf[kk][lane]), acc0); }
                if (active1) { const int lane1 = static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH); acc1 = fma(av, float(b_tile_buf[kk][lane1]), acc1); }
            }
        }

        // Make sure next A tile is ready before swapping
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += BK;
        cur ^= 1;
        next ^= 1;
    }

    if (active0) { C[col0] = half(acc0); }
    if (active1) { C[col1] = half(acc1); }
}

// Row-major variant (index = k*N + column)
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v2_nt_kernel_row(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    if (M != 1) { return; }

    const ushort lane = static_cast<ushort>(lid.x);
    const int group_base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    for (int col_offset = lane; col_offset < COLUMNS_PER_TG; col_offset += THREADGROUP_WIDTH) {
        const int column = group_base_col + col_offset;
        if (column >= N) continue;

        float accumulator = 0.0f;
        size_t index = static_cast<size_t>(column);
        for (int k_idx = 0; k_idx < K; ++k_idx, index += static_cast<size_t>(N)) {
            const float a_val = float(A[k_idx]);
            const float b_val = float(B[index]);
            accumulator = fma(a_val, b_val, accumulator);
        }
        C[column] = half(accumulator);
    }
}

// Instantiate NT kernels for specific tile sizes (column-major indexing)
template [[host_name("m1_dot_product_v2_nt_bn128_col")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<128, 128, 4>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<128, 64, 4>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// New BK/TG variants for occupancy and tiling sweeps
template [[host_name("m1_dot_product_v2_nt_bn128_col_bk64_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_bk64_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn128_col_bk128_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_bk128_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn128_col_bk64_tg64")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<64, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_bk64_tg64")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col<64, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// Instantiate vectorized col kernels
template [[host_name("m1_dot_product_v2_nt_bn128_col_vec4_bk128_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_vec4<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_vec4_bk128_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_vec4<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn128_col_vec4_bk64_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_vec4<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_vec4_bk64_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_vec4<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// Instantiate B-tiling variants
template [[host_name("m1_dot_product_v2_nt_bn128_col_bt_bk64_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_bt<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn128_col_bt_bk128_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_bt<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_bt_bk64_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_bt<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_bt_bk128_tg128")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_bt<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn128_col_bt_bk64_tg64")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_bt<64, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_col_bt_bk64_tg64")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_col_bt<64, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// Instantiate NT kernels for specific tile sizes (row-major indexing)
template [[host_name("m1_dot_product_v2_nt_bn128_row")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_row<128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v2_nt_bn64_row")]] [[kernel]]
void m1_dot_product_v2_nt_kernel_row<128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);
