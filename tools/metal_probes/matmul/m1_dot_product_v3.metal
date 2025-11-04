// M=1 dot product v3 kernels
// Focus: NT layout, A tiling with simdgroup broadcast of A tile reads,
// vectorized B loads (half4), and up to two columns accumulated per thread.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Column-major (NT) variant with A-tiling (double-buffered) and simdgroup broadcast
// for A tile reads to reduce threadgroup-memory traffic and bank conflicts.
// B is streamed from device in half4 where aligned; each thread accumulates up to
// two columns (lane and lane+THREADGROUP_WIDTH).
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]]) {

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

    // Preload first A tile cooperatively across TG
    {
        const int tile_len0 = (K - k_base) > BK ? BK : max(0, K - k_base);
        for (int kk = lane; kk < tile_len0; kk += THREADGROUP_WIDTH) {
            a_tile_buf[cur][kk] = A[k_base + kk];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (k_base < K) {
        const int tile_len = (K - k_base) > BK ? BK : max(0, K - k_base);

        // Prefetch next A tile into alternate buffer
        const int next_base = k_base + BK;
        if (next_base < K) {
            const int next_len = (K - next_base) > BK ? BK : max(0, K - next_base);
            for (int kk = lane; kk < next_len; kk += THREADGROUP_WIDTH) {
                a_tile_buf[next][kk] = A[next_base + kk];
            }
        }

        if (active0 || active1) {
            int kk = 0;

            // Scalar head until k is 4-aligned (for half4 B loads). Broadcast half from lane 0, then convert.
            for (; kk < tile_len && ((k_base + kk) & 3) != 0; ++kk) {
                thread half a_h = half(0.0h);
                if (lane_id == 0) { a_h = a_tile_buf[cur][kk]; }
                const half a_hb = simd_broadcast(a_h, 0);
                const float av = float(a_hb);
                if (active0) { acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0); }
                if (active1) { acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1); }
            }

            // Vectorized core: unroll by 8 using two half4 loads; A via simdgroup broadcast (half4 packed).
            for (; kk + 7 < tile_len; kk += 8) {
                // Lane 0 per simdgroup packs half4 blocks and broadcasts to all lanes
                thread half4 a_pack0 = half4(0.0h);
                thread half4 a_pack1 = half4(0.0h);
                if (lane_id == 0) {
                    a_pack0 = half4(a_tile_buf[cur][kk    ], a_tile_buf[cur][kk + 1], a_tile_buf[cur][kk + 2], a_tile_buf[cur][kk + 3]);
                    a_pack1 = half4(a_tile_buf[cur][kk + 4], a_tile_buf[cur][kk + 5], a_tile_buf[cur][kk + 6], a_tile_buf[cur][kk + 7]);
                }
                const half4 a_b0 = simd_broadcast(a_pack0, 0);
                const half4 a_b1 = simd_broadcast(a_pack1, 0);
                const float4 av0 = float4(a_b0);
                const float4 av1 = float4(a_b1);

                const int k0 = k_base + kk;
                if (active0) {
                    const device half4* p0 = reinterpret_cast<const device half4*>(column_ptr0 + k0);
                    const half4 hb0 = *p0;
                    const half4 hb1 = *(p0 + 1);
                    const float4 fb0 = float4(hb0);
                    const float4 fb1 = float4(hb1);
                    acc0 = fma(av0.x, fb0.x, acc0);
                    acc0 = fma(av0.y, fb0.y, acc0);
                    acc0 = fma(av0.z, fb0.z, acc0);
                    acc0 = fma(av0.w, fb0.w, acc0);
                    acc0 = fma(av1.x, fb1.x, acc0);
                    acc0 = fma(av1.y, fb1.y, acc0);
                    acc0 = fma(av1.z, fb1.z, acc0);
                    acc0 = fma(av1.w, fb1.w, acc0);
                }
                if (active1) {
                    const device half4* p1 = reinterpret_cast<const device half4*>(column_ptr1 + k0);
                    const half4 hb0 = *p1;
                    const half4 hb1 = *(p1 + 1);
                    const float4 fb0 = float4(hb0);
                    const float4 fb1 = float4(hb1);
                    acc1 = fma(av0.x, fb0.x, acc1);
                    acc1 = fma(av0.y, fb0.y, acc1);
                    acc1 = fma(av0.z, fb0.z, acc1);
                    acc1 = fma(av0.w, fb0.w, acc1);
                    acc1 = fma(av1.x, fb1.x, acc1);
                    acc1 = fma(av1.y, fb1.y, acc1);
                    acc1 = fma(av1.z, fb1.z, acc1);
                    acc1 = fma(av1.w, fb1.w, acc1);
                }
            }

            // Scalar tail of current tile; A via simd_broadcast from lane 0
            for (; kk < tile_len; ++kk) {
                thread half a_h = half(0.0h);
                if (lane_id == 0) { a_h = a_tile_buf[cur][kk]; }
                const half a_hb = simd_broadcast(a_h, 0);
                const float av = float(a_hb);
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

// Column-major (NT) variant with A-tiling (double-buffered) but without explicit
// simdgroup broadcasts for A. Each lane reads A from threadgroup memory directly
// (Apple GPUs often broadcast TG reads efficiently in hardware), while B is streamed
// as half4. Each thread accumulates up to two columns.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread(
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

    // Preload first A tile cooperatively
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
            // Scalar head until k is 4-aligned (for half4 B loads)
            for (; kk < tile_len && ((k_base + kk) & 3) != 0; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) { acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0); }
                if (active1) { acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1); }
            }

            // Vectorized core: unroll by 8 using two half4 loads for B and scalar A reads
            for (; kk + 7 < tile_len; kk += 8) {
                // Load 8 A scalars from TG
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
                    const float4 fb0 = float4(*p0);
                    const float4 fb1 = float4(*(p0 + 1));
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
                    const float4 fb0 = float4(*p1);
                    const float4 fb1 = float4(*(p1 + 1));
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

            // Scalar tail
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

// TG-read variant with vectorized A loads (half4) in the aligned core.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA(
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

    // Preload first A tile cooperatively
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
            // Scalar head until k is 4-aligned (for half4 vector loads)
            for (; kk < tile_len && ((k_base + kk) & 3) != 0; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) { acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0); }
                if (active1) { acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1); }
            }

            // Vectorized core: unroll by 8 using two half4 loads for A (manually packed) and B
            for (; kk + 7 < tile_len; kk += 8) {
                const int k0 = k_base + kk;
                const float4 av0 = float4(
                    float(a_tile_buf[cur][kk    ]),
                    float(a_tile_buf[cur][kk + 1]),
                    float(a_tile_buf[cur][kk + 2]),
                    float(a_tile_buf[cur][kk + 3])
                );
                const float4 av1 = float4(
                    float(a_tile_buf[cur][kk + 4]),
                    float(a_tile_buf[cur][kk + 5]),
                    float(a_tile_buf[cur][kk + 6]),
                    float(a_tile_buf[cur][kk + 7])
                );

                if (active0) {
                    const device half4* p0 = reinterpret_cast<const device half4*>(column_ptr0 + k0);
                    const float4 fb0 = float4(*p0);
                    const float4 fb1 = float4(*(p0 + 1));
                    acc0 = fma(av0.x, fb0.x, acc0);
                    acc0 = fma(av0.y, fb0.y, acc0);
                    acc0 = fma(av0.z, fb0.z, acc0);
                    acc0 = fma(av0.w, fb0.w, acc0);
                    acc0 = fma(av1.x, fb1.x, acc0);
                    acc0 = fma(av1.y, fb1.y, acc0);
                    acc0 = fma(av1.z, fb1.z, acc0);
                    acc0 = fma(av1.w, fb1.w, acc0);
                }
                if (active1) {
                    const device half4* p1 = reinterpret_cast<const device half4*>(column_ptr1 + k0);
                    const float4 fb0 = float4(*p1);
                    const float4 fb1 = float4(*(p1 + 1));
                    acc1 = fma(av0.x, fb0.x, acc1);
                    acc1 = fma(av0.y, fb0.y, acc1);
                    acc1 = fma(av0.z, fb0.z, acc1);
                    acc1 = fma(av0.w, fb0.w, acc1);
                    acc1 = fma(av1.x, fb1.x, acc1);
                    acc1 = fma(av1.y, fb1.y, acc1);
                    acc1 = fma(av1.z, fb1.z, acc1);
                    acc1 = fma(av1.w, fb1.w, acc1);
                }
            }

            // Scalar tail
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

// tgread unroll16 (process 16 K per iteration)
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_unroll16(
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
    float acc0 = 0.0f, acc1 = 0.0f;
    const device half* __restrict column_ptr0 = active0 ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict column_ptr1 = active1 ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;
    threadgroup half a_tile_buf[2][BK];
    int k_base = 0, cur = 0, next = 1;
    {
        const int tile_len0 = (K - k_base) > BK ? BK : max(0, K - k_base);
        for (int kk = lane; kk < tile_len0; kk += THREADGROUP_WIDTH) a_tile_buf[cur][kk] = A[k_base + kk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while (k_base < K) {
        const int tile_len = (K - k_base) > BK ? BK : max(0, K - k_base);
        const int next_base = k_base + BK;
        if (next_base < K) {
            const int next_len = (K - next_base) > BK ? BK : max(0, K - next_base);
            for (int kk = lane; kk < next_len; kk += THREADGROUP_WIDTH) a_tile_buf[next][kk] = A[next_base + kk];
        }
        if (active0 || active1) {
            int kk = 0;
            for (; kk < tile_len && ((k_base + kk) & 3) != 0; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0);
                if (active1) acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1);
            }
            for (; kk + 15 < tile_len; kk += 16) {
                const int k0 = k_base + kk;
                // Load 16 A scalars
                float a[16];
                a[0] = float(a_tile_buf[cur][kk    ]); a[1] = float(a_tile_buf[cur][kk + 1]);
                a[2] = float(a_tile_buf[cur][kk + 2]); a[3] = float(a_tile_buf[cur][kk + 3]);
                a[4] = float(a_tile_buf[cur][kk + 4]); a[5] = float(a_tile_buf[cur][kk + 5]);
                a[6] = float(a_tile_buf[cur][kk + 6]); a[7] = float(a_tile_buf[cur][kk + 7]);
                a[8] = float(a_tile_buf[cur][kk + 8]); a[9] = float(a_tile_buf[cur][kk + 9]);
                a[10]= float(a_tile_buf[cur][kk +10]); a[11]= float(a_tile_buf[cur][kk +11]);
                a[12]= float(a_tile_buf[cur][kk +12]); a[13]= float(a_tile_buf[cur][kk +13]);
                a[14]= float(a_tile_buf[cur][kk +14]); a[15]= float(a_tile_buf[cur][kk +15]);
                if (active0) {
                    const device half4* p0 = reinterpret_cast<const device half4*>(column_ptr0 + k0);
                    const float4 fb0 = float4(*p0);
                    const float4 fb1 = float4(*(p0 + 1));
                    const float4 fb2 = float4(*(p0 + 2));
                    const float4 fb3 = float4(*(p0 + 3));
                    acc0 = fma(a[0], fb0.x, acc0);  acc0 = fma(a[1], fb0.y, acc0);
                    acc0 = fma(a[2], fb0.z, acc0);  acc0 = fma(a[3], fb0.w, acc0);
                    acc0 = fma(a[4], fb1.x, acc0);  acc0 = fma(a[5], fb1.y, acc0);
                    acc0 = fma(a[6], fb1.z, acc0);  acc0 = fma(a[7], fb1.w, acc0);
                    acc0 = fma(a[8], fb2.x, acc0);  acc0 = fma(a[9], fb2.y, acc0);
                    acc0 = fma(a[10],fb2.z, acc0);  acc0 = fma(a[11],fb2.w, acc0);
                    acc0 = fma(a[12],fb3.x, acc0);  acc0 = fma(a[13],fb3.y, acc0);
                    acc0 = fma(a[14],fb3.z, acc0);  acc0 = fma(a[15],fb3.w, acc0);
                }
                if (active1) {
                    const device half4* p1 = reinterpret_cast<const device half4*>(column_ptr1 + k0);
                    const float4 fb0 = float4(*p1);
                    const float4 fb1 = float4(*(p1 + 1));
                    const float4 fb2 = float4(*(p1 + 2));
                    const float4 fb3 = float4(*(p1 + 3));
                    acc1 = fma(a[0], fb0.x, acc1);  acc1 = fma(a[1], fb0.y, acc1);
                    acc1 = fma(a[2], fb0.z, acc1);  acc1 = fma(a[3], fb0.w, acc1);
                    acc1 = fma(a[4], fb1.x, acc1);  acc1 = fma(a[5], fb1.y, acc1);
                    acc1 = fma(a[6], fb1.z, acc1);  acc1 = fma(a[7], fb1.w, acc1);
                    acc1 = fma(a[8], fb2.x, acc1);  acc1 = fma(a[9], fb2.y, acc1);
                    acc1 = fma(a[10],fb2.z, acc1);  acc1 = fma(a[11],fb2.w, acc1);
                    acc1 = fma(a[12],fb3.x, acc1);  acc1 = fma(a[13],fb3.y, acc1);
                    acc1 = fma(a[14],fb3.z, acc1);  acc1 = fma(a[15],fb3.w, acc1);
                }
            }
            for (; kk < tile_len; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0);
                if (active1) acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += BK; cur ^= 1; next ^= 1;
    }
    if (active0) C[col0] = half(acc0);
    if (active1) C[col1] = half(acc1);
}

// tgread_vA unroll16 (process 16 K per iteration with vector A)
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA_unroll16(
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
    float acc0 = 0.0f, acc1 = 0.0f;
    const device half* __restrict column_ptr0 = active0 ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict column_ptr1 = active1 ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;
    threadgroup half a_tile_buf[2][BK];
    int k_base = 0, cur = 0, next = 1;
    {
        const int tile_len0 = (K - k_base) > BK ? BK : max(0, K - k_base);
        for (int kk = lane; kk < tile_len0; kk += THREADGROUP_WIDTH) a_tile_buf[cur][kk] = A[k_base + kk];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    while (k_base < K) {
        const int tile_len = (K - k_base) > BK ? BK : max(0, K - k_base);
        const int next_base = k_base + BK;
        if (next_base < K) {
            const int next_len = (K - next_base) > BK ? BK : max(0, K - next_base);
            for (int kk = lane; kk < next_len; kk += THREADGROUP_WIDTH) a_tile_buf[next][kk] = A[next_base + kk];
        }
        if (active0 || active1) {
            int kk = 0;
            for (; kk < tile_len && ((k_base + kk) & 3) != 0; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0);
                if (active1) acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1);
            }
            for (; kk + 15 < tile_len; kk += 16) {
                const int k0 = k_base + kk;
                const float4 av0 = float4(
                    float(a_tile_buf[cur][kk     ]),
                    float(a_tile_buf[cur][kk +  1]),
                    float(a_tile_buf[cur][kk +  2]),
                    float(a_tile_buf[cur][kk +  3])
                );
                const float4 av1 = float4(
                    float(a_tile_buf[cur][kk +  4]),
                    float(a_tile_buf[cur][kk +  5]),
                    float(a_tile_buf[cur][kk +  6]),
                    float(a_tile_buf[cur][kk +  7])
                );
                const float4 av2 = float4(
                    float(a_tile_buf[cur][kk +  8]),
                    float(a_tile_buf[cur][kk +  9]),
                    float(a_tile_buf[cur][kk + 10]),
                    float(a_tile_buf[cur][kk + 11])
                );
                const float4 av3 = float4(
                    float(a_tile_buf[cur][kk + 12]),
                    float(a_tile_buf[cur][kk + 13]),
                    float(a_tile_buf[cur][kk + 14]),
                    float(a_tile_buf[cur][kk + 15])
                );
                if (active0) {
                    const device half4* p0 = reinterpret_cast<const device half4*>(column_ptr0 + k0);
                    const float4 fb0 = float4(*p0);
                    const float4 fb1 = float4(*(p0 + 1));
                    const float4 fb2 = float4(*(p0 + 2));
                    const float4 fb3 = float4(*(p0 + 3));
                    acc0 = fma(av0.x, fb0.x, acc0); acc0 = fma(av0.y, fb0.y, acc0);
                    acc0 = fma(av0.z, fb0.z, acc0); acc0 = fma(av0.w, fb0.w, acc0);
                    acc0 = fma(av1.x, fb1.x, acc0); acc0 = fma(av1.y, fb1.y, acc0);
                    acc0 = fma(av1.z, fb1.z, acc0); acc0 = fma(av1.w, fb1.w, acc0);
                    acc0 = fma(av2.x, fb2.x, acc0); acc0 = fma(av2.y, fb2.y, acc0);
                    acc0 = fma(av2.z, fb2.z, acc0); acc0 = fma(av2.w, fb2.w, acc0);
                    acc0 = fma(av3.x, fb3.x, acc0); acc0 = fma(av3.y, fb3.y, acc0);
                    acc0 = fma(av3.z, fb3.z, acc0); acc0 = fma(av3.w, fb3.w, acc0);
                }
                if (active1) {
                    const device half4* p1 = reinterpret_cast<const device half4*>(column_ptr1 + k0);
                    const float4 fb0 = float4(*p1);
                    const float4 fb1 = float4(*(p1 + 1));
                    const float4 fb2 = float4(*(p1 + 2));
                    const float4 fb3 = float4(*(p1 + 3));
                    acc1 = fma(av0.x, fb0.x, acc1); acc1 = fma(av0.y, fb0.y, acc1);
                    acc1 = fma(av0.z, fb0.z, acc1); acc1 = fma(av0.w, fb0.w, acc1);
                    acc1 = fma(av1.x, fb1.x, acc1); acc1 = fma(av1.y, fb1.y, acc1);
                    acc1 = fma(av1.z, fb1.z, acc1); acc1 = fma(av1.w, fb1.w, acc1);
                    acc1 = fma(av2.x, fb2.x, acc1); acc1 = fma(av2.y, fb2.y, acc1);
                    acc1 = fma(av2.z, fb2.z, acc1); acc1 = fma(av2.w, fb2.w, acc1);
                    acc1 = fma(av3.x, fb3.x, acc1); acc1 = fma(av3.y, fb3.y, acc1);
                    acc1 = fma(av3.z, fb3.z, acc1); acc1 = fma(av3.w, fb3.w, acc1);
                }
            }
            for (; kk < tile_len; ++kk) {
                const float av = float(a_tile_buf[cur][kk]);
                if (active0) acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0);
                if (active1) acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += BK; cur ^= 1; next ^= 1;
    }
    if (active0) C[col0] = half(acc0);
    if (active1) C[col1] = half(acc1);
}
// Instantiations for common occupancy variants (TG=128, BN in {64,128}, BK in {64,128})
template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_sgbr_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint lane_id);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_sgbr_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint lane_id);

template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_sgbr_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint lane_id);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_sgbr_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint lane_id);

// bn256 column tile variants (tg128)
template [[host_name("m1_dot_product_v3_nt_bn256_col_vec4_sgbr_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr<128, 256, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint lane_id);

template [[host_name("m1_dot_product_v3_nt_bn256_col_vec4_sgbr_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_sgbr<128, 256, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint lane_id);

// tgread instantiations (no explicit simdgroup broadcast for A)
template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// tg64 occupancy variants (tgread)
template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_bk64_tg64")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread<64, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_bk64_tg64")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread<64, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// tgread_vA instantiations (A vectorized loads from threadgroup memory)
template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// tg64 occupancy variants (tgread_vA)
template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_vA_bk64_tg64")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA<64, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_vA_bk64_tg64")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA<64, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// unroll16 instantiations (tg128)
template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_unroll16_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_unroll16<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_unroll16_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_unroll16<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn128_col_vec4_tgread_vA_unroll16_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA_unroll16<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v3_nt_bn64_col_vec4_tgread_vA_unroll16_bk64_tg128")]] [[kernel]]
void m1_dot_product_v3_nt_kernel_col_vec4_tgread_vA_unroll16<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);
