// M=1 dot product v4 kernels
// Three strategies:
// 1) Large-N (bn256) vec4 tgread
// 2) Tiny shapes (single-TG minimal-overhead)
// 3) Large-K, small-N: collaborative K-parallel reduction

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ===== Strategy 1: Large N (bn256) — vec4 tgread (A-tiling + B half4) =====
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v4_nt_kernel_col_vec4_tgread(
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
    int k_base = 0, cur = 0, next = 1;

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
                if (active0) acc0 = fma(av, float(column_ptr0[k_base + kk]), acc0);
                if (active1) acc1 = fma(av, float(column_ptr1[k_base + kk]), acc1);
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

// bn256 instantiation (tg128, bk64)
template [[host_name("m1_dot_product_v4_nt_bn256_col_vec4_tgread_bk64_tg128")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_col_vec4_tgread<128, 256, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// bn256 instantiation (tg128, bk128)
template [[host_name("m1_dot_product_v4_nt_bn256_col_vec4_tgread_bk128_tg128")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_col_vec4_tgread<128, 256, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// ===== Strategy 2: Tiny shapes (single-TG friendly) =====
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v4_nt_kernel_tiny(
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

    // Each lane handles columns within this TG's slice
    for (int co = lane; co < static_cast<int>(COLUMNS_PER_TG); co += THREADGROUP_WIDTH) {
        const int col = group_base_col + co;
        if (col >= N) continue;
        const device half* __restrict column_ptr = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;
        int k = 0;
        for (; k + 7 < K; k += 8) {
            acc = fma(float(A[k    ]), float(column_ptr[k    ]), acc);
            acc = fma(float(A[k + 1]), float(column_ptr[k + 1]), acc);
            acc = fma(float(A[k + 2]), float(column_ptr[k + 2]), acc);
            acc = fma(float(A[k + 3]), float(column_ptr[k + 3]), acc);
            acc = fma(float(A[k + 4]), float(column_ptr[k + 4]), acc);
            acc = fma(float(A[k + 5]), float(column_ptr[k + 5]), acc);
            acc = fma(float(A[k + 6]), float(column_ptr[k + 6]), acc);
            acc = fma(float(A[k + 7]), float(column_ptr[k + 7]), acc);
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(column_ptr[k]), acc);
        }
        C[col] = half(acc);
    }
}

// Tiny instantiation: TG=64 threads, 64 columns per TG
template [[host_name("m1_dot_product_v4_nt_tiny_bn64_tg64")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_tiny<64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// Tiny instantiation: TG=64, 32 columns per TG (for very small N)
template [[host_name("m1_dot_product_v4_nt_tiny_bn32_tg64")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_tiny<64, 32>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// ===== Strategy 3: Large-K, small-N — K-parallel collaborative reduction =====
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v4_nt_kernel_largek_smalln(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
    if (M != 1) { return; }

    const int groups = THREADGROUP_WIDTH / 32; // SIMD groups per TG
    const int group_base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    threadgroup float partial[COLUMNS_PER_TG][THREADGROUP_WIDTH / 32];

    // Each SIMD group gets a slice of K
    const int k_per_group = (K + groups - 1) / groups;
    const int k_start = static_cast<int>(simd_group) * k_per_group;
    const int k_end = min(k_start + k_per_group, K);

    // Process a small pack of columns cooperatively
    for (int co = 0; co < static_cast<int>(COLUMNS_PER_TG); ++co) {
        const int col = group_base_col + co;
        if (col >= N) { continue; }

        const device half* __restrict column_ptr = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float local = 0.0f;
        // Stride by lane across the K-slice
        for (int k = k_start + static_cast<int>(simd_lane); k < k_end; k += 32) {
            local = fma(float(A[k]), float(column_ptr[k]), local);
        }
        // Reduce within SIMD group
        float sg_sum = simd_sum(local);
        if (simd_lane == 0) {
            partial[co][simd_group] = sg_sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups (use simd_group 0)
    if (simd_group == 0) {
        for (int co = static_cast<int>(simd_lane); co < static_cast<int>(COLUMNS_PER_TG); co += 32) {
            const int col = group_base_col + co;
            if (col >= N) continue;
            float final_sum = 0.0f;
            for (int sg = 0; sg < groups; ++sg) {
                final_sum += partial[co][sg];
            }
            C[col] = half(final_sum);
        }
    }
}

// Large-K small-N instantiation: TG=256 threads (8 SIMDs), 8 columns per TG
template [[host_name("m1_dot_product_v4_nt_bn8_largek_smalln_tg256")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_largek_smalln<256, 8>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

// Large-K small-N: TG=128, 8 columns per TG (compare occupancy)
template [[host_name("m1_dot_product_v4_nt_bn8_largek_smalln_tg128")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_largek_smalln<128, 8>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

// Large-K small-N: TG=256, 16 columns per TG
template [[host_name("m1_dot_product_v4_nt_bn16_largek_smalln_tg256")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_largek_smalln<256, 16>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

// Large-K small-N: TG=256, 4 columns per TG
template [[host_name("m1_dot_product_v4_nt_bn4_largek_smalln_tg256")]] [[kernel]]
void m1_dot_product_v4_nt_kernel_largek_smalln<256, 4>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);
