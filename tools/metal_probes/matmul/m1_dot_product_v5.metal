// M=1 dot product v5 kernels
// Focused additions over v4:
//  - Ultra-tiny kernel (tg32, no TG memory, heavy unroll)
//  - Large-K/small-N aggressive K-parallel kernel (tg512 variants)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ===== Strategy 2B: Ultra-tiny (n<~1024, k<~1024) — absolute minimal overhead =====
// Each thread computes a single output column within its threadgroup slice.
// No threadgroup memory, manual unroll for ILP, scalar half->float FMAs.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v5_nt_kernel_ultra_tiny(
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
    const int base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    for (int co = lane; co < static_cast<int>(COLUMNS_PER_TG); co += THREADGROUP_WIDTH) {
        const int col = base_col + co;
        if (col >= N) { continue; }

        const device half* __restrict col_ptr = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;

        int k = 0;
        // 16-way unroll main loop
        for (; k + 15 < K; k += 16) {
            acc = fma(float(A[k    ]), float(col_ptr[k    ]), acc);
            acc = fma(float(A[k + 1]), float(col_ptr[k + 1]), acc);
            acc = fma(float(A[k + 2]), float(col_ptr[k + 2]), acc);
            acc = fma(float(A[k + 3]), float(col_ptr[k + 3]), acc);
            acc = fma(float(A[k + 4]), float(col_ptr[k + 4]), acc);
            acc = fma(float(A[k + 5]), float(col_ptr[k + 5]), acc);
            acc = fma(float(A[k + 6]), float(col_ptr[k + 6]), acc);
            acc = fma(float(A[k + 7]), float(col_ptr[k + 7]), acc);
            acc = fma(float(A[k + 8]), float(col_ptr[k + 8]), acc);
            acc = fma(float(A[k + 9]), float(col_ptr[k + 9]), acc);
            acc = fma(float(A[k + 10]), float(col_ptr[k + 10]), acc);
            acc = fma(float(A[k + 11]), float(col_ptr[k + 11]), acc);
            acc = fma(float(A[k + 12]), float(col_ptr[k + 12]), acc);
            acc = fma(float(A[k + 13]), float(col_ptr[k + 13]), acc);
            acc = fma(float(A[k + 14]), float(col_ptr[k + 14]), acc);
            acc = fma(float(A[k + 15]), float(col_ptr[k + 15]), acc);
        }

        // Optional 8-way tail
        if (k + 7 < K) {
            acc = fma(float(A[k    ]), float(col_ptr[k    ]), acc);
            acc = fma(float(A[k + 1]), float(col_ptr[k + 1]), acc);
            acc = fma(float(A[k + 2]), float(col_ptr[k + 2]), acc);
            acc = fma(float(A[k + 3]), float(col_ptr[k + 3]), acc);
            acc = fma(float(A[k + 4]), float(col_ptr[k + 4]), acc);
            acc = fma(float(A[k + 5]), float(col_ptr[k + 5]), acc);
            acc = fma(float(A[k + 6]), float(col_ptr[k + 6]), acc);
            acc = fma(float(A[k + 7]), float(col_ptr[k + 7]), acc);
            k += 8;
        }

        // Scalar tail
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(col_ptr[k]), acc);
        }

        C[col] = half(acc);
    }
}

// Ultra-tiny instantiation: TG=32, BN=32
template [[host_name("m1_dot_product_v5_nt_ultra_tiny_bn32_tg32")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_ultra_tiny<32, 32>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);


// Ultra-tiny single-TG variant: exactly one TG of 32 threads covers all N via grid-stride
template <ushort THREADGROUP_WIDTH>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v5_nt_kernel_ultra_tiny_single(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (M != 1) { return; }

    for (int col = static_cast<int>(tid); col < N; col += THREADGROUP_WIDTH) {
        const device half* __restrict col_ptr = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;
        int k = 0;
        for (; k + 15 < K; k += 16) {
            acc = fma(float(A[k    ]), float(col_ptr[k    ]), acc);
            acc = fma(float(A[k + 1]), float(col_ptr[k + 1]), acc);
            acc = fma(float(A[k + 2]), float(col_ptr[k + 2]), acc);
            acc = fma(float(A[k + 3]), float(col_ptr[k + 3]), acc);
            acc = fma(float(A[k + 4]), float(col_ptr[k + 4]), acc);
            acc = fma(float(A[k + 5]), float(col_ptr[k + 5]), acc);
            acc = fma(float(A[k + 6]), float(col_ptr[k + 6]), acc);
            acc = fma(float(A[k + 7]), float(col_ptr[k + 7]), acc);
            acc = fma(float(A[k + 8]), float(col_ptr[k + 8]), acc);
            acc = fma(float(A[k + 9]), float(col_ptr[k + 9]), acc);
            acc = fma(float(A[k + 10]), float(col_ptr[k + 10]), acc);
            acc = fma(float(A[k + 11]), float(col_ptr[k + 11]), acc);
            acc = fma(float(A[k + 12]), float(col_ptr[k + 12]), acc);
            acc = fma(float(A[k + 13]), float(col_ptr[k + 13]), acc);
            acc = fma(float(A[k + 14]), float(col_ptr[k + 14]), acc);
            acc = fma(float(A[k + 15]), float(col_ptr[k + 15]), acc);
        }
        if (k + 7 < K) {
            acc = fma(float(A[k    ]), float(col_ptr[k    ]), acc);
            acc = fma(float(A[k + 1]), float(col_ptr[k + 1]), acc);
            acc = fma(float(A[k + 2]), float(col_ptr[k + 2]), acc);
            acc = fma(float(A[k + 3]), float(col_ptr[k + 3]), acc);
            acc = fma(float(A[k + 4]), float(col_ptr[k + 4]), acc);
            acc = fma(float(A[k + 5]), float(col_ptr[k + 5]), acc);
            acc = fma(float(A[k + 6]), float(col_ptr[k + 6]), acc);
            acc = fma(float(A[k + 7]), float(col_ptr[k + 7]), acc);
            k += 8;
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(col_ptr[k]), acc);
        }
        C[col] = half(acc);
    }
}

template [[host_name("m1_dot_product_v5_nt_ultra_tiny_single_tg32")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_ultra_tiny_single<32>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint tid);

// ===== Strategy 3B: Large-K, small-N — K-parallel collaborative reduction (v4-proven semantics) =====
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v5_nt_kernel_largek_smalln(
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
    const int lanes  = 32;
    const int sgIndex = static_cast<int>(lid.x) >> 5; // derive SG index from thread index (robust across waves)
    const int group_base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    threadgroup float partial[COLUMNS_PER_TG][THREADGROUP_WIDTH / 32];
    threadgroup atomic_uint ordinalCounter;

    // Each SIMD group gets a slice of K
    const int k_per_group = (K + groups - 1) / groups;
    const int k_start = sgIndex * k_per_group;
    const int k_end = min(k_start + k_per_group, K);

    // Process a small pack of columns cooperatively (mirrors v4)
    for (int co = 0; co < static_cast<int>(COLUMNS_PER_TG); ++co) {
        const int col = group_base_col + co;
        if (col >= N) { continue; }

        const device half* __restrict column_ptr = B + static_cast<size_t>(col) * static_cast<size_t>(K);

        // Reset ordinal counter at start of each column
        if (lid.x == 0) {
            atomic_store_explicit(&ordinalCounter, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Assign a unique ordinal per SIMDGROUP using a threadgroup atomic; broadcast within SG
        uint myOrd = 0u;
        if (simd_lane == 0u) {
            myOrd = atomic_fetch_add_explicit(&ordinalCounter, 1u, memory_order_relaxed);
        }
        myOrd = simd_broadcast_first(myOrd);

        // Compute K slice from ordinal
        const int k_per_group = (K + groups - 1) / groups;
        const int k_start_ord = static_cast<int>(myOrd) * k_per_group;
        const int k_end_ord = min(k_start_ord + k_per_group, K);

        // Per-SIMDGROUP accumulation (use simd_sum across lanes)
        float local = 0.0f;
        for (int k = k_start_ord + static_cast<int>(simd_lane); k < k_end_ord; k += lanes) {
            local = fma(float(A[k]), float(column_ptr[k]), local);
        }
        float sg_sum = simd_sum(local);

        if (simd_lane == 0u) {
            partial[co][static_cast<int>(myOrd)] = sg_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across SIMD groups (use simd_group 0)
    if (sgIndex == 0) {
        for (int co = static_cast<int>(simd_lane); co < static_cast<int>(COLUMNS_PER_TG); co += lanes) {
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

// ===== Strategy 3B (Debug): Large-K, small-N — per-SIMDGROUP partial dump =====
template <ushort THREADGROUP_WIDTH>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v5_nt_kernel_largek_smalln_debug(
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
    const int lanes  = 32;
    const int sgIndex = static_cast<int>(lid.x) >> 5;
    threadgroup atomic_uint dbgOrdinal;

    // Debug column 0
    const int debug_col = 0;
    const device half* __restrict column_ptr = B + static_cast<size_t>(debug_col) * static_cast<size_t>(K);

    // Reset ordinal once
    if (lid.x == 0) { atomic_store_explicit(&dbgOrdinal, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Assign unique ordinal per SG
    uint ord = 0u;
    if (simd_lane == 0u) {
        ord = atomic_fetch_add_explicit(&dbgOrdinal, 1u, memory_order_relaxed);
    }
    ord = simd_broadcast_first(ord);

    // Each SIMD group integrates its K-slice based on ordinal
    const int k_per_group = (K + groups - 1) / groups;
    const int k_start = static_cast<int>(ord) * k_per_group;
    const int k_end = min(k_start + k_per_group, K);

    float sg_sum = 0.0f;
    uint sg_count = 0u;
    if (simd_lane == 0u) {
        for (int k = k_start; k < k_end; ++k) {
            sg_sum = fma(float(A[k]), float(column_ptr[k]), sg_sum);
            sg_count++;
        }
    }

    // Write per-SIMDGROUP partials into first `groups` outputs
    if (simd_lane == 0) {
        if (static_cast<int>(ord) < groups && static_cast<int>(ord) < N) {
            C[ord] = half(sg_sum);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute final reduction (group 0) and write to C[groups]
    if (sgIndex == 0 && simd_lane == 0) {
        float final_sum = 0.0f;
        for (int sg = 0; sg < groups; ++sg) {
            // Read back what was written by each SIMDGROUP
            final_sum += float(C[sg]);
        }
        if (groups + 1 < N) {
            C[groups] = half(final_sum);
        }

        // Also compute a full sequential reference on GPU for the debug column (by one thread)
        float ref = 0.0f;
        for (int k = 0; k < K; ++k) {
            ref = fma(float(A[k]), float(column_ptr[k]), ref);
        }
        if (groups + 2 < N) {
            C[groups + 1] = half(ref);
        }
        // Write expected per-group count (k_per_group) and observed total sum of counts
        if (groups + 3 < N) {
            C[groups + 2] = half(float(k_per_group));
        }
    }

    // Write per-SIMDGROUP processed counts to C[groups*2 + ord]
    if (simd_lane == 0) {
        int dst = groups * 2 + static_cast<int>(ord);
        if (dst < N) { C[dst] = half(float(sg_count)); }
    }
}

// Debug instantiations (single TG runs expected)
template [[host_name("m1_dot_product_v5_nt_dbg_largek_smalln_single_tg512")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln_debug<512>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

template [[host_name("m1_dot_product_v5_nt_dbg_largek_smalln_single_tg256")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln_debug<256>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

// Aggressive large-K small-N instantiations (tg256 – safe across devices)
template [[host_name("m1_dot_product_v5_nt_bn8_largek_smalln_tg256")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln<256, 8>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

template [[host_name("m1_dot_product_v5_nt_bn16_largek_smalln_tg256")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln<256, 16>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

template [[host_name("m1_dot_product_v5_nt_bn4_largek_smalln_tg256")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln<256, 4>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

// Aggressive large-K small-N instantiations (tg512 – re-enabled; Macs support 1024 max threads)
template [[host_name("m1_dot_product_v5_nt_bn8_largek_smalln_tg512")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln<512, 8>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

template [[host_name("m1_dot_product_v5_nt_bn16_largek_smalln_tg512")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln<512, 16>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);

template [[host_name("m1_dot_product_v5_nt_bn4_largek_smalln_tg512")]] [[kernel]]
void m1_dot_product_v5_nt_kernel_largek_smalln<512, 4>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint simd_lane, uint simd_group);
