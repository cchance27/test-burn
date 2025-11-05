// M=1 dot product v6 kernels
// Focus areas:
//  - Large-K small-N: K-parallel with one final threadgroup reduction (single barrier)
//  - Large-N: vec4 tgread with A staged in threadgroup memory (vA)

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ===== Strategy A: Large-K, small-N — K-parallel with hierarchical final reduction =====
// Each simdgroup processes disjoint K-slices and accumulates per-column results in registers.
// After finishing K, a single barrier allows SG0 to reduce across groups and write C.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    if (M != 1) { return; }

    // Use lane index within the current SIMD group (0..31)
    const ushort lane = static_cast<ushort>(simd_lane);
    const int base_col = static_cast<int>(gid.x) * static_cast<int>(COLUMNS_PER_TG);

    // Each lane handles columns with stride 32; for COLUMNS_PER_TG < 32 many lanes will be inactive.
    // We accumulate per-column floats in registers.
    float accs[4];
    accs[0] = 0.0f; accs[1] = 0.0f; accs[2] = 0.0f; accs[3] = 0.0f;

    const int SIMD_SIZE = 32;
    const int NUM_SG = THREADGROUP_WIDTH / SIMD_SIZE;

    // Map up to 4 columns per lane (when COLUMNS_PER_TG <= 128). Use scalar stride across 32.
    int cols[4];
    cols[0] = base_col + static_cast<int>(lane);
    cols[1] = base_col + static_cast<int>(lane) + SIMD_SIZE;
    cols[2] = base_col + static_cast<int>(lane) + 2 * SIMD_SIZE;
    cols[3] = base_col + static_cast<int>(lane) + 3 * SIMD_SIZE;
    bool active[4];
    active[0] = (lane < COLUMNS_PER_TG) && (cols[0] < N);
    active[1] = (lane + SIMD_SIZE < COLUMNS_PER_TG) && (cols[1] < N);
    active[2] = (lane + 2 * SIMD_SIZE < COLUMNS_PER_TG) && (cols[2] < N);
    active[3] = (lane + 3 * SIMD_SIZE < COLUMNS_PER_TG) && (cols[3] < N);

    // K-parallel sweep: each SG takes every NUM_SG-th BK tile along K
    for (int kStart = sg_id * BK; kStart < K; kStart += BK * NUM_SG) {
        // Vectorized inner loop over BK with head/tail guards (half4 when aligned)
        int k = kStart;
        // Head align to 4
        while ((k < (kStart + BK)) && (k < K) && ((k & 3) != 0)) {
            half a_h = A[k];
            float a = float(a_h);
            if (active[0]) { accs[0] = fma(a, float(B[static_cast<size_t>(cols[0]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[0]); }
            if (active[1]) { accs[1] = fma(a, float(B[static_cast<size_t>(cols[1]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[1]); }
            if (active[2]) { accs[2] = fma(a, float(B[static_cast<size_t>(cols[2]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[2]); }
            if (active[3]) { accs[3] = fma(a, float(B[static_cast<size_t>(cols[3]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[3]); }
            ++k;
        }
        // Vector body
        for (; (k + 3) < (kStart + BK) && (k + 3) < K; k += 4) {
            const device half4* pA = reinterpret_cast<const device half4*>(A + k);
            half4 a4 = *pA;
            const float4 af = float4(a4);
            if (active[0]) {
                const device half4* pB0 = reinterpret_cast<const device half4*>(B + static_cast<size_t>(cols[0]) * static_cast<size_t>(K) + static_cast<size_t>(k));
                half4 b4 = *pB0;
                accs[0] += dot(af, float4(b4));
            }
            if (active[1]) {
                const device half4* pB1 = reinterpret_cast<const device half4*>(B + static_cast<size_t>(cols[1]) * static_cast<size_t>(K) + static_cast<size_t>(k));
                half4 b4 = *pB1;
                accs[1] += dot(af, float4(b4));
            }
            if (active[2]) {
                const device half4* pB2 = reinterpret_cast<const device half4*>(B + static_cast<size_t>(cols[2]) * static_cast<size_t>(K) + static_cast<size_t>(k));
                half4 b4 = *pB2;
                accs[2] += dot(af, float4(b4));
            }
            if (active[3]) {
                const device half4* pB3 = reinterpret_cast<const device half4*>(B + static_cast<size_t>(cols[3]) * static_cast<size_t>(K) + static_cast<size_t>(k));
                half4 b4 = *pB3;
                accs[3] += dot(af, float4(b4));
            }
        }
        // Tail
        while (k < (kStart + BK) && k < K) {
            half a_h = A[k];
            float a = float(a_h);
            if (active[0]) { accs[0] = fma(a, float(B[static_cast<size_t>(cols[0]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[0]); }
            if (active[1]) { accs[1] = fma(a, float(B[static_cast<size_t>(cols[1]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[1]); }
            if (active[2]) { accs[2] = fma(a, float(B[static_cast<size_t>(cols[2]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[2]); }
            if (active[3]) { accs[3] = fma(a, float(B[static_cast<size_t>(cols[3]) * static_cast<size_t>(K) + static_cast<size_t>(k)]), accs[3]); }
            ++k;
        }
    }

    // One final barrier; SG0 reduces across SGs
    // Flattened reduction buffer to avoid any 2D aliasing quirks
    threadgroup float tg_acc[ (THREADGROUP_WIDTH / 32) * COLUMNS_PER_TG ];
    // Compute base index for this SG in the flattened buffer and pre-clear active slots
    const int sg_base = static_cast<int>(sg_id) * static_cast<int>(COLUMNS_PER_TG);
    if (active[0]) { tg_acc[ sg_base + static_cast<int>(lane) ] = 0.0f; }
    if (active[1]) { tg_acc[ sg_base + static_cast<int>(lane) + SIMD_SIZE ] = 0.0f; }
    if (active[2]) { tg_acc[ sg_base + static_cast<int>(lane) + 2 * SIMD_SIZE ] = 0.0f; }
    if (active[3]) { tg_acc[ sg_base + static_cast<int>(lane) + 3 * SIMD_SIZE ] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Write per-SG accumulations
    if (active[0]) { tg_acc[ sg_base + static_cast<int>(lane) ] = accs[0]; }
    if (active[1]) { tg_acc[ sg_base + static_cast<int>(lane) + SIMD_SIZE ] = accs[1]; }
    if (active[2]) { tg_acc[ sg_base + static_cast<int>(lane) + 2 * SIMD_SIZE ] = accs[2]; }
    if (active[3]) { tg_acc[ sg_base + static_cast<int>(lane) + 3 * SIMD_SIZE ] = accs[3]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // SG0: lanes sum across SG dimension and write out
    if (sg_id == 0) {
        // Each lane handles up to 4 columns
        for (ushort i = 0; i < 4; ++i) {
            int col = cols[i];
            bool act = active[i];
            if (!act) { continue; }
            float s = 0.0f;
            for (int g = 0; g < NUM_SG; ++g) {
                // Index back to the right column slot in flattened buffer
                int colSlot = static_cast<int>(lane) + static_cast<int>(i) * SIMD_SIZE;
                const int g_base = g * static_cast<int>(COLUMNS_PER_TG);
                s += tg_acc[ g_base + colSlot ];
            }
            if (col < N) {
                C[col] = half(s);
            }
        }
    }
}

// ===== Strategy B: Large-N — vec4 tgread with A staged in threadgroup memory (vA) =====
// Each TG processes a column tile; threads cooperatively stage A tile into TGM; B read in half4 and accumulate.
template <ushort THREADGROUP_WIDTH, ushort COLUMNS_PER_TG, ushort BK>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_WIDTH)]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA(
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

    // Up to two columns per lane (COLUMNS_PER_TG may exceed THREADGROUP_WIDTH)
    const int col0 = group_base_col + static_cast<int>(lane);
    const int col1 = group_base_col + static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH);
    const bool has1 = (static_cast<int>(lane) + static_cast<int>(THREADGROUP_WIDTH)) < static_cast<int>(COLUMNS_PER_TG);
    const bool active0 = (static_cast<int>(lane) < static_cast<int>(COLUMNS_PER_TG)) && (col0 < N);
    const bool active1 = has1 && (col1 < N);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup half tgA[BK];

    for (int kt = 0; kt < K; kt += BK) {
        // Stage A[kt:kt+BK)
        // Cooperative load by lanes; vectorize when aligned
        int lk = static_cast<int>(lane);
        // Head
        while (lk < BK && ((lk & 3) != 0)) {
            int gk = kt + lk;
            tgA[lk] = (gk < K) ? A[gk] : half(0.0h);
            lk += THREADGROUP_WIDTH;
        }
        // Vector body
        for (; (lk + 3) < BK; lk += 4) {
            const int gk = kt + lk;
            half4 v = half4(0.0h);
            if ((gk + 3) < K) {
                const device half4* pA = reinterpret_cast<const device half4*>(A + gk);
                v = *pA;
            } else {
                // Safe tail fill
                v[0] = (gk + 0) < K ? A[gk + 0] : half(0.0h);
                v[1] = (gk + 1) < K ? A[gk + 1] : half(0.0h);
                v[2] = (gk + 2) < K ? A[gk + 2] : half(0.0h);
                v[3] = (gk + 3) < K ? A[gk + 3] : half(0.0h);
            }
            // Scatter to TG
            tgA[lk + 0] = v[0];
            tgA[lk + 1] = v[1];
            tgA[lk + 2] = v[2];
            tgA[lk + 3] = v[3];
        }
        // Tail
        while (lk < BK) {
            int gk = kt + lk;
            tgA[lk] = (gk < K) ? A[gk] : half(0.0h);
            lk += THREADGROUP_WIDTH;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Stream B for col0/col1 and accumulate
        if (active0) {
            const device half* __restrict colB = B + static_cast<size_t>(col0) * static_cast<size_t>(K);
            int k = 0;
            // Head
            while ((k < BK) && ((k & 3) != 0)) {
                acc0 = fma(float(tgA[k]), float(colB[kt + k]), acc0);
                ++k;
            }
            // Body
            for (; (k + 3) < BK; k += 4) {
                const half4 a4 = half4(tgA[k + 0], tgA[k + 1], tgA[k + 2], tgA[k + 3]);
                const device half4* pB = reinterpret_cast<const device half4*>(colB + kt + k);
                half4 b4 = *pB;
                acc0 += dot(float4(a4), float4(b4));
            }
            // Tail
            while (k < BK) {
                acc0 = fma(float(tgA[k]), float(colB[kt + k]), acc0);
                ++k;
            }
        }
        if (active1) {
            const device half* __restrict colB = B + static_cast<size_t>(col1) * static_cast<size_t>(K);
            int k = 0;
            while ((k < BK) && ((k & 3) != 0)) {
                acc1 = fma(float(tgA[k]), float(colB[kt + k]), acc1);
                ++k;
            }
            for (; (k + 3) < BK; k += 4) {
                const half4 a4 = half4(tgA[k + 0], tgA[k + 1], tgA[k + 2], tgA[k + 3]);
                const device half4* pB = reinterpret_cast<const device half4*>(colB + kt + k);
                half4 b4 = *pB;
                acc1 += dot(float4(a4), float4(b4));
            }
            while (k < BK) {
                acc1 = fma(float(tgA[k]), float(colB[kt + k]), acc1);
                ++k;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active0) { C[col0] = half(acc0); }
    if (active1) { C[col1] = half(acc1); }
}

// ===== Instantiations with stable host names =====
// Large-K small-N barrier-free reducer
template [[host_name("m1_dot_product_v6_nt_bn8_largek_smalln_kpar_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 8, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn16_largek_smalln_kpar_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 16, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn8_largek_smalln_kpar_bk256_tg256")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<256, 8, 256>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

// Additional experiments: widen/squeeze columns per TG and BK/TG
template [[host_name("m1_dot_product_v6_nt_bn4_largek_smalln_kpar_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 4, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn32_largek_smalln_kpar_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 32, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn32_largek_smalln_kpar_bk256_tg256")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<256, 32, 256>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

// More k-parallel experiments for largeK-smallN
template [[host_name("m1_dot_product_v6_nt_bn4_largek_smalln_kpar_bk256_tg256")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<256, 4, 256>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn16_largek_smalln_kpar_bk256_tg256")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<256, 16, 256>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn8_largek_smalln_kpar_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 8, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn16_largek_smalln_kpar_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 16, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn32_largek_smalln_kpar_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 32, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn64_largek_smalln_kpar_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

template [[host_name("m1_dot_product_v6_nt_bn64_largek_smalln_kpar_bk256_tg256")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_largek_smalln_kpar<256, 64, 256>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid, uint sg_id, uint simd_lane);

// Large-N vec4 tgread with A staging
template [[host_name("m1_dot_product_v6_nt_bn128_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 128, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn256_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 256, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// Additional experiments: vary BN and BK for large-N vec4 vA path
template [[host_name("m1_dot_product_v6_nt_bn64_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 64, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn192_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 192, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn128_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 128, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn256_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 256, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

// More vec4 vA large-N experiments
template [[host_name("m1_dot_product_v6_nt_bn192_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 192, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn160_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 160, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn160_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 160, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn224_col_vec4_tgread_vA_bk64_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 224, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn224_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 224, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn64_col_vec4_tgread_vA_bk128_tg128")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<128, 64, 128>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);

template [[host_name("m1_dot_product_v6_nt_bn256_col_vec4_tgread_vA_bk64_tg256")]] [[kernel]]
void m1_dot_product_v6_nt_kernel_col_vec4_tgread_vA<256, 256, 64>(
    device const half* A, device const half* B, device half* C,
    constant int& M, constant int& N, constant int& K,
    uint3 gid, uint3 lid);