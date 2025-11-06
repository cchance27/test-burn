// M=1 dot product v7 kernels (functional baseline)
// Two strategies wired for harness integration:
//  - sbcast_vec4_bn256: column-tile per TG; scalar broadcast of A, simple accumulation
//  - kpar2_bn64: intended for large-K/small-N; current implementation uses straightforward per-column accumulation
// Each variant provides plain, bias, and accumulate functions. Accumulate uses D as input and adds matmul output.

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ===== v7 sbcast_vec4 (BN=256) =====
// Threadgroup covers 256 columns; each thread computes one column.
// Improve performance by tiling A into threadgroup memory so all threads
// reuse the same A tile, reducing global A reads by ~threadsPerTG.
// Note: This variant does not perform true vec4 B loads; naming matches harness configuration.
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_sbcast_vec4_bn256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int COLUMNS_PER_TG = 256;
    const int base_col = static_cast<int>(gid.x) * COLUMNS_PER_TG;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    // Tile A into threadgroup memory for reuse by all threads
    threadgroup half a_tile[128];
    // NT path: B is laid out as [N x K] row-major; index via column pointer
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = (K - k_base) < 128 ? (K - k_base) : 128;
        // Load A tile: first tile_len threads participate
        if (static_cast<int>(tid.x) < tile_len) {
            a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate directly from global B (NxK), using the cached A tile
        for (int kk = 0; kk < tile_len; ++kk) {
            const half bval = colB[static_cast<size_t>(k_base + kk)];
            acc = fma(float(a_tile[kk]), float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    D[col] = half(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_sbcast_vec4_bn256_bias(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    device const half* Bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int COLUMNS_PER_TG = 256;
    const int base_col = static_cast<int>(gid.x) * COLUMNS_PER_TG;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = (K - k_base) < 128 ? (K - k_base) : 128;
        if (static_cast<int>(tid.x) < tile_len) {
            a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len; ++kk) {
            const half bval = colB[static_cast<size_t>(k_base + kk)];
            acc = fma(float(a_tile[kk]), float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    acc += float(Bias[col]);
    D[col] = half(acc);
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_sbcast_vec4_bn256_accumulate(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int COLUMNS_PER_TG = 256;
    const int base_col = static_cast<int>(gid.x) * COLUMNS_PER_TG;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = (K - k_base) < 128 ? (K - k_base) : 128;
        if (static_cast<int>(tid.x) < tile_len) {
            a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len; ++kk) {
            const half bval = colB[static_cast<size_t>(k_base + kk)];
            acc = fma(float(a_tile[kk]), float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    // Accumulate onto initial D
    D[col] = half(acc + float(D[col]));
}

// ===== v7 kpar2 (BN=64) =====
// Threadgroup covers 64 columns; use the same A tiling strategy to reduce global reads.
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_kpar2_bn64(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int COLUMNS_PER_TG = 64;
    const int base_col = static_cast<int>(gid.x) * COLUMNS_PER_TG;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    threadgroup half b_tile[64 * 64];
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = (K - k_base) < 128 ? (K - k_base) : 128;
        if (static_cast<int>(tid.x) < tile_len) {
            a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len && kk < 64; ++kk) {
            const size_t row_start = static_cast<size_t>(k_base + kk) * static_cast<size_t>(N) + static_cast<size_t>(base_col);
            const int lane = static_cast<int>(tid.x);
            if (lane < COLUMNS_PER_TG && (base_col + lane) < N) {
                b_tile[kk * COLUMNS_PER_TG + lane] = B[row_start + static_cast<size_t>(lane)];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len && kk < 64; ++kk) {
            const float av = float(a_tile[kk]);
            const half bval = b_tile[kk * COLUMNS_PER_TG + static_cast<int>(tid.x)];
            acc = fma(av, float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    D[col] = half(acc);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_kpar2_bn64_bias(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    device const half* Bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int COLUMNS_PER_TG = 64;
    const int base_col = static_cast<int>(gid.x) * COLUMNS_PER_TG;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    threadgroup half b_tile[64 * 64];
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = (K - k_base) < 128 ? (K - k_base) : 128;
        if (static_cast<int>(tid.x) < tile_len) {
            a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len && kk < 64; ++kk) {
            const size_t row_start = static_cast<size_t>(k_base + kk) * static_cast<size_t>(N) + static_cast<size_t>(base_col);
            const int lane = static_cast<int>(tid.x);
            if (lane < COLUMNS_PER_TG && (base_col + lane) < N) {
                b_tile[kk * COLUMNS_PER_TG + lane] = B[row_start + static_cast<size_t>(lane)];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len && kk < 64; ++kk) {
            const float av = float(a_tile[kk]);
            const half bval = b_tile[kk * COLUMNS_PER_TG + static_cast<int>(tid.x)];
            acc = fma(av, float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    acc += float(Bias[col]);
    D[col] = half(acc);
}

[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_kpar2_bn64_accumulate(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int COLUMNS_PER_TG = 64;
    const int base_col = static_cast<int>(gid.x) * COLUMNS_PER_TG;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    threadgroup half b_tile[64 * 64];
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = (K - k_base) < 128 ? (K - k_base) : 128;
        if (static_cast<int>(tid.x) < tile_len) {
            a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len && kk < 64; ++kk) {
            const size_t row_start = static_cast<size_t>(k_base + kk) * static_cast<size_t>(N) + static_cast<size_t>(base_col);
            const int lane = static_cast<int>(tid.x);
            if (lane < COLUMNS_PER_TG && (base_col + lane) < N) {
                b_tile[kk * COLUMNS_PER_TG + lane] = B[row_start + static_cast<size_t>(lane)];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len && kk < 64; ++kk) {
            const float av = float(a_tile[kk]);
            const half bval = b_tile[kk * COLUMNS_PER_TG + static_cast<int>(tid.x)];
            acc = fma(av, float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    D[col] = half(acc + float(D[col]));
}

// ===== New v7 test kernels (ideas exploration) =====

// 1) Hybrid Memory Access Pattern for Large-K Small-N (BN=8, TG=256)
//    - K-parallel reduction within simdgroups (32 lanes)
//    - Vectorized half4 loads for A on aligned K blocks (tail-safe)
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn8_largek_smalln_hybrid_kpar_bk256_tg256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 8;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col_local = static_cast<int>(simdgroup_id);
    const int col = base_col + col_local;
    if (col_local >= BN || col >= N) { return; }

    float acc = 0.0f;
    const uint W = 32u; // simdgroup width
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);

    // Phase A: vectorized A loads (half4) when aligned
    int K4 = (K & ~3);
    for (int k = static_cast<int>(simd_lane_id) * 4; k < K4; k += static_cast<int>(W) * 4) {
        // Load 4 contiguous A values
        const device half4* A4 = (const device half4*)(A + k);
        half4 av4 = *A4;
        // Accumulate four steps (B remains scalar due to stride N)
        const int k0 = k + 0;
        const int k1 = k + 1;
        const int k2 = k + 2;
        const int k3 = k + 3;
        acc = fma(float(av4.x), float(colB[static_cast<size_t>(k0)]), acc);
        acc = fma(float(av4.y), float(colB[static_cast<size_t>(k1)]), acc);
        acc = fma(float(av4.z), float(colB[static_cast<size_t>(k2)]), acc);
        acc = fma(float(av4.w), float(colB[static_cast<size_t>(k3)]), acc);
    }

    // Phase B: tail using scalar strided lanes
    for (int k = K4 + static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc);
    }

    // Reduce within simdgroup
    acc += simd_shuffle_down(acc, 16u);
    acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);
    if (simd_lane_id == 0u) {
        D[col] = half(acc);
    }
}

// 2) Threadgroup Work Distribution Optimization (dynamic work stealing)
//    Base geometry BN=128, TG=128; threads claim columns from a TG-local queue.
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_nt_bn128_col_vec4_worksteal_bk128_tg128(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 128;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int cols_in_tg = (base_col < N) ? min(BN, N - base_col) : 0;
    if (cols_in_tg <= 0) { return; }

    threadgroup atomic_uint work_index;
    if (tidx == 0u) {
        atomic_store_explicit(&work_index, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (true) {
        uint idx = atomic_fetch_add_explicit(&work_index, 1u, memory_order_relaxed);
        if (static_cast<int>(idx) >= cols_in_tg) { break; }
        const int col = base_col + static_cast<int>(idx);
        float acc = 0.0f;
        // Simple per-column accumulation using NT layout (NxK)
        const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        for (int k = 0; k < K; ++k) {
            acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc);
        }
        D[col] = half(acc);
    }
}

// 3) Specialized Small-N Optimization (N<=16) using simdgroup reductions
//    BN=16, TG=256: 16 simdgroups, each reduces K for one column, no TGM.
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn16_smalln_simdgroupmm_tg256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 16;
    const int base_col = static_cast<int>(gid.x) * BN;
    // Two columns per simdgroup to cover BN=16 with TG=256 (8 SGs)
    const int SG_COLS = 2;
    const int col_pair = static_cast<int>(simdgroup_id);
    const int col0 = base_col + col_pair * SG_COLS + 0;
    const int col1 = base_col + col_pair * SG_COLS + 1;
    if (col_pair >= (BN / SG_COLS)) { return; }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint W = 32u;
    const device half* __restrict colB0 = (col0 < N) ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict colB1 = (col1 < N) ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;
    for (int k = static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        const float a = float(A[k]);
        if (colB0) { acc0 = fma(a, float(colB0[static_cast<size_t>(k)]), acc0); }
        if (colB1) { acc1 = fma(a, float(colB1[static_cast<size_t>(k)]), acc1); }
    }
    // simdgroup reduction
    acc0 += simd_shuffle_down(acc0, 16u);
    acc0 += simd_shuffle_down(acc0, 8u);
    acc0 += simd_shuffle_down(acc0, 4u);
    acc0 += simd_shuffle_down(acc0, 2u);
    acc0 += simd_shuffle_down(acc0, 1u);
    acc1 += simd_shuffle_down(acc1, 16u);
    acc1 += simd_shuffle_down(acc1, 8u);
    acc1 += simd_shuffle_down(acc1, 4u);
    acc1 += simd_shuffle_down(acc1, 2u);
    acc1 += simd_shuffle_down(acc1, 1u);
    if (simd_lane_id == 0u) {
        if (col0 < N) { D[col0] = half(acc0); }
        if (col1 < N) { D[col1] = half(acc1); }
    }
}

// 4) Prefetching Pipeline Optimization (Triple-buffer A tiles)
//    BN=128, TG=128, BK=64: overlap loads/compute via ring buffers.
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_nt_bn128_col_vec4_triplebuf_bk64_tg128(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 128;
    const int BK = 64;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col = base_col + static_cast<int>(tidx);
    if (static_cast<int>(tidx) >= BN || col >= N) { return; }

    threadgroup half a0[BK];
    threadgroup half a1[BK];
    threadgroup half a2[BK];

    float acc = 0.0f;
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);

    // Track base K and tile lengths for each buffer
    int baseK[3] = {0, 0, 0};
    int tlen[3]  = {0, 0, 0};

    // Preload first (a0)
    baseK[0] = 0;
    tlen[0] = min(BK, K - baseK[0]);
    if (static_cast<int>(tidx) < tlen[0]) { a0[static_cast<int>(tidx)] = A[baseK[0] + static_cast<int>(tidx)]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Preload second (a1)
    baseK[1] = baseK[0] + tlen[0];
    tlen[1] = (baseK[1] < K) ? min(BK, K - baseK[1]) : 0;
    if (tlen[1] > 0) {
        if (static_cast<int>(tidx) < tlen[1]) { a1[static_cast<int>(tidx)] = A[baseK[1] + static_cast<int>(tidx)]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int nextBase = baseK[1] + tlen[1];
    int cur = 0;
    int nextIndex = 2; // fill a2 first

    while (tlen[0] > 0 || tlen[1] > 0 || tlen[2] > 0) {
        threadgroup half* curBuf = (cur == 0) ? a0 : ((cur == 1) ? a1 : a2);
        const int len = tlen[cur];
        const int kbase = baseK[cur];

        for (int kk = 0; kk < len; ++kk) {
            const half bval = colB[static_cast<size_t>(kbase + kk)];
            acc = fma(float(curBuf[kk]), float(bval), acc);
        }

        // mark consumed
        tlen[cur] = 0;

        // prefetch into nextIndex
        if (nextBase < K) {
            baseK[nextIndex] = nextBase;
            tlen[nextIndex] = min(BK, K - baseK[nextIndex]);
            if (static_cast<int>(tidx) < tlen[nextIndex]) {
                (nextIndex == 0 ? a0 : (nextIndex == 1 ? a1 : a2))[static_cast<int>(tidx)] = A[baseK[nextIndex] + static_cast<int>(tidx)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            nextBase += tlen[nextIndex];
        }

        cur = (cur + 1) % 3;
        nextIndex = (nextIndex + 1) % 3;
    }
    D[col] = half(acc);
}

// Utility: GELU approximate
inline float v7_gelu(float x) {
    const float c0 = 0.7978845608028654f; // sqrt(2/pi)
    const float c1 = 0.044715f;
    float x3 = x * x * x;
    float t = tanh(c0 * (x + c1 * x3));
    return 0.5f * x * (1.0f + t);
}

// 5) Output Fusion: Bias + ReLU (BN=256, TG=256)
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn256_fused_bias_relu_tg256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    device const half* Bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 256;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = min(128, K - k_base);
        if (static_cast<int>(tid.x) < tile_len) { a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len; ++kk) {
            const half bval = colB[static_cast<size_t>(k_base + kk)];
            acc = fma(float(a_tile[kk]), float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    acc += float(Bias[col]);
    acc = max(acc, 0.0f);
    D[col] = half(acc);
}

// 5b) Output Fusion: Bias + GELU (BN=256, TG=256)
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn256_fused_bias_gelu_tg256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    device const half* Bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 256;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col = base_col + static_cast<int>(tid.x);
    if (col >= N) { return; }
    float acc = 0.0f;
    threadgroup half a_tile[128];
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = min(128, K - k_base);
        if (static_cast<int>(tid.x) < tile_len) { a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < tile_len; ++kk) {
            const half bval = colB[static_cast<size_t>(k_base + kk)];
            acc = fma(float(a_tile[kk]), float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    acc += float(Bias[col]);
    acc = v7_gelu(acc);
    D[col] = half(acc);
}

// 6) Adaptive Kernel Selection (inside kernel) for BN=128, TG=128
//    Choose k-par reduction for largeK/smallN, else col-tiled path.
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_nt_adaptive_bn128_tg128(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 128;
    const int base_col = static_cast<int>(gid.x) * BN;

    bool use_kpar = (K >= 2048 && N <= 2048);
    if (use_kpar) {
        // Process all BN columns per threadgroup: each simdgroup reduces multiple columns in sequence.
        const int SG_COUNT = 4; // 128 threads / 32 lanes per simdgroup
        const int col_local = static_cast<int>(simdgroup_id); // 0..SG_COUNT-1
        const uint W = 32u;
        for (int colStart = 0; colStart < BN; colStart += SG_COUNT) {
            const int col = base_col + colStart + col_local;
            if (col >= N) { continue; }
            const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
            float acc = 0.0f;
            for (int k = static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
                acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc);
            }
            // simdgroup reduction
            acc += simd_shuffle_down(acc, 16u);
            acc += simd_shuffle_down(acc, 8u);
            acc += simd_shuffle_down(acc, 4u);
            acc += simd_shuffle_down(acc, 2u);
            acc += simd_shuffle_down(acc, 1u);
            if (simd_lane_id == 0u) { D[col] = half(acc); }
        }
    } else {
        const int col = base_col + static_cast<int>(tidx);
        if (static_cast<int>(tidx) >= BN || col >= N) { return; }
        float acc = 0.0f;
        threadgroup half a_tile[128];
        int k_base = 0;
        while (k_base < K) {
            const int tile_len = min(128, K - k_base);
            if (static_cast<int>(tidx) < tile_len) { a_tile[static_cast<int>(tidx)] = A[k_base + static_cast<int>(tidx)]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
            for (int kk = 0; kk < tile_len; ++kk) {
                const half bval = colB[static_cast<size_t>(k_base + kk)];
                acc = fma(float(a_tile[kk]), float(bval), acc);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_base += tile_len;
        }
        D[col] = half(acc);
    }
}

// ==========================================================
// New v7 tiny, barrier-free param kernels with vector loads
// - Designed for m=1, medium-N (e.g., ~10k), K~512-2048
// - No threadgroup memory, no barriers; rely on L1 for A
// - BN, BK and UNROLL provided at runtime via constant buffers
// - TG (threads per threadgroup) is chosen at dispatch time
// ==========================================================

inline void v7_tiny_vec4_core(
    device const half* __restrict A,
    device const half* __restrict B,
    device half* __restrict C,
    int M,
    int N,
    int K,
    int BN,
    int UNROLL,
    ushort tgWidth,
    uint3 gid,
    uint3 lid)
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    // tgWidth provided from host via builtin threads_per_threadgroup
    const int group_base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += tgWidth) {
        const int col = group_base_col + co;
        if (col >= N) continue;

        const device half* __restrict bcol = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;

        int k = 0;
        // align to 4 for half4 loads from B
        for (; k < K && ((k & 3) != 0); ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        const int step4 = 4;
        const int U = max(1, UNROLL);
        // unrolled loop over half4
        for (; k + U*step4 - 1 < K; k += U*step4) {
            #pragma unroll 1
            for (int u = 0; u < U; ++u) {
                const int kk = k + u*step4;
                const device half4* pB = reinterpret_cast<const device half4*>(bcol + kk);
                half4 hb = *pB;
                float4 fb = float4(hb);
                acc = fma(float(A[kk + 0]), fb.x, acc);
                acc = fma(float(A[kk + 1]), fb.y, acc);
                acc = fma(float(A[kk + 2]), fb.z, acc);
                acc = fma(float(A[kk + 3]), fb.w, acc);
            }
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        C[col] = half(acc);
    }
}

inline void v7_tiny_vec4A_core(
    device const half* __restrict A,
    device const half* __restrict B,
    device half* __restrict C,
    int M,
    int N,
    int K,
    int BN,
    int UNROLL,
    ushort tgWidth,
    uint3 gid,
    uint3 lid)
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    // tgWidth provided
    const int group_base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += tgWidth) {
        const int col = group_base_col + co;
        if (col >= N) continue;
        const device half* __restrict bcol = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;

        int k = 0;
        // align both A and B to 4
        for (; k < K && ((k & 3) != 0); ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        const int U = max(1, UNROLL);
        for (; k + 4*U - 1 < K; k += 4*U) {
            #pragma unroll 1
            for (int u = 0; u < U; ++u) {
                const int kk = k + 4*u;
                const device half4* pA = reinterpret_cast<const device half4*>(A + kk);
                const device half4* pB = reinterpret_cast<const device half4*>(bcol + kk);
                float4 fa = float4(*pA);
                float4 fb = float4(*pB);
                acc = fma(fa.x, fb.x, acc);
                acc = fma(fa.y, fb.y, acc);
                acc = fma(fa.z, fb.z, acc);
                acc = fma(fa.w, fb.w, acc);
            }
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        C[col] = half(acc);
    }
}

inline void v7_tiny_vec8_core(
    device const half* __restrict A,
    device const half* __restrict B,
    device half* __restrict C,
    int M,
    int N,
    int K,
    int BN,
    int UNROLL,
    ushort tgWidth,
    uint3 gid,
    uint3 lid)
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    // tgWidth provided
    const int group_base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += tgWidth) {
        const int col = group_base_col + co;
        if (col >= N) continue;
        const device half* __restrict bcol = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;

        int k = 0;
        // align to 8 for paired half4 reads from B
        for (; k < K && ((k & 7) != 0); ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        const int U = max(1, UNROLL);
        for (; k + 8*U - 1 < K; k += 8*U) {
            #pragma unroll 1
            for (int u = 0; u < U; ++u) {
                const int kk = k + 8*u;
                const device half4* pB0 = reinterpret_cast<const device half4*>(bcol + kk);
                const device half4* pB1 = reinterpret_cast<const device half4*>(bcol + kk + 4);
                float4 fb0 = float4(*pB0);
                float4 fb1 = float4(*pB1);
                acc = fma(float(A[kk + 0]), fb0.x, acc);
                acc = fma(float(A[kk + 1]), fb0.y, acc);
                acc = fma(float(A[kk + 2]), fb0.z, acc);
                acc = fma(float(A[kk + 3]), fb0.w, acc);
                acc = fma(float(A[kk + 4]), fb1.x, acc);
                acc = fma(float(A[kk + 5]), fb1.y, acc);
                acc = fma(float(A[kk + 6]), fb1.z, acc);
                acc = fma(float(A[kk + 7]), fb1.w, acc);
            }
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        C[col] = half(acc);
    }
}

// Kernel wrappers (param encoding order matches Swift runner):
// buffers: 0=A,1=B,2=C, [bias?], M,N,K, BN, BK, UNROLL

[[kernel]]
void m1_dot_product_v7_nt_tiny_vec4_param(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& BN [[buffer(6)]],
    constant int& BK [[buffer(7)]],
    constant int& UNROLL [[buffer(8)]],
    uint3 tgsz [[threads_per_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    (void)BK;
    v7_tiny_vec4_core(A, B, C, M, N, K, BN, UNROLL, static_cast<ushort>(tgsz.x), gid, lid);
}

[[kernel]]
void m1_dot_product_v7_nt_tiny_vec4a_param(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& BN [[buffer(6)]],
    constant int& BK [[buffer(7)]],
    constant int& UNROLL [[buffer(8)]],
    uint3 tgsz [[threads_per_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    (void)BK;
    v7_tiny_vec4A_core(A, B, C, M, N, K, BN, UNROLL, static_cast<ushort>(tgsz.x), gid, lid);
}

[[kernel]]
void m1_dot_product_v7_nt_tiny_vec8_param(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& BN [[buffer(6)]],
    constant int& BK [[buffer(7)]],
    constant int& UNROLL [[buffer(8)]],
    uint3 tgsz [[threads_per_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    (void)BK;
    v7_tiny_vec8_core(A, B, C, M, N, K, BN, UNROLL, static_cast<ushort>(tgsz.x), gid, lid);
}

// Specialized wrapper for best-known configuration (bn32, bk256, tg256, unroll=4)
// Matches generic runner's parameter layout (M,N,K only after output), no BN/BK/UNROLL buffers needed.
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_tiny_vec8_bn32_bk256_tg256_u4(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 tgsz [[threads_per_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    const int BN = 32;
    const int UNROLL = 4;
    v7_tiny_vec8_core(A, B, C, M, N, K, BN, UNROLL, static_cast<ushort>(tgsz.x), gid, lid);
}

// ==========================================================
// v7 tiny clone of v4 winner (BN=64, TG=64), M=1 only
// Barrier-free, scalar A/B loads with 8-FMA unroll
[[kernel, max_total_threads_per_threadgroup(64)]]
void m1_dot_product_v7_nt_tiny_bn64_tg64(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    const int BN = 64;
    const int base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += 64) {
        const int col = base_col + co;
        if (col >= N) continue;
        const device half* __restrict bcol = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;
        int k = 0;
        for (; k + 7 < K; k += 8) {
            acc = fma(float(A[k    ]), float(bcol[k    ]), acc);
            acc = fma(float(A[k + 1]), float(bcol[k + 1]), acc);
            acc = fma(float(A[k + 2]), float(bcol[k + 2]), acc);
            acc = fma(float(A[k + 3]), float(bcol[k + 3]), acc);
            acc = fma(float(A[k + 4]), float(bcol[k + 4]), acc);
            acc = fma(float(A[k + 5]), float(bcol[k + 5]), acc);
            acc = fma(float(A[k + 6]), float(bcol[k + 6]), acc);
            acc = fma(float(A[k + 7]), float(bcol[k + 7]), acc);
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        C[col] = half(acc);
    }
}

// ==========================================================
// v7 tiny simdgroup-friendly variant (BN=64, TG=64)
// Use half4 B vector loads and 8-FMA unroll to raise ILP
[[kernel, max_total_threads_per_threadgroup(64)]]
void m1_dot_product_v7_nt_tiny_sgmm_bn64_tg64(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    const int BN = 64;
    const int base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += 64) {
        const int col = base_col + co;
        if (col >= N) continue;
        const device half* __restrict bcol = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc = 0.0f;
        int k = 0;
        // align to 4
        for (; k < K && ((k & 3) != 0); ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        for (; k + 7 < K; k += 8) {
            // two half4 loads of B, scalar A to keep reg pressure reasonable
            const device half4* pB0 = reinterpret_cast<const device half4*>(bcol + k);
            const device half4* pB1 = reinterpret_cast<const device half4*>(bcol + k + 4);
            float4 fb0 = float4(*pB0);
            float4 fb1 = float4(*pB1);
            acc = fma(float(A[k + 0]), fb0.x, acc);
            acc = fma(float(A[k + 1]), fb0.y, acc);
            acc = fma(float(A[k + 2]), fb0.z, acc);
            acc = fma(float(A[k + 3]), fb0.w, acc);
            acc = fma(float(A[k + 4]), fb1.x, acc);
            acc = fma(float(A[k + 5]), fb1.y, acc);
            acc = fma(float(A[k + 6]), fb1.z, acc);
            acc = fma(float(A[k + 7]), fb1.w, acc);
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(bcol[k]), acc);
        }
        C[col] = half(acc);
    }
}

// ==========================================================
// v7 multi-row batching variant (BN=64, TG=64, rowsPerTG=8)
// Reuse B across 8 rows to raise arithmetic intensity
[[kernel, max_total_threads_per_threadgroup(64)]]
void m1_dot_product_v7_nt_tiny_multim_bn64_tg64_rows8(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    const int rowsPerTG = 8;
    const int rowBase = static_cast<int>(gid.y) * rowsPerTG;
    const ushort lane = static_cast<ushort>(lid.x);
    const int BN = 64;
    const int base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += 64) {
        const int col = base_col + co;
        if (col >= N) continue;
        const device half* __restrict bcol = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        float acc[rowsPerTG];
        for (int r = 0; r < rowsPerTG; ++r) acc[r] = 0.0f;

        int k = 0;
        for (; k + 7 < K; k += 8) {
            // Load B once, reuse across rows
            half b0 = bcol[k + 0];
            half b1 = bcol[k + 1];
            half b2 = bcol[k + 2];
            half b3 = bcol[k + 3];
            half b4 = bcol[k + 4];
            half b5 = bcol[k + 5];
            half b6 = bcol[k + 6];
            half b7 = bcol[k + 7];
            for (int r = 0; r < rowsPerTG; ++r) {
                const int row = rowBase + r;
                if (row >= M) break;
                const device half* arow = A + static_cast<size_t>(row) * static_cast<size_t>(K);
                acc[r] = fma(float(arow[k + 0]), float(b0), acc[r]);
                acc[r] = fma(float(arow[k + 1]), float(b1), acc[r]);
                acc[r] = fma(float(arow[k + 2]), float(b2), acc[r]);
                acc[r] = fma(float(arow[k + 3]), float(b3), acc[r]);
                acc[r] = fma(float(arow[k + 4]), float(b4), acc[r]);
                acc[r] = fma(float(arow[k + 5]), float(b5), acc[r]);
                acc[r] = fma(float(arow[k + 6]), float(b6), acc[r]);
                acc[r] = fma(float(arow[k + 7]), float(b7), acc[r]);
            }
        }
        for (; k < K; ++k) {
            half bv = bcol[k];
            for (int r = 0; r < rowsPerTG; ++r) {
                const int row = rowBase + r;
                if (row >= M) break;
                const device half* arow = A + static_cast<size_t>(row) * static_cast<size_t>(K);
                acc[r] = fma(float(arow[k]), float(bv), acc[r]);
            }
        }
        for (int r = 0; r < rowsPerTG; ++r) {
            const int row = rowBase + r;
            if (row >= M) break;
            C[static_cast<size_t>(row) * static_cast<size_t>(N) + static_cast<size_t>(col)] = half(acc[r]);
        }
    }
}

// 7) Memory Layout Transformation for Better Coalescing (BN=64, BK=64)
//    Transform B tile into TG memory for contiguous access.
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_nt_bn64_col_vec4_transformB_tg128(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 64;
    const int BK = 64;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col = base_col + static_cast<int>(tid.x);
    if (static_cast<int>(tid.x) >= BN || col >= N) { return; }

    threadgroup half a_tile[BK];
    threadgroup half b_tile[BK * BN]; // row-major: kk * BN + lane

    float acc = 0.0f;
    int k_base = 0;
    while (k_base < K) {
        const int tile_len = min(BK, K - k_base);
        // Load A tile cooperatively
        if (static_cast<int>(tid.x) < tile_len) { a_tile[static_cast<int>(tid.x)] = A[k_base + static_cast<int>(tid.x)]; }

        // Transform B submatrix (K tile x BN columns) into TG memory from NT layout (NxK)
        const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
        for (int kk = 0; kk < tile_len; ++kk) {
            b_tile[kk * BN + static_cast<int>(tid.x)] = colB[static_cast<size_t>(k_base + kk)];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate using contiguous TG B tile
        for (int kk = 0; kk < tile_len; ++kk) {
            const half bval = b_tile[kk * BN + static_cast<int>(tid.x)];
            acc = fma(float(a_tile[kk]), float(bval), acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        k_base += tile_len;
    }
    D[col] = half(acc);
}

// 2b) Work stealing with chunked claims (reduces atomic contention), BN=128
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_nt_bn128_col_vec4_worksteal_chunk4_bk128_tg128(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 128;
    const int CHUNK = 4;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int cols_in_tg = (base_col < N) ? min(BN, N - base_col) : 0;
    if (cols_in_tg <= 0) { return; }

    threadgroup atomic_uint work_index;
    if (tidx == 0u) { atomic_store_explicit(&work_index, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (true) {
        uint idx = atomic_fetch_add_explicit(&work_index, CHUNK, memory_order_relaxed);
        if (static_cast<int>(idx) >= cols_in_tg) { break; }
        int limit = min(CHUNK, cols_in_tg - static_cast<int>(idx));
        for (int i = 0; i < limit; ++i) {
            const int col = base_col + static_cast<int>(idx) + i;
            float acc = 0.0f;
            const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
            for (int k = 0; k < K; ++k) { acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc); }
            D[col] = half(acc);
        }
    }
}

// 2c) Work stealing with chunked claims (reduces atomic contention), BN=64
[[kernel, max_total_threads_per_threadgroup(128)]]
void m1_dot_product_v7_nt_bn64_col_vec4_worksteal_chunk4_bk128_tg128(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint tidx [[thread_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 64;
    const int CHUNK = 4;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int cols_in_tg = (base_col < N) ? min(BN, N - base_col) : 0;
    if (cols_in_tg <= 0) { return; }

    threadgroup atomic_uint work_index;
    if (tidx == 0u) { atomic_store_explicit(&work_index, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (true) {
        uint idx = atomic_fetch_add_explicit(&work_index, CHUNK, memory_order_relaxed);
        if (static_cast<int>(idx) >= cols_in_tg) { break; }
        int limit = min(CHUNK, cols_in_tg - static_cast<int>(idx));
        for (int i = 0; i < limit; ++i) {
            const int col = base_col + static_cast<int>(idx) + i;
            float acc = 0.0f;
            const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
            for (int k = 0; k < K; ++k) { acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc); }
            D[col] = half(acc);
        }
    }
}

// 1b) Large-K/Small-N K-parallel with vec4 loads for both A and B, BN=8
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn8_largek_smalln_kpar_vec4_both_bk256_tg256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 8;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col = base_col + static_cast<int>(simdgroup_id);
    if (static_cast<int>(simdgroup_id) >= BN || col >= N) { return; }

    const uint W = 32u;
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    float acc = 0.0f;
    int K4 = (K & ~3);
    for (int k = static_cast<int>(simd_lane_id) * 4; k + 3 < K4; k += static_cast<int>(W) * 4) {
        const device half4* A4 = reinterpret_cast<const device half4*>(A + k);
        const device half4* B4 = reinterpret_cast<const device half4*>(colB + k);
        acc += dot(float4(*A4), float4(*B4));
    }
    for (int k = K4 + static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc);
    }
    acc += simd_shuffle_down(acc, 16u);
    acc += simd_shuffle_down(acc, 8u);
    acc += simd_shuffle_down(acc, 4u);
    acc += simd_shuffle_down(acc, 2u);
    acc += simd_shuffle_down(acc, 1u);
    if (simd_lane_id == 0u) { D[col] = half(acc); }
}

// 1c) Large-K/Small-N K-parallel with vec4 loads for both A and B, BN=16 (two columns per SG)
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn16_largek_smalln_kpar_vec4_both_bk256_tg256(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    if (M != 1) { return; }
    const int BN = 16;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int SG_COLS = 2;
    const int col_pair = static_cast<int>(simdgroup_id);
    if (col_pair >= (BN / SG_COLS)) { return; }
    const int col0 = base_col + col_pair * SG_COLS + 0;
    const int col1 = base_col + col_pair * SG_COLS + 1;

    const uint W = 32u;
    const device half* __restrict colB0 = (col0 < N) ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict colB1 = (col1 < N) ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int K4 = (K & ~3);
    for (int k = static_cast<int>(simd_lane_id) * 4; k + 3 < K4; k += static_cast<int>(W) * 4) {
        const device half4* A4 = reinterpret_cast<const device half4*>(A + k);
        const float4 af = float4(*A4);
        if (colB0) { const device half4* B40 = reinterpret_cast<const device half4*>(colB0 + k); acc0 += dot(af, float4(*B40)); }
        if (colB1) { const device half4* B41 = reinterpret_cast<const device half4*>(colB1 + k); acc1 += dot(af, float4(*B41)); }
    }
    for (int k = K4 + static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        const float a = float(A[k]);
        if (colB0) { acc0 = fma(a, float(colB0[static_cast<size_t>(k)]), acc0); }
        if (colB1) { acc1 = fma(a, float(colB1[static_cast<size_t>(k)]), acc1); }
    }
    acc0 += simd_shuffle_down(acc0, 16u);
    acc0 += simd_shuffle_down(acc0, 8u);
    acc0 += simd_shuffle_down(acc0, 4u);
    acc0 += simd_shuffle_down(acc0, 2u);
    acc0 += simd_shuffle_down(acc0, 1u);
    acc1 += simd_shuffle_down(acc1, 16u);
    acc1 += simd_shuffle_down(acc1, 8u);
    acc1 += simd_shuffle_down(acc1, 4u);
    acc1 += simd_shuffle_down(acc1, 2u);
    acc1 += simd_shuffle_down(acc1, 1u);
    if (simd_lane_id == 0u) {
        if (col0 < N) { D[col0] = half(acc0); }
        if (col1 < N) { D[col1] = half(acc1); }
    }
}

// Optimized for K-parallel reduction with full vectorization
// Changes: manual FMA, XOR shuffle, __restrict hints.
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn8_largek_smalln_kpar_vec4_both_bk256_tg256_fma_xorshuf(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) 
{
    if (M != 1) { return; }
    
    const int BN = 8;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int col = base_col + static_cast<int>(simdgroup_id);
    
    if (static_cast<int>(simdgroup_id) >= BN || col >= N) { return; }

    const uint W = 32u;
    const device half* __restrict colB = B + static_cast<size_t>(col) * static_cast<size_t>(K);
    float acc = 0.0f;
    
    // Process K in half4 chunks, padded to multiple of 4
    const int K4 = K & ~3;
    
    // Vectorized path: manual FMA for better fusion than dot()
    for (int k = static_cast<int>(simd_lane_id) * 4; k + 3 < K4; k += static_cast<int>(W) * 4) {
        const device half4* A4 = reinterpret_cast<const device half4*>(A + k);
        const device half4* B4 = reinterpret_cast<const device half4*>(colB + k);
        
        const float4 af = float4(*A4);
        const float4 bf = float4(*B4);
        
        acc = fma(af.x, bf.x, acc);
        acc = fma(af.y, bf.y, acc);
        acc = fma(af.z, bf.z, acc);
        acc = fma(af.w, bf.w, acc);
    }
    
    // Scalar tail for misaligned K
    for (int k = K4 + static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        acc = fma(float(A[k]), float(colB[static_cast<size_t>(k)]), acc);
    }
    
    // XOR-based reduction (fewer dependencies than shuffle_down)
    acc += simd_shuffle_xor(acc, 16u);
    acc += simd_shuffle_xor(acc, 8u);
    acc += simd_shuffle_xor(acc, 4u);
    acc += simd_shuffle_xor(acc, 2u);
    acc += simd_shuffle_xor(acc, 1u);
    
    if (simd_lane_id == 0u) { 
        D[col] = half(acc); 
    }
}

[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nt_bn16_largek_smalln_kpar_vec4_both_bk256_tg256_fma_xorshuf(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& N [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) 
{
    if (M != 1) { return; }
    
    const int BN = 16;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int SG_COLS = 2;
    const int col_pair = static_cast<int>(simdgroup_id);
    
    if (col_pair >= (BN / SG_COLS)) { return; }
    
    const int col0 = base_col + col_pair * SG_COLS + 0;
    const int col1 = base_col + col_pair * SG_COLS + 1;

    const uint W = 32u;
    const device half* __restrict colB0 = (col0 < N) ? (B + static_cast<size_t>(col0) * static_cast<size_t>(K)) : nullptr;
    const device half* __restrict colB1 = (col1 < N) ? (B + static_cast<size_t>(col1) * static_cast<size_t>(K)) : nullptr;
    
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int K4 = K & ~3;
    
    // Dual-column vectorized accumulation
    for (int k = static_cast<int>(simd_lane_id) * 4; k + 3 < K4; k += static_cast<int>(W) * 4) {
        const device half4* A4 = reinterpret_cast<const device half4*>(A + k);
        const float4 af = float4(*A4);
        
        if (colB0) {
            const device half4* B40 = reinterpret_cast<const device half4*>(colB0 + k);
            const float4 bf0 = float4(*B40);
            acc0 = fma(af.x, bf0.x, acc0);
            acc0 = fma(af.y, bf0.y, acc0);
            acc0 = fma(af.z, bf0.z, acc0);
            acc0 = fma(af.w, bf0.w, acc0);
        }
        
        if (colB1) {
            const device half4* B41 = reinterpret_cast<const device half4*>(colB1 + k);
            const float4 bf1 = float4(*B41);
            acc1 = fma(af.x, bf1.x, acc1);
            acc1 = fma(af.y, bf1.y, acc1);
            acc1 = fma(af.z, bf1.z, acc1);
            acc1 = fma(af.w, bf1.w, acc1);
        }
    }
    
    // Scalar tail
    for (int k = K4 + static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        const float a_val = float(A[k]);
        if (colB0) { acc0 = fma(a_val, float(colB0[static_cast<size_t>(k)]), acc0); }
        if (colB1) { acc1 = fma(a_val, float(colB1[static_cast<size_t>(k)]), acc1); }
    }
    
    // XOR reduction for both columns
    acc0 += simd_shuffle_xor(acc0, 16u);
    acc0 += simd_shuffle_xor(acc0, 8u);
    acc0 += simd_shuffle_xor(acc0, 4u);
    acc0 += simd_shuffle_xor(acc0, 2u);
    acc0 += simd_shuffle_xor(acc0, 1u);
    
    acc1 += simd_shuffle_xor(acc1, 16u);
    acc1 += simd_shuffle_xor(acc1, 8u);
    acc1 += simd_shuffle_xor(acc1, 4u);
    acc1 += simd_shuffle_xor(acc1, 2u);
    acc1 += simd_shuffle_xor(acc1, 1u);
    
    if (simd_lane_id == 0u) {
        if (col0 < N) { D[col0] = half(acc0); }
        if (col1 < N) { D[col1] = half(acc1); }
    }
}

// ==========================================================
// BIAS-ENABLED VARIANTS OF FASTEST v7 KERNELS
// ==========================================================

// Bias variant: nn_tiny_bn64_tg64 with bias support (tB=0, non-transposed B: K×N layout)
[[kernel, max_total_threads_per_threadgroup(64)]]
void m1_dot_product_v7_nn_tiny_bn64_tg64_bias(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    const int BN = 64;
    const int base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += 64) {
        const int col = base_col + co;
        if (col >= N) continue;
        // For tB=0: B is K×N, so element B[k,col] is at B[k*N + col]
        const size_t col_offset = static_cast<size_t>(col);
        const size_t N_size = static_cast<size_t>(N);
        float acc = 0.0f;
        int k = 0;
        for (; k + 7 < K; k += 8) {
            acc = fma(float(A[k    ]), float(B[static_cast<size_t>(k    ) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 1]), float(B[static_cast<size_t>(k + 1) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 2]), float(B[static_cast<size_t>(k + 2) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 3]), float(B[static_cast<size_t>(k + 3) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 4]), float(B[static_cast<size_t>(k + 4) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 5]), float(B[static_cast<size_t>(k + 5) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 6]), float(B[static_cast<size_t>(k + 6) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 7]), float(B[static_cast<size_t>(k + 7) * N_size + col_offset]), acc);
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(B[static_cast<size_t>(k) * N_size + col_offset]), acc);
        }
        // Add bias
        acc += float(bias[col]);
        C[col] = half(acc);
    }
}

// Bias variant: nn_tiny_sgmm_bn64_tg64 with bias support (tB=0, non-transposed B: K×N layout)
[[kernel, max_total_threads_per_threadgroup(64)]]
void m1_dot_product_v7_nn_tiny_sgmm_bn64_tg64_bias(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]])
{
    if (M != 1) { return; }
    const ushort lane = static_cast<ushort>(lid.x);
    const int BN = 64;
    const int base_col = static_cast<int>(gid.x) * BN;

    for (int co = lane; co < BN; co += 64) {
        const int col = base_col + co;
        if (col >= N) continue;
        // For tB=0: B is K×N, access pattern B[k*N + col]
        const size_t col_offset = static_cast<size_t>(col);
        const size_t N_size = static_cast<size_t>(N);
        float acc = 0.0f;
        int k = 0;
        // align to 4
        for (; k < K && ((k & 3) != 0); ++k) {
            acc = fma(float(A[k]), float(B[static_cast<size_t>(k) * N_size + col_offset]), acc);
        }
        for (; k + 7 < K; k += 8) {
            // Load A scalars and B elements with strided access
            acc = fma(float(A[k + 0]), float(B[static_cast<size_t>(k + 0) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 1]), float(B[static_cast<size_t>(k + 1) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 2]), float(B[static_cast<size_t>(k + 2) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 3]), float(B[static_cast<size_t>(k + 3) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 4]), float(B[static_cast<size_t>(k + 4) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 5]), float(B[static_cast<size_t>(k + 5) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 6]), float(B[static_cast<size_t>(k + 6) * N_size + col_offset]), acc);
            acc = fma(float(A[k + 7]), float(B[static_cast<size_t>(k + 7) * N_size + col_offset]), acc);
        }
        for (; k < K; ++k) {
            acc = fma(float(A[k]), float(B[static_cast<size_t>(k) * N_size + col_offset]), acc);
        }
        // Add bias
        acc += float(bias[col]);
        C[col] = half(acc);
    }
}

// Bias variant: nn_bn16_largek_smalln_kpar_vec4_both_bk256_tg256 with bias support (tB=0, non-transposed B: K×N layout)
[[kernel, max_total_threads_per_threadgroup(256)]]
void m1_dot_product_v7_nn_bn16_largek_smalln_kpar_vec4_both_bk256_tg256_bias(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* D [[buffer(2)]],
    device const half* bias [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) 
{
    if (M != 1) { return; }
    const int BN = 16;
    const int base_col = static_cast<int>(gid.x) * BN;
    const int SG_COLS = 2;
    const int col_pair = static_cast<int>(simdgroup_id);
    if (col_pair >= (BN / SG_COLS)) { return; }
    const int col0 = base_col + col_pair * SG_COLS + 0;
    const int col1 = base_col + col_pair * SG_COLS + 1;

    const uint W = 32u;
    // For tB=0: B is K×N, so B[k,col] is at B[k*N + col]
    const bool valid0 = (col0 < N);
    const bool valid1 = (col1 < N);
    
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    
    // K-parallel reduction: each lane handles elements spaced by W
    for (int k = static_cast<int>(simd_lane_id); k < K; k += static_cast<int>(W)) {
        const float a = float(A[k]);
        const size_t row_offset = static_cast<size_t>(k) * static_cast<size_t>(N);
        if (valid0) { acc0 = fma(a, float(B[row_offset + col0]), acc0); }
        if (valid1) { acc1 = fma(a, float(B[row_offset + col1]), acc1); }
    }
    
    // Reduce across simdgroup
    acc0 += simd_shuffle_down(acc0, 16u);
    acc0 += simd_shuffle_down(acc0, 8u);
    acc0 += simd_shuffle_down(acc0, 4u);
    acc0 += simd_shuffle_down(acc0, 2u);
    acc0 += simd_shuffle_down(acc0, 1u);
    acc1 += simd_shuffle_down(acc1, 16u);
    acc1 += simd_shuffle_down(acc1, 8u);
    acc1 += simd_shuffle_down(acc1, 4u);
    acc1 += simd_shuffle_down(acc1, 2u);
    acc1 += simd_shuffle_down(acc1, 1u);
    
    if (simd_lane_id == 0u) {
        if (valid0) { 
            acc0 += float(bias[col0]);
            D[col0] = half(acc0); 
        }
        if (valid1) { 
            acc1 += float(bias[col1]);
            D[col1] = half(acc1); 
        }
    }
}