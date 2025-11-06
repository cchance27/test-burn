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
