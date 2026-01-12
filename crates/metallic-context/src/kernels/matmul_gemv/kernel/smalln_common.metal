// Small-N GEMV shared helper for FP16

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

#ifndef GEMV_SMALLN_TILE_K
#define GEMV_SMALLN_TILE_K 64u
#endif

// Generic FP16 small-N GEMV implementation.
// Each thread computes a single (row, col) output element for COLS columns per threadgroup row.
// Parameters:
//  - COLS: columns computed per threadgroup row (N for this kernel variant)
//  - ROWS_PER_TG: number of rows handled by a single threadgroup
//  - TILE_K_SIZE: staging tile size along K in shared memory
//  - UseGridY: if true, rows advance along tgid.y; otherwise along tgid.x
template <uint COLS, uint ROWS_PER_TG, uint TILE_K_SIZE = GEMV_SMALLN_TILE_K, bool UseGridY = false>
ALWAYS_INLINE void gemv_f16_smalln_impl(
    device const half* A,
    device const half* B,
    device half*       C,
    constant uint&     M,
    constant uint&     K,
    uint               tid,
    uint2              tgid,
    threadgroup half*  smemB) {

    const uint row_in_tg  = tid / COLS;
    const uint col_in_row = tid % COLS;

    const uint grid_group = UseGridY ? tgid.y : tgid.x;
    const uint m_idx = grid_group * ROWS_PER_TG + row_in_tg;
    const bool row_valid = m_idx < M;

    float accum = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K_SIZE) {
        const uint k_tile_size = min(TILE_K_SIZE, K - k_base);

        // Cooperative load of B tile into shared memory.
        // Keep all threads participating; guard accesses individually.
        for (uint i = tid; i < k_tile_size * COLS; i += ROWS_PER_TG * COLS) {
            const uint k_idx = i / COLS;
            const uint n_idx = i % COLS;
            if (k_idx < k_tile_size) {
                smemB[k_idx * COLS + n_idx] = B[(k_base + k_idx) * COLS + n_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot product for this row/column
        if (row_valid) {
            const uint row_offset = m_idx * K + k_base;
            const device half *row_ptr = A + row_offset;
            const threadgroup half *shared_col = smemB + col_in_row;
            accum += gemv_dot_shared_device<half>(
                shared_col,
                COLS,
                row_ptr,
                1u,
                k_tile_size);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_valid) {
        C[m_idx * COLS + col_in_row] = half(accum);
    }
}
