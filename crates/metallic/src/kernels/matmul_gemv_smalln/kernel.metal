#include <metal_stdlib>
using namespace metal;

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
    const uint ROWS_PER_TG = 8;
    const uint COLS_PER_TG = 8;
    const uint TILE_K_SIZE = 64;

    threadgroup half smemB[TILE_K_SIZE * COLS_PER_TG];

    uint row_in_tg = tid / COLS_PER_TG;
    uint col_in_row = tid % COLS_PER_TG;

    uint m_idx = tgid.x * ROWS_PER_TG + row_in_tg;
    // Keep every thread alive for cooperative loads; guard accesses for rows outside M.
    bool row_valid = m_idx < M;

    float accum = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K_SIZE) {
        uint k_tile_size = min(TILE_K_SIZE, K - k_base);

        // Cooperative load of B tile into shared memory
        for (uint i = tid; i < k_tile_size * COLS_PER_TG; i += ROWS_PER_TG * COLS_PER_TG) {
            uint k_idx = i / COLS_PER_TG;
            uint n_idx = i % COLS_PER_TG;
            if (k_idx < k_tile_size) {
                smemB[k_idx * COLS_PER_TG + n_idx] = B[(k_base + k_idx) * 8 + n_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot product for this row and column
        if (row_valid) {
            uint row_offset = m_idx * K + k_base;
            for (uint k = 0; k < k_tile_size; ++k) {
                half a_val = A[row_offset + k];
                half b_val = smemB[k * COLS_PER_TG + col_in_row];
                accum += (float)a_val * (float)b_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_valid) {
        C[m_idx * COLS_PER_TG + col_in_row] = half(accum);
    }
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
    const uint ROWS_PER_TG = 16;
    const uint COLS_PER_TG = 4;
    const uint TILE_K_SIZE = 64;

    threadgroup half smemB[TILE_K_SIZE * COLS_PER_TG];

    uint row_in_tg = tid / COLS_PER_TG;
    uint col_in_row = tid % COLS_PER_TG;

    uint m_idx = tgid.y * ROWS_PER_TG + row_in_tg;
    if (m_idx >= M) return;

    float4 accum = float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (uint k_base = 0; k_base < K; k_base += TILE_K_SIZE) {
        uint k_tile_size = min(TILE_K_SIZE, K - k_base);

        for (uint i = tid; i < k_tile_size * COLS_PER_TG; i += 64) {
            uint k_idx = i / COLS_PER_TG;
            uint n_idx = i % COLS_PER_TG;
            if (k_idx < k_tile_size) {
                smemB[k_idx * COLS_PER_TG + n_idx] = B[(k_base + k_idx) * COLS_PER_TG + n_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < k_tile_size; ++k) {
            half4 a_val = half4(
                A[m_idx * K + (k_base + k)],
                A[m_idx * K + (k_base + k)],
                A[m_idx * K + (k_base + k)],
                A[m_idx * K + (k_base + k)]
            );
            half4 b_val = half4(
                smemB[k * COLS_PER_TG + 0],
                smemB[k * COLS_PER_TG + 1],
                smemB[k * COLS_PER_TG + 2],
                smemB[k * COLS_PER_TG + 3]
            );
            accum += float4(a_val * b_val);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    half4 result = half4(accum[0] + accum[1] + accum[2] + accum[3],
                         accum[0] + accum[1] + accum[2] + accum[3],
                         accum[0] + accum[1] + accum[2] + accum[3],
                         accum[0] + accum[1] + accum[2] + accum[3]);
    C[m_idx * 4 + col_in_row] = result[col_in_row];
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
    const uint ROWS_PER_TG = 32;
    const uint COLS_PER_TG = 1;
    const uint TILE_K_SIZE = 64;

    threadgroup half smemB[TILE_K_SIZE * COLS_PER_TG];

    uint row_in_tg = tid / COLS_PER_TG;
    uint col_in_row = tid % COLS_PER_TG;

    uint m_idx = tgid.x * ROWS_PER_TG + row_in_tg;
    if (m_idx >= M) return;

    float accum = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K_SIZE) {
        uint k_tile_size = min(TILE_K_SIZE, K - k_base);

        // Cooperative load of B tile into shared memory
        for (uint i = tid; i < k_tile_size * COLS_PER_TG; i += ROWS_PER_TG * COLS_PER_TG) {
            uint k_idx = i / COLS_PER_TG;
            uint n_idx = i % COLS_PER_TG;
            if (k_idx < k_tile_size) {
                smemB[k_idx * COLS_PER_TG + n_idx] = B[(k_base + k_idx) * 1 + n_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot product for this row
        if (row_in_tg < ROWS_PER_TG) {
            uint row_offset = m_idx * K + k_base;
            for (uint k = 0; k < k_tile_size; ++k) {
                half a_val = A[row_offset + k];
                half b_val = smemB[k * COLS_PER_TG + col_in_row];
                accum += (float)a_val * (float)b_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_in_tg < ROWS_PER_TG) {
        C[m_idx * COLS_PER_TG + col_in_row] = half(accum);
    }
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
    const uint ROWS_PER_TG = 32;
    const uint COLS_PER_TG = 2;
    const uint TILE_K_SIZE = 64;

    threadgroup half smemB[TILE_K_SIZE * COLS_PER_TG];

    uint row_in_tg = tid / COLS_PER_TG;
    uint col_in_row = tid % COLS_PER_TG;

    uint m_idx = tgid.x * ROWS_PER_TG + row_in_tg;
    if (m_idx >= M) return;

    float2 accum = float2(0.0f, 0.0f);

    for (uint k_base = 0; k_base < K; k_base += TILE_K_SIZE) {
        uint k_tile_size = min(TILE_K_SIZE, K - k_base);

        // Cooperative load of B tile into shared memory
        for (uint i = tid; i < k_tile_size * COLS_PER_TG; i += ROWS_PER_TG * COLS_PER_TG) {
            uint k_idx = i / COLS_PER_TG;
            uint n_idx = i % COLS_PER_TG;
            if (k_idx < k_tile_size) {
                smemB[k_idx * COLS_PER_TG + n_idx] = B[(k_base + k_idx) * 2 + n_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot product for this row
        if (row_in_tg < ROWS_PER_TG) {
            uint row_offset = m_idx * K + k_base;
            for (uint k = 0; k < k_tile_size; ++k) {
                half a_val = A[row_offset + k];
                half2 b_val = half2(
                    smemB[k * COLS_PER_TG + 0],
                    smemB[k * COLS_PER_TG + 1]
                );
                accum += float2(a_val * b_val);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_in_tg < ROWS_PER_TG) {
        half2 result = half2(accum[0], accum[1]);
        C[m_idx * 2 + col_in_row] = result[col_in_row];
    }
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
    const uint ROWS_PER_TG = 4;
    const uint COLS_PER_TG = 16;
    const uint TILE_K_SIZE = 64;

    threadgroup half smemB[TILE_K_SIZE * COLS_PER_TG];

    uint row_in_tg = tid / COLS_PER_TG;
    uint col_in_row = tid % COLS_PER_TG;

    uint m_idx = tgid.x * ROWS_PER_TG + row_in_tg;
    bool row_valid = m_idx < M;

    float accum = 0.0f;

    for (uint k_base = 0; k_base < K; k_base += TILE_K_SIZE) {
        uint k_tile_size = min(TILE_K_SIZE, K - k_base);

        // Cooperative load of B tile into shared memory
        for (uint i = tid; i < k_tile_size * COLS_PER_TG; i += ROWS_PER_TG * COLS_PER_TG) {
            uint k_idx = i / COLS_PER_TG;
            uint n_idx = i % COLS_PER_TG;
            if (k_idx < k_tile_size) {
                smemB[k_idx * COLS_PER_TG + n_idx] = B[(k_base + k_idx) * 16 + n_idx];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute dot product for this row and column
        if (row_valid) {
            uint row_offset = m_idx * K + k_base;
            for (uint k = 0; k < k_tile_size; ++k) {
                half a_val = A[row_offset + k];
                half b_val = smemB[k * COLS_PER_TG + col_in_row];
                accum += (float)a_val * (float)b_val;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row_valid) {
        C[m_idx * COLS_PER_TG + col_in_row] = half(accum);
    }
}
