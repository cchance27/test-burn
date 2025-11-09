// Shared GEMV core with loader/Epilog strategy.
#include <metal_stdlib>
using namespace metal;

struct GemvParams {
    uint K;
    uint N;
};

#define THREADGROUP_WIDTH 256u
#define TILE_N THREADGROUP_WIDTH
#define TILE_K 256u

// Plain F16 loader
struct PlainF16Loader {
    const device half *A;
    uint N;
    const device half *ptr;
    inline void begin(uint out_idx, uint tile_base) {
        ptr = A + tile_base * N + out_idx;
    }
    inline float next() {
        float v = (float)ptr[0];
        ptr += N;
        return v;
    }
};

// Q8_0 constants shared with quant kernels
constant uint Q8_BLOCK_WEIGHTS = 32u;
constant uint Q8_BLOCK_BYTES   = 34u; // 2 (scale f16) + 32 (i8)

inline float q8_read_scale(const device uchar* __restrict w, uint block_id) {
    const uint base = block_id * Q8_BLOCK_BYTES;
    ushort lo = (ushort)w[base];
    ushort hi = (ushort)w[base + 1];
    ushort bits = (ushort)(lo | (hi << 8));
    half h = as_type<half>(bits);
    return (float)h;
}

// Q8_0 loader for row-major [N,K] packed blocks along last dim (K)
struct Q8_0Loader {
    const device uchar *W;
    uint K;              // columns
    uint blocks_per_row; // ceil(K/32)
    const device char *qs;
    float scale;
    uint current_k;
    uint row_base_block; // n * blocks_per_row
    inline void load_block(uint block_idx) {
        const uint block_id = row_base_block + block_idx;
        const uint bb = block_id * Q8_BLOCK_BYTES;
        scale = q8_read_scale(W, block_id);
        qs = (const device char*)(W + bb + 2u);
    }
    inline void begin(uint out_idx, uint tile_base) {
        blocks_per_row = (K + 31u) >> 5;
        row_base_block = out_idx * blocks_per_row;
        current_k = tile_base;
        load_block(current_k >> 5);
    }
    inline float next() {
        uint block_idx = current_k >> 5;
        uint inner = current_k & 31u;
        float v = (float)qs[inner] * scale;
        current_k++;
        uint next_block_idx = current_k >> 5;
        if (next_block_idx != block_idx) {
            load_block(next_block_idx);
        }
        return v;
    }
};

// Q8_0 loader for row-major [K,N] packed blocks along last dim (N)
struct Q8_0LoaderKN {
    const device uchar *W;
    uint N; // columns
    uint blocks_per_row; // ceil(N/32)
    const device char *qs;
    float scale;
    uint inner;      // n & 31
    uint block_idx;  // n >> 5 within the row
    uint current_k;
    inline void load_block(uint k_row) {
        const uint block_id = k_row * blocks_per_row + block_idx;
        const uint bb = block_id * Q8_BLOCK_BYTES;
        scale = q8_read_scale(W, block_id);
        qs = (const device char*)(W + bb + 2u);
    }
    inline void begin(uint out_idx, uint tile_base) {
        blocks_per_row = (N + 31u) >> 5;
        block_idx = (out_idx >> 5);
        inner = out_idx & 31u;
        current_k = tile_base;
        load_block(current_k);
    }
    inline float next() {
        float v = (float)qs[inner] * scale;
        current_k++;
        load_block(current_k);
        return v;
    }
};

template <typename Loader>
inline void gemv_core_nobias(
    thread Loader &loader,
    const device half* __restrict x,
    device half*       __restrict y,
    const constant GemvParams *params,
    threadgroup float* x_tile,
    uint3 gid, uint3 lid)
{
    const uint N = params->N;
    const uint K = params->K;

    const uint out_idx = gid.x * TILE_N + lid.x;
    const bool is_active = out_idx < N;
    float acc = 0.0f;

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        for (uint local = lid.x; local < TILE_K; local += THREADGROUP_WIDTH) {
            const uint gk = tile_base + local;
            x_tile[local] = gk < K ? (float)x[gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (is_active) {
            loader.begin(out_idx, tile_base);
            const uint tile_limit = min(TILE_K, K - tile_base);
            uint local = 0u;
            for (; local + 3u < tile_limit; local += 4u) {
                acc = fma(x_tile[local + 0u], loader.next(), acc);
                acc = fma(x_tile[local + 1u], loader.next(), acc);
                acc = fma(x_tile[local + 2u], loader.next(), acc);
                acc = fma(x_tile[local + 3u], loader.next(), acc);
            }
            for (; local < tile_limit; ++local) {
                acc = fma(x_tile[local], loader.next(), acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (is_active) y[out_idx] = (half)acc;
}

template <typename Loader>
inline void gemv_core_bias(
    thread Loader &loader,
    const device half* __restrict x,
    device half*       __restrict y,
    const constant GemvParams *params,
    const device half* __restrict bias,
    threadgroup float* x_tile,
    uint3 gid, uint3 lid)
{
    const uint N = params->N;
    const uint K = params->K;

    const uint out_idx = gid.x * TILE_N + lid.x;
    const bool is_active = out_idx < N;
    float acc = 0.0f;

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        for (uint local = lid.x; local < TILE_K; local += THREADGROUP_WIDTH) {
            const uint gk = tile_base + local;
            x_tile[local] = gk < K ? (float)x[gk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (is_active) {
            loader.begin(out_idx, tile_base);
            const uint tile_limit = min(TILE_K, K - tile_base);
            uint local = 0u;
            for (; local + 3u < tile_limit; local += 4u) {
                acc = fma(x_tile[local + 0u], loader.next(), acc);
                acc = fma(x_tile[local + 1u], loader.next(), acc);
                acc = fma(x_tile[local + 2u], loader.next(), acc);
                acc = fma(x_tile[local + 3u], loader.next(), acc);
            }
            for (; local < tile_limit; ++local) {
                acc = fma(x_tile[local], loader.next(), acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (is_active) y[out_idx] = (half)(acc + (float)bias[out_idx]);
}
