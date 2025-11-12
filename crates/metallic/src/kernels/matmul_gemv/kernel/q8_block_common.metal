// Shared helpers for Q8 block GEMV dot products.

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

template <uint COLS>
ALWAYS_INLINE void q8_block_dot_multi(
    const device char *cached_ptrs[COLS],
    const float cached_scales[COLS],
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile,
    float contrib[COLS]) {
    q8_block_dot_core<COLS, false>(cached_ptrs, inner, count, local_base, x_tile, contrib);
    for (uint c = 0; c < COLS; ++c) {
        contrib[c] *= cached_scales[c];
    }
}

template <uint COLS>
ALWAYS_INLINE void q8_block_dot_multi_wide(
    const device char *cached_ptrs[COLS],
    const float cached_scales[COLS],
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile,
    float contrib[COLS]) {
    q8_block_dot_core<COLS, true>(cached_ptrs, inner, count, local_base, x_tile, contrib);
    for (uint c = 0; c < COLS; ++c) {
        contrib[c] *= cached_scales[c];
    }
}

template <uint COLS>
struct Q8HeadRef {
    uint N;
    uint scale_stride;
    uint scale_elem_bytes;
    uint data_stride;
    const device uchar *scales;
    const device uchar *data;
    thread float *sum;
};

template <uint COLS, uint HEADS>
ALWAYS_INLINE void q8_accumulate_heads_tile(
    thread const Q8HeadRef<COLS> (&heads)[HEADS],
    uint base_col,
    uint weights_per_block,
    uint tile_base,
    uint tile_limit,
    uint K,
    bool use_wide,
    threadgroup float *x_tile) {
    int last_block_idx[HEADS];
    for (uint h = 0u; h < HEADS; ++h) {
        last_block_idx[h] = -1;
    }

    float cached_scales[HEADS][COLS];
    const device char *cached_block_base[HEADS][COLS];

    uint k_local = 0u;
    while (k_local < tile_limit) {
        const uint k_abs = tile_base + k_local;
        const uint block_idx = k_abs >> 5;
        const uint inner = k_abs & 31u;
        const uint block_base_k = block_idx * weights_per_block;
        const uint block_valid = min(weights_per_block, K - block_base_k);
        const uint block_left = block_valid - inner;
        const uint tile_left = tile_limit - k_local;
        const uint count = (block_left < tile_left) ? block_left : tile_left;

        for (uint h = 0u; h < HEADS; ++h) {
            if ((int)block_idx != last_block_idx[h]) {
                for (uint c = 0u; c < COLS; ++c) {
                    const uint col = base_col + c;
                    const bool valid = (col < heads[h].N);
                    const uint scale_offset = block_idx * heads[h].scale_stride + col * heads[h].scale_elem_bytes;
                    const ushort lo = valid ? (ushort)heads[h].scales[scale_offset + 0u] : (ushort)0;
                    const ushort hi = valid ? (ushort)heads[h].scales[scale_offset + 1u] : (ushort)0;
                    const ushort bits = (ushort)(lo | (hi << 8));
                    cached_scales[h][c] = static_cast<float>(as_type<half>(bits));
                    const uint data_offset = block_idx * heads[h].data_stride + col * weights_per_block;
                    cached_block_base[h][c] = (const device char *)(heads[h].data + (valid ? data_offset : 0u));
                }
                last_block_idx[h] = (int)block_idx;
            }

            float contrib[COLS];
            for (uint c = 0u; c < COLS; ++c) {
                contrib[c] = 0.0f;
            }

            if (use_wide) {
                q8_block_dot_multi_wide<COLS>(cached_block_base[h], cached_scales[h], inner, count, k_local, x_tile, contrib);
            } else {
                q8_block_dot_multi<COLS>(cached_block_base[h], cached_scales[h], inner, count, k_local, x_tile, contrib);
            }

            for (uint c = 0u; c < COLS; ++c) {
                heads[h].sum[c] += contrib[c];
            }
        }

        k_local += count;
    }
}

template <typename Body>
ALWAYS_INLINE void q8_for_each_tile(
    uint K,
    const device half *vector_x,
    threadgroup float *x_tile,
    uint3 lid,
    uint dbuf_tile_k,
    uint threads_per_tg,
    Body body) {

    threadgroup float *x_buf0 = x_tile;
    threadgroup float *x_buf1 = x_tile + dbuf_tile_k;

    uint tile_base = 0u;
    uint cur_limit = min(dbuf_tile_k, K - tile_base);
    const uint thread_linear = lid.x;
    gemv_stage_vector_tile(vector_x, tile_base, cur_limit, threads_per_tg, thread_linear, x_buf0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (; tile_base < K; tile_base += dbuf_tile_k, buf ^= 1u) {
        const threadgroup float *cur_x = (buf == 0u) ? x_buf0 : x_buf1;
        cur_limit = min(dbuf_tile_k, K - tile_base);

        body(tile_base, cur_x, cur_limit);

        const uint next_base = tile_base + dbuf_tile_k;
        if (next_base < K) {
            const uint next_limit = min(dbuf_tile_k, K - next_base);
            threadgroup float *next_x = (buf == 0u) ? x_buf1 : x_buf0;
            gemv_stage_vector_tile(vector_x, next_base, next_limit, threads_per_tg, thread_linear, next_x);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint HEADS>
ALWAYS_INLINE void q8_run_fused_heads(
    thread const Q8HeadRef<GEMV_COLS_PER_THREAD> (&heads)[HEADS],
    uint base_col,
    uint weights_per_block,
    uint K,
    bool use_wide,
    const device half *vector_x,
    uint3 lid,
    threadgroup float *x_tile) {

    auto tile_body = [&](uint tile_base_local, const threadgroup float *cur_x, uint tile_limit) {
        q8_accumulate_heads_tile<GEMV_COLS_PER_THREAD>(
            heads,
            base_col,
            weights_per_block,
            tile_base_local,
            tile_limit,
            K,
            use_wide,
            const_cast<threadgroup float *>(cur_x));
    };

    const uint thread_linear = lid.x;
    if (K <= Q8_DBUF_TILE_K) {
        const uint tile_limit = K;
        gemv_stage_vector_tile(vector_x, 0u, tile_limit, THREADGROUP_WIDTH, thread_linear, x_tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        q8_accumulate_heads_tile<GEMV_COLS_PER_THREAD>(
            heads,
            base_col,
            weights_per_block,
            0u,
            tile_limit,
            K,
            use_wide,
            x_tile);
        return;
    }

    q8_for_each_tile(
        K,
        vector_x,
        x_tile,
        lid,
        Q8_DBUF_TILE_K,
        THREADGROUP_WIDTH,
        tile_body);
}

template <uint HEADS>
ALWAYS_INLINE void q8_write_fused_heads(
    const float sums[HEADS][GEMV_COLS_PER_THREAD],
    thread const Q8FusedHeadOut<GEMV_COLS_PER_THREAD> (&outs)[HEADS],
    uint base_col) {
    for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
        const uint out_idx = base_col + c;
        for (uint h = 0u; h < HEADS; ++h) {
            if (out_idx >= outs[h].N) {
                continue;
            }
            float value = sums[h][c];
            if (outs[h].has_bias != 0u) {
                value += static_cast<float>(outs[h].bias[out_idx]);
            }
            outs[h].out[out_idx] = static_cast<half>(value);
        }
    }
}
