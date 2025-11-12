// Copyright © 2024 Apple Inc.

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

#define THREADGROUP_WIDTH 256u
#define GEMV_COLS_PER_THREAD 1u

#define TILE_N (THREADGROUP_WIDTH * GEMV_COLS_PER_THREAD)
#define TILE_K 8192u

constant uint LOAD_LANES = THREADGROUP_WIDTH;
#define Q8_DBUF_TILE_K (TILE_K / 2u)

#define Q8_BLOCK_BYTES 34u
#define Q8_CANONICAL_SCALE_BYTES 2u
#define Q8_0_WEIGHTS_PER_BLOCK 32u

template <uint COLS, bool Wide>
ALWAYS_INLINE void q8_block_dot_core(
    const device char *qs_ptrs_in[COLS],
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile,
    float contrib[COLS]);

template <typename MatrixT>
struct MatrixPointerAccessor {
    const device MatrixT *matrix;
    const device MatrixT *ptr;
    uint stride;

    inline void init(const device MatrixT *matrix_, uint stride_) {
        matrix = matrix_;
        stride = stride_;
    }

    inline void begin(uint out_idx, uint tile_base) {
        ptr = matrix + tile_base * stride + out_idx;
    }

    inline float load() {
        float value = static_cast<float>(ptr[0]);
        ptr += stride;
        return value;
    }
};

inline float q8nk_block_dot(
    const device uchar *W,
    uint block_idx,
    uint inner_offset,
    uint count,
    uint local_idx,
    uint out_block_stride,
    uint out_base_offset,
    threadgroup float *x_tile) {

    if (count == 0) {
        return 0.0f;
    }

    const uint block_byte_offset = block_idx * out_block_stride + out_base_offset;
    ushort scale_bits = ((ushort)W[block_byte_offset]) | (((ushort)W[block_byte_offset + 1]) << 8);
    const float scale = static_cast<float>(as_type<half>(scale_bits));
    const uint data_offset = block_byte_offset + 2u + inner_offset;
    const device char *ptr = (const device char *)(W + data_offset);

    float acc_scalar = 0.0f;
    float acc_vec0 = 0.0f;
    float acc_vec1 = 0.0f;
    uint i = 0;

    uint mis = data_offset & 3u;
    if (mis != 0u) {
        const uint align_count = min(count, 4u - mis);
        for (; i < align_count; ++i) {
            acc_scalar = fma(x_tile[local_idx + i], (float)ptr[0], acc_scalar);
            ptr += 1;
        }
    }

    for (; i + 8u <= count; i += 8u) {
        const device char4 *qv0 = (const device char4 *)(ptr);
        const device char4 *qv1 = (const device char4 *)(ptr + 4u);
        char4 q0 = *qv0;
        char4 q1 = *qv1;
        const uint base_local = local_idx + i;
        float4 x0 = float4(
            x_tile[base_local + 0u],
            x_tile[base_local + 1u],
            x_tile[base_local + 2u],
            x_tile[base_local + 3u]);
        float4 x1 = float4(
            x_tile[base_local + 4u],
            x_tile[base_local + 5u],
            x_tile[base_local + 6u],
            x_tile[base_local + 7u]);
        acc_vec0 += dot(x0, float4(q0));
        acc_vec1 += dot(x1, float4(q1));
        ptr += 8u;
    }

    for (; i + 4u <= count; i += 4u) {
        const device char4 *qv = (const device char4 *)(ptr);
        char4 q = *qv;
        const uint base_local = local_idx + i;
        float4 x = float4(
            x_tile[base_local + 0u],
            x_tile[base_local + 1u],
            x_tile[base_local + 2u],
            x_tile[base_local + 3u]);
        acc_scalar += dot(x, float4(q));
        ptr += 4u;
    }

    for (; i < count; ++i) {
        acc_scalar = fma(x_tile[local_idx + i], (float)ptr[0], acc_scalar);
        ptr += 1;
    }

    return scale * (acc_scalar + acc_vec0 + acc_vec1);
}

ALWAYS_INLINE float q8_canonical_block_dot(
    const device char *qs,
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile) {

    if (count == 0u) {
        return 0.0f;
    }
    const device char *qs_ptrs[1] = {qs};
    float contrib[1] = {0.0f};
    q8_block_dot_core<1u, false>(qs_ptrs, inner, count, local_base, x_tile, contrib);
    return contrib[0];
}

ALWAYS_INLINE float q8_canonical_block_dot_wide(
    const device char *qs,
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile) {

    if (count == 0u) {
        return 0.0f;
    }
    const device char *qs_ptrs[1] = {qs};
    float contrib[1] = {0.0f};
    q8_block_dot_core<1u, true>(qs_ptrs, inner, count, local_base, x_tile, contrib);
    return contrib[0];
}

struct Q8MatrixAccessorKN {
    const device uchar *W;
    uint N;
    uint blocks_per_row;
    const device char *qs;
    float scale;
    uint inner;
    uint block_idx;
    uint current_k;
    uint base_bytes;
    uint stride_bytes;

    inline void init(const device uchar *W_, uint N_) {
        W = W_;
        N = N_;
    }

    inline void begin(uint out_idx, uint tile_base) {
        blocks_per_row = (N + 31u) >> 5;
        inner = out_idx & 31u;
        block_idx = out_idx >> 5;
        current_k = tile_base;
        stride_bytes = blocks_per_row * Q8_BLOCK_BYTES;
        base_bytes = (tile_base * blocks_per_row + block_idx) * Q8_BLOCK_BYTES;
        const uint base = base_bytes;
        ushort lo = (ushort)W[base];
        ushort hi = (ushort)W[base + 1];
        ushort bits = (ushort)(lo | (hi << 8));
        scale = static_cast<float>(as_type<half>(bits));
        qs = (const device char *)(W + base + 2u);
    }

    inline void load_block_next() {
        base_bytes += stride_bytes;
        const uint base = base_bytes;
        ushort lo = (ushort)W[base];
        ushort hi = (ushort)W[base + 1];
        ushort bits = (ushort)(lo | (hi << 8));
        scale = static_cast<float>(as_type<half>(bits));
        qs = (const device char *)(W + base + 2u);
    }

    inline float load() {
        float value = (float)qs[inner] * scale;
        current_k++;
        load_block_next();
        return value;
    }
};

template <bool HasBias>
struct GemvBiasReader {
    template <typename Scalar>
    inline static float load(const device Scalar *, uint) {
        return 0.0f;
    }
};

template <>
struct GemvBiasReader<true> {
    template <typename Scalar>
    inline static float load(const device Scalar *bias_ptr, uint out_idx) {
        return static_cast<float>(bias_ptr[out_idx]);
    }
};

template <typename SharedScalar>
ALWAYS_INLINE float gemv_dot_shared_device(
    const threadgroup SharedScalar *shared_base,
    uint shared_stride,
    const device half *device_ptr,
    uint device_stride,
    uint tile_limit) {
    float acc = 0.0f;
    const device half *ptr = device_ptr;
    uint idx = 0u;
    for (; idx + 4u <= tile_limit; idx += 4u) {
        const float s0 = static_cast<float>(shared_base[(idx + 0u) * shared_stride]);
        const float d0 = static_cast<float>(ptr[0]);
        ptr += device_stride;
        const float s1 = static_cast<float>(shared_base[(idx + 1u) * shared_stride]);
        const float d1 = static_cast<float>(ptr[0]);
        ptr += device_stride;
        const float s2 = static_cast<float>(shared_base[(idx + 2u) * shared_stride]);
        const float d2 = static_cast<float>(ptr[0]);
        ptr += device_stride;
        const float s3 = static_cast<float>(shared_base[(idx + 3u) * shared_stride]);
        const float d3 = static_cast<float>(ptr[0]);
        ptr += device_stride;
        acc = fma(s0, d0, acc);
        acc = fma(s1, d1, acc);
        acc = fma(s2, d2, acc);
        acc = fma(s3, d3, acc);
    }
    for (; idx < tile_limit; ++idx) {
        const float s = static_cast<float>(shared_base[idx * shared_stride]);
        const float d = static_cast<float>(ptr[0]);
        acc = fma(s, d, acc);
        ptr += device_stride;
    }
    return acc;
}

ALWAYS_INLINE void gemv_stage_vector_tile(
    const device half *vector_x,
    uint tile_base,
    uint tile_limit,
    uint threads_per_tg,
    uint thread_linear,
    threadgroup float *dest) {
    for (uint local = thread_linear; local < tile_limit; local += threads_per_tg) {
        dest[local] = static_cast<float>(vector_x[tile_base + local]);
    }
}

template <bool HasBias>
ALWAYS_INLINE float gemv_epilogue_value(
    float accum,
    const device half *bias_ptr,
    const device half *residual_ptr,
    float alpha,
    float beta,
    uint out_idx) {
    float bias_val = GemvBiasReader<HasBias>::template load<half>(bias_ptr, out_idx);
    float residual_val = (beta != 0.0f && residual_ptr != (const device half *)nullptr)
        ? static_cast<float>(residual_ptr[out_idx])
        : 0.0f;
    return alpha * (accum + bias_val) + beta * residual_val;
}

template <bool HasBias>
ALWAYS_INLINE void gemv_store_results(
    const float sum[GEMV_COLS_PER_THREAD],
    const device half *bias_ptr,
    const device half *residual_ptr,
    float alpha,
    float beta,
    uint base_col,
    uint N,
    device half *result_y) {
    for (uint c = 0u; c < GEMV_COLS_PER_THREAD; ++c) {
        const uint out_idx = base_col + c;
        if (out_idx >= N) {
            break;
        }
        const float value = gemv_epilogue_value<HasBias>(sum[c], bias_ptr, residual_ptr, alpha, beta, out_idx);
        result_y[out_idx] = static_cast<half>(value);
    }
}

// Shared multi-head accumulation helpers

template <uint COLS, bool Wide>
ALWAYS_INLINE void q8_block_dot_core(
    const device char *qs_ptrs_in[COLS],
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile,
    float contrib[COLS]) {

    if (count == 0u) {
        return;
    }

    const uint misalign = inner & 3u;
    uint processed = 0u;

    const device char *qs_ptrs[COLS];
    for (uint c = 0; c < COLS; ++c) {
        qs_ptrs[c] = qs_ptrs_in[c];
    }

    if (misalign != 0u) {
        const uint align_count = min(count, 4u - misalign);
        for (uint i = 0; i < align_count; ++i) {
            const float x = x_tile[local_base + processed + i];
            for (uint c = 0; c < COLS; ++c) {
                contrib[c] = fma(x, (float)qs_ptrs[c][i], contrib[c]);
            }
        }
        for (uint c = 0; c < COLS; ++c) {
            qs_ptrs[c] += align_count;
        }
        processed += align_count;
    }

    if (Wide) {
        while (processed + 32u <= count) {
            const uint base_local = local_base + processed;
            const float4 x0 = float4(x_tile[base_local + 0u],  x_tile[base_local + 1u],  x_tile[base_local + 2u],  x_tile[base_local + 3u]);
            const float4 x1 = float4(x_tile[base_local + 4u],  x_tile[base_local + 5u],  x_tile[base_local + 6u],  x_tile[base_local + 7u]);
            const float4 x2 = float4(x_tile[base_local + 8u],  x_tile[base_local + 9u],  x_tile[base_local + 10u], x_tile[base_local + 11u]);
            const float4 x3 = float4(x_tile[base_local + 12u], x_tile[base_local + 13u], x_tile[base_local + 14u], x_tile[base_local + 15u]);
            const float4 x4 = float4(x_tile[base_local + 16u], x_tile[base_local + 17u], x_tile[base_local + 18u], x_tile[base_local + 19u]);
            const float4 x5 = float4(x_tile[base_local + 20u], x_tile[base_local + 21u], x_tile[base_local + 22u], x_tile[base_local + 23u]);
            const float4 x6 = float4(x_tile[base_local + 24u], x_tile[base_local + 25u], x_tile[base_local + 26u], x_tile[base_local + 27u]);
            const float4 x7 = float4(x_tile[base_local + 28u], x_tile[base_local + 29u], x_tile[base_local + 30u], x_tile[base_local + 31u]);
            for (uint c = 0; c < COLS; ++c) {
                const device char4 *qv = (const device char4 *)(qs_ptrs[c]);
                const char4 q0 = qv[0];
                const char4 q1 = qv[1];
                const char4 q2 = qv[2];
                const char4 q3 = qv[3];
                const char4 q4 = qv[4];
                const char4 q5 = qv[5];
                const char4 q6 = qv[6];
                const char4 q7 = qv[7];
                float tmp = 0.0f;
                tmp += dot(x0, float4(q0));
                tmp += dot(x1, float4(q1));
                tmp += dot(x2, float4(q2));
                tmp += dot(x3, float4(q3));
                tmp += dot(x4, float4(q4));
                tmp += dot(x5, float4(q5));
                tmp += dot(x6, float4(q6));
                tmp += dot(x7, float4(q7));
                contrib[c] += tmp;
                qs_ptrs[c] += 32u;
            }
            processed += 32u;
        }

        while (processed + 16u <= count) {
            const uint base_local = local_base + processed;
            const float4 x0 = float4(x_tile[base_local + 0u],  x_tile[base_local + 1u],  x_tile[base_local + 2u],  x_tile[base_local + 3u]);
            const float4 x1 = float4(x_tile[base_local + 4u],  x_tile[base_local + 5u],  x_tile[base_local + 6u],  x_tile[base_local + 7u]);
            const float4 x2 = float4(x_tile[base_local + 8u],  x_tile[base_local + 9u],  x_tile[base_local + 10u], x_tile[base_local + 11u]);
            const float4 x3 = float4(x_tile[base_local + 12u], x_tile[base_local + 13u], x_tile[base_local + 14u], x_tile[base_local + 15u]);
            for (uint c = 0; c < COLS; ++c) {
                const device char4 *qv = (const device char4 *)(qs_ptrs[c]);
                const char4 q0 = qv[0];
                const char4 q1 = qv[1];
                const char4 q2 = qv[2];
                const char4 q3 = qv[3];
                float tmp = 0.0f;
                tmp += dot(x0, float4(q0));
                tmp += dot(x1, float4(q1));
                tmp += dot(x2, float4(q2));
                tmp += dot(x3, float4(q3));
                contrib[c] += tmp;
                qs_ptrs[c] += 16u;
            }
            processed += 16u;
        }
    }

    while (processed + 8u <= count) {
        const uint base_local = local_base + processed;
        const float4 x0 = float4(
            x_tile[base_local + 0u], x_tile[base_local + 1u], x_tile[base_local + 2u], x_tile[base_local + 3u]);
        const float4 x1 = float4(
            x_tile[base_local + 4u], x_tile[base_local + 5u], x_tile[base_local + 6u], x_tile[base_local + 7u]);
        for (uint c = 0; c < COLS; ++c) {
            const device char4 *qv0 = (const device char4 *)(qs_ptrs[c]);
            const device char4 *qv1 = (const device char4 *)(qs_ptrs[c] + 4u);
            const char4 q0 = *qv0;
            const char4 q1 = *qv1;
            contrib[c] += dot(x0, float4(q0)) + dot(x1, float4(q1));
            qs_ptrs[c] += 8u;
        }
        processed += 8u;
    }

    while (processed + 4u <= count) {
        const uint base_local = local_base + processed;
        const float4 x = float4(
            x_tile[base_local + 0u], x_tile[base_local + 1u], x_tile[base_local + 2u], x_tile[base_local + 3u]);
        for (uint c = 0; c < COLS; ++c) {
            const char4 q = *(const device char4 *)(qs_ptrs[c]);
            contrib[c] += dot(x, float4(q));
            qs_ptrs[c] += 4u;
        }
        processed += 4u;
    }

    while (processed < count) {
        const float x = x_tile[local_base + processed];
        for (uint c = 0; c < COLS; ++c) {
            contrib[c] = fma(x, (float)qs_ptrs[c][0], contrib[c]);
            qs_ptrs[c] += 1u;
        }
        processed += 1u;
    }
}

// (Definition copied from original kernel.metal)
template <typename Body>
ALWAYS_INLINE void q8_for_each_block_k(
    uint blocks_per_k,
    uint weights_per_block,
    uint K,
    Body body) {
    for (uint block = 0u; block < blocks_per_k; ++block) {
        const uint block_k_start = block * weights_per_block;
        if (block_k_start >= K) {
            break;
        }
        const uint valid = min(weights_per_block, K - block_k_start);
        body(block, block_k_start, valid);
    }
}
