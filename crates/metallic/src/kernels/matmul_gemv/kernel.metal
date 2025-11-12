// Copyright © 2024 Apple Inc.

#include <metal_stdlib>

using namespace metal;

struct GemvParams {
    uint K;
    uint N;
    uint blocks_per_k;
    uint weights_per_block;
};

#define THREADGROUP_WIDTH 256u
#define TILE_N THREADGROUP_WIDTH
// Increase K tile to reduce passes over K for larger inner dimensions (e.g., K=4864).
// Shared memory usage: TILE_K * sizeof(float) = 4096 * 4 = 16KB, safe on Apple GPUs.
// Favor single K-pass on decode-sized K=4864 by using 8192.
// Shared memory for x_tile: 8192 * 4 bytes = 32KB (within typical tg limits).
#define TILE_K 8192u
// Widen cooperative load lanes to fully utilize the threadgroup for staging x.
constant uint LOAD_LANES = THREADGROUP_WIDTH;

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

// GGML Q8_0 block layout: half scale (d) then 32 int8 values (qs)
// Total bytes per block = 2 (fp16) + 32 = 34
// Perhaps instead of enum being a caller we could pass the block size and use that so we don't need a specific q8,q4, etc loader?
constant uint Q8_BLOCK_BYTES = 34u;
constant uint Q8_CANONICAL_SCALE_BYTES = 2u;
constant uint Q8_0_WEIGHTS_PER_BLOCK = 32u;

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

inline float q8_canonical_block_dot(
    const device char *qs,
    uint inner,
    uint count,
    uint local_base,
    threadgroup float *x_tile) {

    if (count == 0u) {
        return 0.0f;
    }

    float acc_scalar = 0.0f;
    float acc_vec0 = 0.0f;
    float acc_vec1 = 0.0f;
    uint processed = 0;

    const uint misalign = inner & 3u;
    if (misalign != 0u) {
        const uint align_count = min(count, 4u - misalign);
        for (uint i = 0; i < align_count; ++i) {
            acc_scalar = fma(x_tile[local_base + processed + i], (float)qs[i], acc_scalar);
        }
        qs += align_count;
        processed += align_count;
    }

    // Unrolled 32-element processing (8 x char4) to reduce loop/control overhead
    while (processed + 32u <= count) {
        const uint base_local = local_base + processed;
        const device char4 *qv = (const device char4 *)(qs);
        char4 q0 = qv[0];
        char4 q1 = qv[1];
        char4 q2 = qv[2];
        char4 q3 = qv[3];
        char4 q4 = qv[4];
        char4 q5 = qv[5];
        char4 q6 = qv[6];
        char4 q7 = qv[7];
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
        float4 x2 = float4(
            x_tile[base_local + 8u],
            x_tile[base_local + 9u],
            x_tile[base_local + 10u],
            x_tile[base_local + 11u]);
        float4 x3 = float4(
            x_tile[base_local + 12u],
            x_tile[base_local + 13u],
            x_tile[base_local + 14u],
            x_tile[base_local + 15u]);
        float4 x4 = float4(
            x_tile[base_local + 16u],
            x_tile[base_local + 17u],
            x_tile[base_local + 18u],
            x_tile[base_local + 19u]);
        float4 x5 = float4(
            x_tile[base_local + 20u],
            x_tile[base_local + 21u],
            x_tile[base_local + 22u],
            x_tile[base_local + 23u]);
        float4 x6 = float4(
            x_tile[base_local + 24u],
            x_tile[base_local + 25u],
            x_tile[base_local + 26u],
            x_tile[base_local + 27u]);
        float4 x7 = float4(
            x_tile[base_local + 28u],
            x_tile[base_local + 29u],
            x_tile[base_local + 30u],
            x_tile[base_local + 31u]);

        float tmp = 0.0f;
        tmp += dot(x0, float4(q0));
        tmp += dot(x1, float4(q1));
        tmp += dot(x2, float4(q2));
        tmp += dot(x3, float4(q3));
        tmp += dot(x4, float4(q4));
        tmp += dot(x5, float4(q5));
        tmp += dot(x6, float4(q6));
        tmp += dot(x7, float4(q7));
        acc_scalar += tmp;
        qs += 32u;
        processed += 32u;
    }

    // Unrolled 16-element processing (4 x char4) to reduce loop/control overhead
    while (processed + 16u <= count) {
        const uint base_local = local_base + processed;
        const device char4 *qv = (const device char4 *)(qs);
        char4 q0 = qv[0];
        char4 q1 = qv[1];
        char4 q2 = qv[2];
        char4 q3 = qv[3];
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
        float4 x2 = float4(
            x_tile[base_local + 8u],
            x_tile[base_local + 9u],
            x_tile[base_local + 10u],
            x_tile[base_local + 11u]);
        float4 x3 = float4(
            x_tile[base_local + 12u],
            x_tile[base_local + 13u],
            x_tile[base_local + 14u],
            x_tile[base_local + 15u]);
        acc_scalar += dot(x0, float4(q0));
        acc_vec0   += dot(x1, float4(q1));
        acc_vec1   += dot(x2, float4(q2));
        acc_scalar += dot(x3, float4(q3));
        qs += 16u;
        processed += 16u;
    }

    while (processed + 8u <= count) {
        const device char4 *qv0 = (const device char4 *)(qs);
        const device char4 *qv1 = (const device char4 *)(qs + 4u);
        const uint base_local = local_base + processed;
        char4 q0 = *qv0;
        char4 q1 = *qv1;
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
        qs += 8u;
        processed += 8u;
    }

    while (processed + 4u <= count) {
        const uint base_local = local_base + processed;
        char4 q = *(const device char4 *)(qs);
        float4 x = float4(
            x_tile[base_local + 0u],
            x_tile[base_local + 1u],
            x_tile[base_local + 2u],
            x_tile[base_local + 3u]);
        acc_scalar += dot(x, float4(q));
        qs += 4u;
        processed += 4u;
    }

    while (processed < count) {
        acc_scalar = fma(x_tile[local_base + processed], (float)qs[0], acc_scalar);
        qs += 1u;
        processed += 1u;
    }

    return acc_scalar + acc_vec0 + acc_vec1;
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

template <typename MatrixAccessor, typename VectorT, typename Scalar, bool HasBias>
inline void gemv_kernel(
    MatrixAccessor matrix_accessor,
    const device VectorT *vector_x,
    device Scalar *result_y,
    const constant GemvParams *params,
    const device Scalar *bias_ptr,
    uint3 gid,
    uint3 lid,
    threadgroup float *shared_x_tile) {

    const uint N = params->N;
    const uint K = params->K;

    const uint out_idx = gid.x * TILE_N + lid.x;
    const bool is_active = out_idx < N;

    float sum = 0.0f;

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        const uint tile_limit = min(TILE_K, K - tile_base);
        // Stage a tile of the input vector in shared memory so threads reuse it.
        if (lid.x < LOAD_LANES) {
            for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                const uint global_k = tile_base + local;
                shared_x_tile[local] = static_cast<float>(vector_x[global_k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_active) {
            matrix_accessor.begin(out_idx, tile_base);
            const uint tile_limit = min(TILE_K, K - tile_base);
            uint local = 0;
            for (; local + 3 < tile_limit; local += 4) {
                sum = fma(shared_x_tile[local], matrix_accessor.load(), sum);
                sum = fma(shared_x_tile[local + 1], matrix_accessor.load(), sum);
                sum = fma(shared_x_tile[local + 2], matrix_accessor.load(), sum);
                sum = fma(shared_x_tile[local + 3], matrix_accessor.load(), sum);
            }

            for (; local < tile_limit; ++local) {
                sum = fma(shared_x_tile[local], matrix_accessor.load(), sum);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (is_active) {
        float bias_val = GemvBiasReader<HasBias>::template load<Scalar>(bias_ptr, out_idx);
        result_y[out_idx] = static_cast<Scalar>(sum + bias_val);
    }
}

[[kernel]] void gemv_f32(
    const device float *matrix_a [[buffer(0)]],
    const device float *vector_x [[buffer(1)]],
    device float *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device float *bias [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float x_tile[TILE_K];
    MatrixPointerAccessor<float> accessor;
    accessor.init(matrix_a, params->N);
    if (has_bias) {
        gemv_kernel<MatrixPointerAccessor<float>, float, float, true>(
            accessor,
            vector_x,
            result_y,
            params,
            bias,
            gid,
            lid,
            x_tile);
    } else {
        gemv_kernel<MatrixPointerAccessor<float>, float, float, false>(
            accessor,
            vector_x,
            result_y,
            params,
            bias,
            gid,
            lid,
            x_tile);
    }
}

enum GemvLoaderMode : uint {
    GemvLoaderDense = 0,
    GemvLoaderDenseBias = 1,
    GemvLoaderQ8Canonical = 2,
    GemvLoaderQ8CanonicalBias = 3,
    GemvLoaderQ8CanonicalDebug = 4,
};

template <bool HasBias>
inline void run_gemv_dense(
    const device half *matrix_data,
    const device half *vector_x,
    device half *result_y,
    const constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    const float alpha,
    const float beta,
    uint3 gid,
    uint3 lid,
    threadgroup float *x_tile) {

    const uint N = params->N;
    const uint K = params->K;

    const uint out_idx = gid.x * TILE_N + lid.x;
    const bool is_active = out_idx < N;

    float sum = 0.0f;

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        const uint tile_limit = min(TILE_K, K - tile_base);
        if (lid.x < LOAD_LANES) {
            for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                const uint global_k = tile_base + local;
                x_tile[local] = static_cast<float>(vector_x[global_k]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_active) {
            const uint tile_limit = min(TILE_K, K - tile_base);
            const device half *matrix_ptr = matrix_data + tile_base * N + out_idx;

            uint local = 0;
            for (; local + 3 < tile_limit; local += 4) {
                sum = fma(x_tile[local], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
                sum = fma(x_tile[local + 1], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
                sum = fma(x_tile[local + 2], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
                sum = fma(x_tile[local + 3], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
            }

            for (; local < tile_limit; ++local) {
                sum = fma(x_tile[local], static_cast<float>(matrix_ptr[0]), sum);
                matrix_ptr += N;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (is_active) {
        float bias_val = GemvBiasReader<HasBias>::template load<half>(bias, out_idx);
        float c = (beta != 0.0f && residual != (const device half*)nullptr) ? static_cast<float>(residual[out_idx]) : 0.0f;
        float y = alpha * (sum + bias_val) + beta * c;
        result_y[out_idx] = static_cast<half>(y);
    }
}

template <bool HasBias, bool Debug>
void run_gemv_q8_canonical(
    const device uchar *data,
    const device uchar *scale_bytes,
    const device half *vector_x,
    device half *result_y,
    const constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    const float alpha,
    const float beta,
    const constant uint &diag_col,
    uint3 gid,
    uint3 lid,
    threadgroup float *x_tile) {

    const uint N = params->N;
    const uint K = params->K;
    const uint weights_per_block = params->weights_per_block;
    const uint blocks_per_k = params->blocks_per_k;
    const uint out_idx = gid.x * TILE_N + lid.x;
    const bool is_active = out_idx < N;
    if (!is_active || weights_per_block == 0u) {
        return;
    }

    float sum = 0.0f;
    const bool is_diag = Debug && (out_idx == diag_col);

    if constexpr (Debug) {
        if (is_diag) {
            for (uint b = 0; b < blocks_per_k; ++b) {
                result_y[b] = static_cast<half>(0.0);
            }
        }
    }

    // Process K in tiles with x staged in shared memory for reuse
    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        const uint tile_limit = min(TILE_K, K - tile_base);
        // Stage x into shared memory once per tile
        if (lid.x < LOAD_LANES) {
            for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                const uint gk = tile_base + local;
                x_tile[local] = static_cast<float>(vector_x[gk]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (is_active) {
            const uint k_remain = tile_limit;
            uint k_local = 0;
            while (k_local < k_remain) {
                const uint k_abs = tile_base + k_local;
                const uint block_idx = k_abs >> 5;           // 32-elem block id
                const uint inner     = k_abs & 31u;           // offset within block
                const uint base_idx = block_idx * N + out_idx;
                const uint sb = base_idx * 2u;
                const ushort bits = (ushort)scale_bytes[sb] | ((ushort)scale_bytes[sb + 1] << 8);
                const float scale = static_cast<float>(as_type<half>(bits));
                const uint base_byte = base_idx * weights_per_block + inner;
                const device char *qs = (const device char *)(data + base_byte);
                const uint count = min((uint)(weights_per_block - inner), k_remain - k_local);

                float block_sum = q8_canonical_block_dot(qs, inner, count, k_local, x_tile);
                const float contrib = scale * block_sum;
                if constexpr (Debug) {
                    if (is_diag) {
                        const float prev = static_cast<float>(result_y[block_idx]);
                        result_y[block_idx] = static_cast<half>(prev + contrib);
                    }
                } else {
                    sum += contrib;
                }
                k_local += count;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if constexpr (!Debug) {
        if (!is_active) {
            return;
        }
        float bias_val = GemvBiasReader<HasBias>::template load<half>(bias, out_idx);
        float c = (beta != 0.0f && residual != (const device half*)nullptr) ? static_cast<float>(residual[out_idx]) : 0.0f;
        float y = alpha * (sum + bias_val) + beta * c;
        result_y[out_idx] = static_cast<half>(y);
    }
}

// Use a typed half* for buffer(0) to preserve alignment/vectorization on dense FP16 path.
// Dense FP16 no-bias entry
[[kernel]] void gemv_f16_dense(
    const device half *matrix_a [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float x_tile[TILE_K];
    run_gemv_dense<false>(matrix_a, vector_x, result_y, params, (const device half *)nullptr, (const device half *)nullptr, 1.0f, 0.0f, gid, lid, x_tile);
}

// Unified GEMV entry for dense FP16 and Q8 using loader_mode
[[kernel]] void gemv_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float x_tile[TILE_K];

    // Prefer canonical Q8 routing whenever metadata indicates quantized layout,
    // regardless of loader_mode (defensive against host-side mismatches).
    if (params->weights_per_block != 0u && loader_mode != GemvLoaderQ8CanonicalDebug) {
        if (loader_mode == GemvLoaderQ8CanonicalBias || loader_mode == GemvLoaderDenseBias) {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
        } else {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
        }
        return;
    }
    switch (loader_mode) {
        case GemvLoaderDense: {
            const device half *matrix_a = (const device half *)matrix_data;
            run_gemv_dense<false>(matrix_a, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, x_tile);
            return;
        }
        case GemvLoaderDenseBias: {
            const device half *matrix_a = (const device half *)matrix_data;
            run_gemv_dense<true>(matrix_a, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, x_tile);
            return;
        }
        case GemvLoaderQ8Canonical: {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
            return;
        }
        case GemvLoaderQ8CanonicalBias: {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
            return;
        }
        case GemvLoaderQ8CanonicalDebug: {
            run_gemv_q8_canonical<false, true>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
            return;
        }
        default: {
            const device half *matrix_a = (const device half *)matrix_data;
            run_gemv_dense<false>(matrix_a, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, x_tile);
            return;
        }
    }
}

struct QkvFusedParams {
    uint K;
    uint Nq;
    uint Nk;
    uint Nv;
    uint blocks_per_k;
    uint weights_per_block;
    uint has_bias_q;
    uint has_bias_k;
    uint has_bias_v;
};

struct GemmQ8NtParams {
    uint m;
    uint n;
    uint k;
    uint lda;
    uint ldc;
    uint blocks_per_k;
    uint weights_per_block;
    uint has_bias;
};

// Fused QKV GEMV for canonical Q8 weights (supports distinct N per row).
// Computes three outputs (Q, K, V) in a single pass of K, sharing the staged x_tile.
[[kernel]] void gemv_q8_fused3_f16(
    const device uchar *data_q [[buffer(0)]],
    const device uchar *data_k [[buffer(1)]],
    const device uchar *data_v [[buffer(2)]],
    const device half *vector_x [[buffer(3)]],
    device half *out_q [[buffer(4)]],
    device half *out_k [[buffer(5)]],
    device half *out_v [[buffer(6)]],
    const constant QkvFusedParams *params [[buffer(7)]],
    const device uchar *scales_q [[buffer(8)]],
    const device uchar *scales_k [[buffer(9)]],
    const device uchar *scales_v [[buffer(10)]],
    const device half *bias_q [[buffer(11)]],
    const device half *bias_k [[buffer(12)]],
    const device half *bias_v [[buffer(13)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    threadgroup float x_tile[TILE_K];

    const uint NQ = params->Nq;
    const uint NK = params->Nk;
    const uint NV = params->Nv;
    const uint K = params->K;
    const uint weights_per_block = params->weights_per_block;
    const uint out_idx = gid.x * TILE_N + lid.x;
    if (weights_per_block == 0u) {
        return;
    }

    float sum_q = 0.0f;
    float sum_k = 0.0f;
    float sum_v = 0.0f;

    for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
        const uint tile_limit = min(TILE_K, K - tile_base);
        if (lid.x < LOAD_LANES) {
            for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                const uint gk = tile_base + local;
                x_tile[local] = static_cast<float>(vector_x[gk]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k_remain = tile_limit;
        uint k_local = 0;
        while (k_local < k_remain) {
            const uint k_abs = tile_base + k_local;
            const uint block_idx = k_abs >> 5;           // 32-elem block id
            const uint inner     = k_abs & 31u;           // offset within block
            const uint chunk = min((uint)(weights_per_block - inner), k_remain - k_local);
            if (out_idx < NQ) {
                const uint base_idx_q = block_idx * NQ + out_idx;
                const uint sb_q = base_idx_q * 2u;
                const ushort bits_q = (ushort)scales_q[sb_q] | ((ushort)scales_q[sb_q + 1] << 8);
                const float scale_q = static_cast<float>(as_type<half>(bits_q));
                const uint base_byte_q = base_idx_q * weights_per_block + inner;
                const device char *qs_q = (const device char *)(data_q + base_byte_q);
                float block_sum_q = q8_canonical_block_dot(qs_q, inner, chunk, k_local, x_tile);
                sum_q = fma(scale_q, block_sum_q, sum_q);
            }
            if (out_idx < NK) {
                const uint base_idx_k = block_idx * NK + out_idx;
                const uint sb_k = base_idx_k * 2u;
                const ushort bits_k = (ushort)scales_k[sb_k] | ((ushort)scales_k[sb_k + 1] << 8);
                const float scale_k = static_cast<float>(as_type<half>(bits_k));
                const uint base_byte_k = base_idx_k * weights_per_block + inner;
                const device char *qs_k = (const device char *)(data_k + base_byte_k);
                float block_sum_k = q8_canonical_block_dot(qs_k, inner, chunk, k_local, x_tile);
                sum_k = fma(scale_k, block_sum_k, sum_k);
            }
            if (out_idx < NV) {
                const uint base_idx_v = block_idx * NV + out_idx;
                const uint sb_v = base_idx_v * 2u;
                const ushort bits_v = (ushort)scales_v[sb_v] | ((ushort)scales_v[sb_v + 1] << 8);
                const float scale_v = static_cast<float>(as_type<half>(bits_v));
                const uint base_byte_v = base_idx_v * weights_per_block + inner;
                const device char *qs_v = (const device char *)(data_v + base_byte_v);
                float block_sum_v = q8_canonical_block_dot(qs_v, inner, chunk, k_local, x_tile);
                sum_v = fma(scale_v, block_sum_v, sum_v);
            }

            k_local += chunk;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (out_idx < NQ) {
        float val_q = sum_q;
        if (params->has_bias_q != 0u) {
            val_q += static_cast<float>(bias_q[out_idx]);
        }
        out_q[out_idx] = static_cast<half>(val_q);
    }
    if (out_idx < NK) {
        float val_k = sum_k;
        if (params->has_bias_k != 0u) {
            val_k += static_cast<float>(bias_k[out_idx]);
        }
        out_k[out_idx] = static_cast<half>(val_k);
    }
    if (out_idx < NV) {
        float val_v = sum_v;
        if (params->has_bias_v != 0u) {
            val_v += static_cast<float>(bias_v[out_idx]);
        }
        out_v[out_idx] = static_cast<half>(val_v);
    }
}

// Fused 2-output GEMV for canonical Q8 weights (e.g., gate and up projections).
// Computes two outputs (G0, G1) in a single pass of K, sharing the staged x_tile.
struct Q2FusedParams {
    uint K;
    uint N0;
    uint N1;
    uint blocks_per_k;
    uint weights_per_block;
    uint has_bias0;
    uint has_bias1;
};

[[kernel]] void gemv_q8_fused2_f16(
    const device uchar *data0 [[buffer(0)]],
    const device uchar *data1 [[buffer(1)]],
    const device half *vector_x [[buffer(2)]],
    device half *out0 [[buffer(3)]],
    device half *out1 [[buffer(4)]],
    const constant Q2FusedParams *params [[buffer(5)]],
    const device uchar *scales0 [[buffer(6)]],
    const device uchar *scales1 [[buffer(7)]],
    const device half *bias0 [[buffer(8)]],
    const device half *bias1 [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float x_tile[TILE_K];

    const uint K = params->K;
    const uint N0 = params->N0;
    const uint N1 = params->N1;
    const uint blocks_per_k = params->blocks_per_k;
    const uint weights_per_block = params->weights_per_block;
    const bool use_bias0 = (params->has_bias0 != 0u) && (bias0 != (const device half*)nullptr);
    const bool use_bias1 = (params->has_bias1 != 0u) && (bias1 != (const device half*)nullptr);

    // Lane processes one output col across tiles; we cover N0 and N1 in separate loops to keep indexing simple.
    // First output group (0)
    {
        const uint out_idx = gid.x * TILE_N + lid.x;
        const bool active = out_idx < N0;
        float sum = 0.0f;
        if (weights_per_block != 0u) {
            for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
                const uint tile_limit = min(TILE_K, K - tile_base);
                if (lid.x < LOAD_LANES) {
                    for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                        const uint gk = tile_base + local;
                        x_tile[local] = static_cast<float>(vector_x[gk]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active) {
                    uint k_local = 0;
                    while (k_local < tile_limit) {
                        const uint k_abs = tile_base + k_local;
                        const uint block_idx = k_abs >> 5; // 32 elements per block
                        const uint inner = k_abs & 31u;
                        const uint base_idx = block_idx * N0 + out_idx;
                        const uint sb = base_idx * 2u;
                        const ushort bits = (ushort)scales0[sb] | ((ushort)scales0[sb + 1] << 8);
                        const float scale = static_cast<float>(as_type<half>(bits));
                        const uint base_byte = base_idx * weights_per_block + inner;
                        const device char *qs = (const device char *)(data0 + base_byte);
                        const uint count = min((uint)(weights_per_block - inner), tile_limit - k_local);
                        float block_sum = q8_canonical_block_dot(qs, inner, count, k_local, x_tile);
                        sum += scale * block_sum;
                        k_local += count;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (active) {
                float b = (use_bias0 ? static_cast<float>(bias0[out_idx]) : 0.0f);
                out0[out_idx] = static_cast<half>(sum + b);
            }
        }
    }

    // Second output group (1)
    {
        const uint out_idx1 = gid.x * TILE_N + lid.x;
        const bool active1 = out_idx1 < N1;
        float sum1 = 0.0f;
        if (weights_per_block != 0u) {
            for (uint tile_base = 0; tile_base < K; tile_base += TILE_K) {
                const uint tile_limit = min(TILE_K, K - tile_base);
                if (lid.x < LOAD_LANES) {
                    for (uint local = lid.x; local < tile_limit; local += LOAD_LANES) {
                        const uint gk = tile_base + local;
                        x_tile[local] = static_cast<float>(vector_x[gk]);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (active1) {
                    uint k_local = 0;
                    while (k_local < tile_limit) {
                        const uint k_abs = tile_base + k_local;
                        const uint block_idx = k_abs >> 5;
                        const uint inner = k_abs & 31u;
                        const uint base_idx = block_idx * N1 + out_idx1;
                        const uint sb = base_idx * 2u;
                        const ushort bits = (ushort)scales1[sb] | ((ushort)scales1[sb + 1] << 8);
                        const float scale = static_cast<float>(as_type<half>(bits));
                        const uint base_byte = base_idx * weights_per_block + inner;
                        const device char *qs = (const device char *)(data1 + base_byte);
                        const uint count = min((uint)(weights_per_block - inner), tile_limit - k_local);
                        float block_sum = q8_canonical_block_dot(qs, inner, count, k_local, x_tile);
                        sum1 += scale * block_sum;
                        k_local += count;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (active1) {
                float b1 = (use_bias1 ? static_cast<float>(bias1[out_idx1]) : 0.0f);
                out1[out_idx1] = static_cast<half>(sum1 + b1);
            }
        }
    }
}

// GEMM kernel for canonical Q8 weights with transpose_b = true (NT layout).
// Supports up to 4 output rows per threadgroup tile for decode workloads.
[[kernel]] void gemm_q8_nt_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint blocks_per_k = params->blocks_per_k;
    const uint weights_per_block = params->weights_per_block;

    // Tile configuration tuned for decode (m small, n large)
    constexpr uint TILE_COLS = 128;
    constexpr uint ROWS_PER_TILE = 4;
    constexpr uint THREADS_PER_TG = TILE_COLS;

    threadgroup float a_tile[ROWS_PER_TILE * Q8_0_WEIGHTS_PER_BLOCK];

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) {
        return;
    }
    const uint rows_this_tile = min((uint)ROWS_PER_TILE, m - row_tile);

    const uint col = gid.x * TILE_COLS + tid.x;
    if (col >= n) {
        return;
    }

    float accum[ROWS_PER_TILE];
    for (uint r = 0; r < rows_this_tile; ++r) {
        accum[r] = 0.0f;
    }

    for (uint block = 0; block < blocks_per_k; ++block) {
        // Stage the A tile for this K block
        for (uint idx = tid.x; idx < rows_this_tile * weights_per_block; idx += THREADS_PER_TG) {
            const uint local_row = idx / weights_per_block;
            const uint offset = idx % weights_per_block;
            const uint global_k = block * weights_per_block + offset;
            float val = 0.0f;
            if (global_k < k) {
                const uint global_row = row_tile + local_row;
                val = static_cast<float>(matrix_a[global_row * lda + global_k]);
            }
            a_tile[local_row * weights_per_block + offset] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load scale and quant data for this column/block
        const uint scale_idx = (block * n + col) * Q8_CANONICAL_SCALE_BYTES;
        const device uchar *sb = scale_bytes + scale_idx;
        const ushort bits = (ushort)sb[0] | ((ushort)sb[1] << 8);
        const float scale = static_cast<float>(as_type<half>(bits));
        const device char *qs = (const device char *)(matrix_data + (block * n + col) * weights_per_block);

        for (uint local_row = 0; local_row < rows_this_tile; ++local_row) {
            float block_sum = 0.0f;
            const threadgroup float *a_base = a_tile + local_row * weights_per_block;
            for (uint i = 0; i < weights_per_block; ++i) {
                const uint global_k = block * weights_per_block + i;
                if (global_k >= k) {
                    break;
                }
                block_sum = fma(a_base[i], (float)qs[i], block_sum);
            }
            accum[local_row] = fma(scale, block_sum, accum[local_row]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint local_row = 0; local_row < rows_this_tile; ++local_row) {
        const uint out_row = row_tile + local_row;
        float value = accum[local_row];
        if (params->has_bias != 0u) {
            value += static_cast<float>(bias[col]);
        }
        result_y[out_row * ldc + col] = static_cast<half>(value);
    }
}

// Canonical large-N GEMM kernel for Q8 weights. Extends the NT variant by
// supporting additional output rows per tile (8 vs 4) and aggressively
// vectorizing the inner products for high-N decode blocks.
[[kernel]] void gemm_q8_canonical_large_n_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    constexpr uint ROWS_PER_TILE = 32u;
    constexpr uint TILE_COLS_TOTAL = 128u;
    constexpr uint TILE_COLS_PER_TG = TILE_COLS_TOTAL / 2u;
    constexpr uint COLS_PER_THREAD = 2u;
    constexpr uint TG_ROWS = 4u;
    constexpr uint TG_COL_LANES = TILE_COLS_PER_TG / COLS_PER_THREAD;
    constexpr uint THREADS_PER_TG = TG_COL_LANES * TG_ROWS;

    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint blocks_per_k = params->blocks_per_k;
    const uint weights_per_block = params->weights_per_block;

    threadgroup float a_tile[2][ROWS_PER_TILE * Q8_0_WEIGHTS_PER_BLOCK];

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) {
        return;
    }
    const uint rows_this_tile = min(ROWS_PER_TILE, m - row_tile);

    const uint lane_x = tid3.x;
    const uint lane_y = tid3.y;
    const uint col_block = gid.x * TILE_COLS_PER_TG + lane_x * COLS_PER_THREAD;
    if (col_block >= n) {
        return;
    }

    float accum[ROWS_PER_TILE][COLS_PER_THREAD];
    for (uint r = 0; r < rows_this_tile; ++r) {
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            accum[r][c] = 0.0f;
        }
    }

    const uint scale_block_stride = n * Q8_CANONICAL_SCALE_BYTES;
    const uint data_block_stride = n * weights_per_block;
    const device uchar *scale_lane_ptrs[COLS_PER_THREAD];
    const device char *data_lane_ptrs[COLS_PER_THREAD];
    bool column_active[COLS_PER_THREAD];
    for (uint c = 0; c < COLS_PER_THREAD; ++c) {
        const uint col = col_block + c;
        if (col < n) {
            scale_lane_ptrs[c] = scale_bytes + col * Q8_CANONICAL_SCALE_BYTES;
            data_lane_ptrs[c] = (const device char *)(matrix_data + col * weights_per_block);
            column_active[c] = true;
        } else {
            scale_lane_ptrs[c] = scale_bytes;
            data_lane_ptrs[c] = (const device char *)(matrix_data);
            column_active[c] = false;
        }
    }

    const uint total_blocks = blocks_per_k;
    const uint linear_tid = lane_y * TG_COL_LANES + lane_x;
    for (uint block = 0; block < total_blocks; ++block) {
        const uint block_k_start = block * weights_per_block;
        if (block_k_start >= k) {
            break;
        }

        const uint buffer_idx = block & 1u;
        // Stage the A tile for this K block
        for (uint idx = linear_tid; idx < rows_this_tile * weights_per_block; idx += THREADS_PER_TG) {
            const uint local_row = idx / weights_per_block;
            const uint offset = idx % weights_per_block;
            const uint global_k = block_k_start + offset;
            float val = 0.0f;
            if (global_k < k) {
                const uint global_row = row_tile + local_row;
                val = static_cast<float>(matrix_a[global_row * lda + global_k]);
            }
            a_tile[buffer_idx][local_row * weights_per_block + offset] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint valid = min(weights_per_block, k - block_k_start);
        if (valid == 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        // Prepare scale and QS pointers for current block
        float scale_vec[COLS_PER_THREAD];
        const device char *qs_ptrs[COLS_PER_THREAD];
        bool active_cols[COLS_PER_THREAD];
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            if (column_active[c]) {
                const ushort bits = (ushort)scale_lane_ptrs[c][0] | ((ushort)scale_lane_ptrs[c][1] << 8);
                scale_vec[c] = static_cast<float>(as_type<half>(bits));
                qs_ptrs[c] = data_lane_ptrs[c];
                active_cols[c] = true;
            } else {
                scale_vec[c] = 0.0f;
                qs_ptrs[c] = (const device char *)(matrix_data);
                active_cols[c] = false;
            }
        }

        const uint consume_buffer = buffer_idx;
        for (uint local_row = lane_y; local_row < rows_this_tile; local_row += TG_ROWS) {
            threadgroup float *a_base = &a_tile[consume_buffer][local_row * weights_per_block];
            uint processed = 0u;

            while (processed + 8u <= valid) {
                const threadgroup float *x_ptr = a_base + processed;
                const float4 x0 = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
                const float4 x1 = float4(x_ptr[4], x_ptr[5], x_ptr[6], x_ptr[7]);

                // Unroll over COLS_PER_THREAD=2 to reduce loop overhead
                if (active_cols[0]) {
                    const device char4 *qv0 = (const device char4 *)(qs_ptrs[0] + processed + 0u);
                    const device char4 *qv1 = (const device char4 *)(qs_ptrs[0] + processed + 4u);
                    const char4 q0 = *qv0;
                    const char4 q1 = *qv1;
                    const float block_sum0 = dot(x0, float4(q0)) + dot(x1, float4(q1));
                    accum[local_row][0] = fma(scale_vec[0], block_sum0, accum[local_row][0]);
                }
                if (COLS_PER_THREAD > 1 && active_cols[1]) {
                    const device char4 *qv0 = (const device char4 *)(qs_ptrs[1] + processed + 0u);
                    const device char4 *qv1 = (const device char4 *)(qs_ptrs[1] + processed + 4u);
                    const char4 q0 = *qv0;
                    const char4 q1 = *qv1;
                    const float block_sum1 = dot(x0, float4(q0)) + dot(x1, float4(q1));
                    accum[local_row][1] = fma(scale_vec[1], block_sum1, accum[local_row][1]);
                }

                processed += 8u;
            }

            while (processed + 4u <= valid) {
                const threadgroup float *x_ptr = a_base + processed;
                const float4 x = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);

                if (active_cols[0]) {
                    const device char4 *qv = (const device char4 *)(qs_ptrs[0] + processed);
                    const char4 q = *qv;
                    const float block_sum0 = dot(x, float4(q));
                    accum[local_row][0] = fma(scale_vec[0], block_sum0, accum[local_row][0]);
                }
                if (COLS_PER_THREAD > 1 && active_cols[1]) {
                    const device char4 *qv = (const device char4 *)(qs_ptrs[1] + processed);
                    const char4 q = *qv;
                    const float block_sum1 = dot(x, float4(q));
                    accum[local_row][1] = fma(scale_vec[1], block_sum1, accum[local_row][1]);
                }

                processed += 4u;
            }

            while (processed < valid) {
                const float x_val = a_base[processed];
                for (uint c = 0; c < COLS_PER_THREAD; ++c) {
                    if (!active_cols[c]) {
                        continue;
                    }
                    const float q_val = (float)qs_ptrs[c][processed];
                    accum[local_row][c] = fma(scale_vec[c], x_val * q_val, accum[local_row][c]);
                }
                processed += 1u;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            if (column_active[c]) {
                scale_lane_ptrs[c] += scale_block_stride;
                data_lane_ptrs[c] += data_block_stride;
            }
        }
    }

    for (uint local_row = lane_y; local_row < rows_this_tile; local_row += TG_ROWS) {
        const uint out_row = row_tile + local_row;
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            const uint col = col_block + c;
            if (col >= n) {
                continue;
            }
            float value = accum[local_row][c];
            if (params->has_bias != 0u) {
                value += static_cast<float>(bias[col]);
            }
            result_y[out_row * ldc + col] = static_cast<half>(value);
        }
    }
}

// Same as gemm_q8_canonical_large_n_f16 but with half the row tile (16) for small-m cases.
[[kernel]] void gemm_q8_canonical_large_n_rows16_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]]) {
    constexpr uint ROWS_PER_TILE = 16u;
    constexpr uint TILE_COLS_TOTAL = 128u;
    constexpr uint TILE_COLS_PER_TG = TILE_COLS_TOTAL / 2u;
    constexpr uint COLS_PER_THREAD = 2u;
    constexpr uint TG_ROWS = 4u;
    constexpr uint TG_COL_LANES = TILE_COLS_PER_TG / COLS_PER_THREAD;
    constexpr uint THREADS_PER_TG = TG_COL_LANES * TG_ROWS;

    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint blocks_per_k = params->blocks_per_k;
    const uint weights_per_block = params->weights_per_block;

    threadgroup float a_tile[2][ROWS_PER_TILE * Q8_0_WEIGHTS_PER_BLOCK];

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) {
        return;
    }
    const uint rows_this_tile = min(ROWS_PER_TILE, m - row_tile);

    const uint lane_x = tid3.x;
    const uint lane_y = tid3.y;
    const uint col_block = gid.x * TILE_COLS_PER_TG + lane_x * COLS_PER_THREAD;
    if (col_block >= n) {
        return;
    }

    float accum[ROWS_PER_TILE][COLS_PER_THREAD];
    for (uint r = 0; r < rows_this_tile; ++r) {
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            accum[r][c] = 0.0f;
        }
    }

    const uint scale_block_stride = n * Q8_CANONICAL_SCALE_BYTES;
    const uint data_block_stride = n * weights_per_block;
    const device uchar *scale_lane_ptrs[COLS_PER_THREAD];
    const device char *data_lane_ptrs[COLS_PER_THREAD];
    bool column_active[COLS_PER_THREAD];
    for (uint c = 0; c < COLS_PER_THREAD; ++c) {
        const uint col = col_block + c;
        if (col < n) {
            scale_lane_ptrs[c] = scale_bytes + col * Q8_CANONICAL_SCALE_BYTES;
            data_lane_ptrs[c] = (const device char *)(matrix_data + col * weights_per_block);
            column_active[c] = true;
        } else {
            scale_lane_ptrs[c] = scale_bytes;
            data_lane_ptrs[c] = (const device char *)(matrix_data);
            column_active[c] = false;
        }
    }

    const uint total_blocks = blocks_per_k;
    const uint linear_tid = lane_y * TG_COL_LANES + lane_x;
    for (uint block = 0; block < total_blocks; ++block) {
        const uint block_k_start = block * weights_per_block;
        if (block_k_start >= k) {
            break;
        }

        const uint buffer_idx = block & 1u;
        for (uint idx = linear_tid; idx < rows_this_tile * weights_per_block; idx += THREADS_PER_TG) {
            const uint local_row = idx / weights_per_block;
            const uint offset = idx % weights_per_block;
            const uint global_k = block_k_start + offset;
            float val = 0.0f;
            if (global_k < k) {
                const uint global_row = row_tile + local_row;
                val = static_cast<float>(matrix_a[global_row * lda + global_k]);
            }
            a_tile[buffer_idx][local_row * weights_per_block + offset] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint valid = min(weights_per_block, k - block_k_start);
        if (valid == 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            continue;
        }

        float scale_vec[COLS_PER_THREAD];
        const device char *qs_ptrs[COLS_PER_THREAD];
        bool active_cols[COLS_PER_THREAD];
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            if (column_active[c]) {
                const ushort bits = (ushort)scale_lane_ptrs[c][0] | ((ushort)scale_lane_ptrs[c][1] << 8);
                scale_vec[c] = static_cast<float>(as_type<half>(bits));
                qs_ptrs[c] = data_lane_ptrs[c];
                active_cols[c] = true;
            } else {
                scale_vec[c] = 0.0f;
                qs_ptrs[c] = (const device char *)(matrix_data);
                active_cols[c] = false;
            }
        }

        const uint consume_buffer = buffer_idx;
        for (uint local_row = lane_y; local_row < rows_this_tile; local_row += TG_ROWS) {
            threadgroup float *a_base = &a_tile[consume_buffer][local_row * weights_per_block];
            uint processed = 0u;

            while (processed + 8u <= valid) {
                const threadgroup float *x_ptr = a_base + processed;
                const float4 x0 = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);
                const float4 x1 = float4(x_ptr[4], x_ptr[5], x_ptr[6], x_ptr[7]);

                if (active_cols[0]) {
                    const device char4 *qv0 = (const device char4 *)(qs_ptrs[0] + processed + 0u);
                    const device char4 *qv1 = (const device char4 *)(qs_ptrs[0] + processed + 4u);
                    const char4 q0 = *qv0;
                    const char4 q1 = *qv1;
                    const float block_sum0 = dot(x0, float4(q0)) + dot(x1, float4(q1));
                    accum[local_row][0] = fma(scale_vec[0], block_sum0, accum[local_row][0]);
                }
                if (COLS_PER_THREAD > 1 && active_cols[1]) {
                    const device char4 *qv0 = (const device char4 *)(qs_ptrs[1] + processed + 0u);
                    const device char4 *qv1 = (const device char4 *)(qs_ptrs[1] + processed + 4u);
                    const char4 q0 = *qv0;
                    const char4 q1 = *qv1;
                    const float block_sum1 = dot(x0, float4(q0)) + dot(x1, float4(q1));
                    accum[local_row][1] = fma(scale_vec[1], block_sum1, accum[local_row][1]);
                }

                processed += 8u;
            }

            while (processed + 4u <= valid) {
                const threadgroup float *x_ptr = a_base + processed;
                const float4 x = float4(x_ptr[0], x_ptr[1], x_ptr[2], x_ptr[3]);

                if (active_cols[0]) {
                    const device char4 *qv = (const device char4 *)(qs_ptrs[0] + processed);
                    const char4 q = *qv;
                    const float block_sum0 = dot(x, float4(q));
                    accum[local_row][0] = fma(scale_vec[0], block_sum0, accum[local_row][0]);
                }
                if (COLS_PER_THREAD > 1 && active_cols[1]) {
                    const device char4 *qv = (const device char4 *)(qs_ptrs[1] + processed);
                    const char4 q = *qv;
                    const float block_sum1 = dot(x, float4(q));
                    accum[local_row][1] = fma(scale_vec[1], block_sum1, accum[local_row][1]);
                }

                processed += 4u;
            }

            while (processed < valid) {
                const float x_val = a_base[processed];
                for (uint c = 0; c < COLS_PER_THREAD; ++c) {
                    if (!active_cols[c]) {
                        continue;
                    }
                    const float q_val = (float)qs_ptrs[c][processed];
                    accum[local_row][c] = fma(scale_vec[c], x_val * q_val, accum[local_row][c]);
                }
                processed += 1u;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            if (column_active[c]) {
                scale_lane_ptrs[c] += scale_block_stride;
                data_lane_ptrs[c] += data_block_stride;
            }
        }
    }

    for (uint local_row = lane_y; local_row < rows_this_tile; local_row += TG_ROWS) {
        const uint out_row = row_tile + local_row;
        for (uint c = 0; c < COLS_PER_THREAD; ++c) {
            const uint col = col_block + c;
            if (col >= n) {
                continue;
            }
            float value = accum[local_row][c];
            if (params->has_bias != 0u) {
                value += static_cast<float>(bias[col]);
            }
            result_y[out_row * ldc + col] = static_cast<half>(value);
        }
    }
}

// Multi-row GEMV for canonical Q8 weights (computes up to 4 rows of Y per tile).
// Uses the same canonical addressing as GEMV: block-major across K, then columns N.
[[kernel]] void gemv_q8_rows_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    const device half *matrix_a [[buffer(2)]],
    device half *result_y [[buffer(3)]],
    const constant GemmQ8NtParams *params [[buffer(4)]],
    const device half *bias [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
    const uint m = params->m;
    const uint n = params->n;
    const uint k = params->k;
    const uint lda = params->lda;
    const uint ldc = params->ldc;
    const uint weights_per_block = params->weights_per_block;
    const uint has_bias = params->has_bias;

    constexpr uint ROWS_PER_TILE = 4;
    constexpr uint ROWS_TILE_K = TILE_K / ROWS_PER_TILE; // resized TILE_K to fit in threadgroup memory limit
    threadgroup float x_rows[ROWS_PER_TILE * ROWS_TILE_K];

    const uint row_tile = gid.y * ROWS_PER_TILE;
    if (row_tile >= m) return;
    const uint rows_this_tile = min((uint)ROWS_PER_TILE, m - row_tile);

    const uint col = gid.x * TILE_N + tid.x;
    if (col >= n) return;

    float4 accum = float4(0.0f);

    for (uint tile_base = 0; tile_base < k; tile_base += ROWS_TILE_K) {
        const uint tile_limit = min(ROWS_TILE_K, k - tile_base);

        if (tid.x < LOAD_LANES) {
            for (uint row = 0; row < rows_this_tile; ++row) {
                for (uint local = tid.x; local < tile_limit; local += LOAD_LANES) {
                    const uint gk = tile_base + local;
                    float val = 0.0f;
                    if (gk < k) {
                        const uint g_row = row_tile + row;
                        val = static_cast<float>(matrix_a[g_row * lda + gk]);
                    }
                    x_rows[row * ROWS_TILE_K + local] = val;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint k_local = 0;
        while (k_local < tile_limit) {
            const uint k_abs = tile_base + k_local;
            const uint block_idx = k_abs >> 5;
            const uint inner = k_abs & 31u;
            const uint base_idx = block_idx * n + col;
            const uint sb = base_idx * 2u;
            const ushort bits = (ushort)scale_bytes[sb] | ((ushort)scale_bytes[sb + 1] << 8);
            const float scale = static_cast<float>(as_type<half>(bits));
            const uint count = min((uint)(weights_per_block - inner), tile_limit - k_local);
            const uint base_byte = base_idx * weights_per_block + inner;
            const device char *qs = (const device char *)(matrix_data + base_byte);

            const uint vec_chunks = count / 4u;
            const uint tail = count & 3u;
            float4 q_vecs[Q8_0_WEIGHTS_PER_BLOCK / 4];
            for (uint vc = 0; vc < vec_chunks; ++vc) {
                const device char4 *qv = (const device char4 *)(qs + vc * 4u);
                char4 q = *qv;
                q_vecs[vc] = float4((float)q.x, (float)q.y, (float)q.z, (float)q.w);
            }
            float q_tail[3];
            for (uint t = 0; t < tail; ++t) {
                q_tail[t] = (float)qs[vec_chunks * 4u + t];
            }

            for (uint row = 0; row < rows_this_tile; ++row) {
                const threadgroup float *x_base = x_rows + row * ROWS_TILE_K + k_local;
                float block_sum = 0.0f;
                for (uint vc = 0; vc < vec_chunks; ++vc) {
                    const uint idx = vc * 4u;
                    float4 x_vec = float4(
                        x_base[idx + 0u],
                        x_base[idx + 1u],
                        x_base[idx + 2u],
                        x_base[idx + 3u]);
                    block_sum += dot(x_vec, q_vecs[vc]);
                }
                for (uint t = 0; t < tail; ++t) {
                    block_sum = fma(x_base[vec_chunks * 4u + t], q_tail[t], block_sum);
                }
                accum[row] = fma(scale, block_sum, accum[row]);
            }

            k_local += count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (has_bias != 0u) {
        accum += float4(
            static_cast<float>(bias[col]),
            static_cast<float>(bias[col]),
            static_cast<float>(bias[col]),
            static_cast<float>(bias[col]));
    }
    if (rows_this_tile > 0) result_y[(row_tile + 0u) * ldc + col] = static_cast<half>(accum[0]);
    if (rows_this_tile > 1) result_y[(row_tile + 1u) * ldc + col] = static_cast<half>(accum[1]);
    if (rows_this_tile > 2) result_y[(row_tile + 2u) * ldc + col] = static_cast<half>(accum[2]);
    if (rows_this_tile > 3) result_y[(row_tile + 3u) * ldc + col] = static_cast<half>(accum[3]);
}
