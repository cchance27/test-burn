#include <metal_simdgroup_matrix>

inline float flash_q2_scalar(const threadgroup FlashDecodeVec2T* q2_shared, uint idx) {
    FlashDecodeVec2T qv = q2_shared[idx >> 1];
    return (float)qv[idx & 1u];
}

inline float flash_q4_scalar(const threadgroup FlashDecodeVec4T* q4_shared, uint idx) {
    FlashDecodeVec4T qv = q4_shared[idx >> 2];
    return (float)qv[idx & 3u];
}

inline float flash_decode_tile_score_from_pair(float s0, float s1, uint t) {
    uint src_lane = (t < 2u) ? 0u : ((t < 4u) ? 1u : ((t < 6u) ? 8u : 9u));
    return ((t & 1u) == 0u) ? simd_shuffle(s0, src_lane) : simd_shuffle(s1, src_lane);
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
inline void flash_decode_warp_tiled_m1_half2(
    const threadgroup FlashDecodeVec2T* q2_shared,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device OutputStorageT* output,
    uint warp,
    uint lane,
    constant SdpaParams& params,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type* shared_warp_out
) {
    uint kv_len = params.kv_len;
    uint head_dim = params.head_dim;
    uint stride_k_s = params.stride_k_s;
    uint stride_v_s = params.stride_v_s;

    // Defensive guards. We currently optimize for the primary Qwen2.5 shape (D=64).
    if (head_dim != 64 || stride_k_s != 64 || stride_v_s != 64) {
        return;
    }
    if (kv_len == 0 || kv_len > 65536) {
        return;
    }

    // Running stats (maintained in warp 0 lane 0; alpha/beta broadcast each block).
    float m = -1e30f;
    float l = 0.0f;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Output accumulator (only meaningful for warp 0).
    float2 out_acc = float2(0.0f);

    float scale = params.scale;

    constexpr uint KEYS = WARPS * KEYS_PER_WARP;

    // Process KV in blocks of KEYS keys.
    // Larger blocks reduce the number of (m,l) renormalizations and tend to match
    // the materialized softmax path more closely for inference.
    for (uint base = 0; base < kv_len; base += KEYS) {
        // === Scores for this warp's keys ===
        float score_j[KEYS_PER_WARP];
        
        // OPTIMIZATION: Removed kv_idx_j array to save registers. 
        // We recompute indices in the second loop.
        
        float my_warp_max = -1e30f;

        #pragma unroll
        for (uint j = 0; j < KEYS_PER_WARP; ++j) {
            uint kv_idx = base + warp + j * WARPS;
            float partial = 0.0f;
            float score = -1e30f;
            if (kv_idx < kv_len) {
                FlashDecodeVec2T qh = q2_shared[lane];
                FlashDecodeVec2T kh = flash_load_as_half2(k_base, (ulong)kv_idx * (ulong)stride_k_s + ((ulong)lane << 1)); // INDEX64_OK
                float2 qf = half2_to_float2(qh);
                float2 kf = half2_to_float2(kh);
                partial = qf[0] * kf[0] + qf[1] * kf[1];
                score = simd_sum(partial) * scale;
            }
            if (!isfinite(score)) {
                score = -1e30f;
            }
            score_j[j] = score;
            my_warp_max = max(my_warp_max, score);
        }
        
        // 1. Store per-warp max to shared memory
        if (lane == 0) {
            shared_warp_max[warp] = my_warp_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Compute the block max in *each warp* from shared_warp_max.
        // This avoids a warp0-only reduction + an extra threadgroup barrier.
        float val = (lane < WARPS) ? shared_warp_max[lane] : -1e30f;
        float block_max = simd_max(val);

        // === Accumulate per-warp partial output + sumexp ===
        float2 warp_out = float2(0.0f);
        float warp_sum = 0.0f;
        #pragma unroll
        for (uint j = 0; j < KEYS_PER_WARP; ++j) {
            uint kv_idx = base + warp + j * WARPS;
            if (kv_idx >= kv_len) {
                continue;
            }
            float score = score_j[j];
            float w_local = 0.0f;
            if (lane == 0) {
                w_local = metal::fast::exp(score - block_max);
            }
            float w = simd_broadcast(w_local, 0);
            FlashDecodeVec2T vh = flash_load_as_half2(v_base, (ulong)kv_idx * (ulong)stride_v_s + ((ulong)lane << 1)); // INDEX64_OK
            warp_out += half2_to_float2(vh) * w;
            if (lane == 0) {
                warp_sum += w;
            }
        }
        shared_warp_out[warp * 32 + lane] = flash_pack_out2<TG_OUT_HALF>(warp_out);
        if (lane == 0) {
            shared_warp_sums[warp] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Warp 0 reduces contributions + updates running state ===
        if (warp == 0) {
            float2 block_out = float2(0.0f);
            for (uint widx = 0; widx < WARPS; ++widx) {
                block_out += flash_unpack_out2<TG_OUT_HALF>(shared_warp_out[widx * 32 + lane]);
            }

            // Parallel reduction of block sum using simd_sum
            float block_sum = 0.0f;
            
            // Lane 0..7 load the 8 warp sums.
            if (lane < WARPS) {
                block_sum = shared_warp_sums[lane];
            }
            // All lanes in warp 0 participate in the reduction.
            block_sum = simd_sum(block_sum);

            float alpha_val = 0.0f;
            float beta_val = 0.0f;

            if (lane == 0) {
                // Update running (m, l) once per block.
                float m_new = max(m, block_max);
                alpha = metal::fast::exp(m - m_new);
                beta = metal::fast::exp(block_max - m_new);
                l = l * alpha + block_sum * beta;
                m = m_new;

                // Store l back to shared memory for final normalization (and reuse slot).
                shared_warp_sums[0] = l;
                
                // Expose alpha/beta for broadcast (via local var).
                alpha_val = alpha;
                beta_val = beta;
            }

            // Broadcast alpha/beta from lane 0 to all lanes in warp 0.
            // This avoids the need for a barrier inside this divergent branch.
            float alpha_b = simd_broadcast(alpha_val, 0);
            float beta_b = simd_broadcast(beta_val, 0);
            
            out_acc = out_acc * alpha_b + block_out * beta_b;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Final normalize and store (warp 0 only) ===
    if (warp == 0) {
        float denom = shared_warp_sums[0];
        float inv = (isfinite(denom) && abs(denom) > 1e-6f) ? (1.0f / denom) : 1.0f;
        float2 out = out_acc * inv;
        // Defensive: avoid propagating NaNs/Infs into downstream layers.
        out[0] = isfinite(out[0]) ? out[0] : 0.0f;
        out[1] = isfinite(out[1]) ? out[1] : 0.0f;
        flash_store_from_float2(output, ((ulong)lane << 1), out); // INDEX64_OK
    }
}

// -----------------------------------------------------------------------------
// FlashDecodeVec4T path (supports head_dim==64 and head_dim==128).
// Intended for fused RoPE -> SDPA compound kernels where Q is naturally stored as FlashDecodeVec4T.
// -----------------------------------------------------------------------------
template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
inline void flash_decode_warp_tiled_m1_half4(
    const threadgroup FlashDecodeVec4T* q4_shared,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device OutputStorageT* output,
    uint warp,
    uint lane,
    constant SdpaParams& params,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type* shared_warp_out
) {
    uint kv_len = params.kv_len;
    uint head_dim = params.head_dim;
    uint stride_k_s = params.stride_k_s;
    uint stride_v_s = params.stride_v_s;

    // Defensive guards. Fused path currently supports D=64 and D=128.
    if (!((head_dim == 64) || (head_dim == 128))) {
        return;
    }
    if (stride_k_s != head_dim || stride_v_s != head_dim) {
        return;
    }
    if (kv_len == 0 || kv_len > 65536) {
        return;
    }

    const uint vec4 = head_dim >> 2; // number of FlashDecodeVec4T per head

    // Running stats (maintained in warp 0 lane 0; alpha/beta broadcast each block).
    float m = -1e30f;
    float l = 0.0f;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Output accumulator (only meaningful for warp 0).
    float4 out_acc = float4(0.0f);
    float scale = params.scale;

    constexpr uint KEYS = WARPS * KEYS_PER_WARP;

    for (uint base = 0; base < kv_len; base += KEYS) {
        float score_j[KEYS_PER_WARP];
        
        // OPTIMIZATION: Removed kv_idx_j array to save registers.
        // We recompute indices in the second loop.
        
        float my_warp_max = -1e30f;

        #pragma unroll
        for (uint j = 0; j < KEYS_PER_WARP; ++j) {
            uint kv_idx = base + warp + j * WARPS;
            
            float partial = 0.0f;
            float score = -1e30f;
            if (kv_idx < kv_len && lane < vec4) {
                FlashDecodeVec4T qh = q4_shared[lane];
                FlashDecodeVec4T kh = flash_load_as_half4(k_base, (ulong)kv_idx * (ulong)stride_k_s + ((ulong)lane << 2)); // INDEX64_OK
                float4 qf = half4_to_float4(qh);
                float4 kf = half4_to_float4(kh);
                partial = qf[0] * kf[0] + qf[1] * kf[1] + qf[2] * kf[2] + qf[3] * kf[3];
            }
            if (kv_idx < kv_len) {
                // `simd_sum` must be executed uniformly by the warp.
                score = simd_sum(partial) * scale;
            }
            if (!isfinite(score)) {
                score = -1e30f;
            }
            score_j[j] = score;
            my_warp_max = max(my_warp_max, score);
        }
        
        if (lane == 0) {
            shared_warp_max[warp] = my_warp_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute the block max in each warp (see FlashDecodeVec2T path above).
        float val = (lane < WARPS) ? shared_warp_max[lane] : -1e30f;
        float block_max = simd_max(val);

        float4 warp_out = float4(0.0f);
        float warp_sum = 0.0f;
        #pragma unroll
        for (uint j = 0; j < KEYS_PER_WARP; ++j) {
            uint kv_idx = base + warp + j * WARPS;
            if (kv_idx >= kv_len) {
                continue;
            }
            float score = score_j[j];
            float w_local = 0.0f;
            if (lane == 0) {
                w_local = metal::fast::exp(score - block_max);
            }
            float w = simd_broadcast(w_local, 0);
            if (lane < vec4) {
                FlashDecodeVec4T vh = flash_load_as_half4(v_base, (ulong)kv_idx * (ulong)stride_v_s + ((ulong)lane << 2)); // INDEX64_OK
                warp_out += half4_to_float4(vh) * w;
            }
            if (lane == 0) {
                warp_sum += w;
            }
        }
        shared_warp_out[warp * 32 + lane] = flash_pack_out4<TG_OUT_HALF>(warp_out);
        if (lane == 0) {
            shared_warp_sums[warp] = warp_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp == 0) {
            float4 block_out = float4(0.0f);
            if (lane < vec4) {
                for (uint widx = 0; widx < WARPS; ++widx) {
                    block_out += flash_unpack_out4<TG_OUT_HALF>(shared_warp_out[widx * 32 + lane]);
                }
            }

            float alpha_val = 0.0f;
            float beta_val = 0.0f;

            // Parallel reduction of block sum using simd_sum
            float block_sum = 0.0f;
            // Lane 0..7 load the 8 warp sums.
            if (lane < WARPS) {
                block_sum = shared_warp_sums[lane];
            }
            // All lanes in warp 0 participate in the reduction.
            block_sum = simd_sum(block_sum);

            if (lane == 0) {
                float m_new = max(m, block_max);
                alpha = metal::fast::exp(m - m_new);
                beta = metal::fast::exp(block_max - m_new);
                l = l * alpha + block_sum * beta;
                m = m_new;

                shared_warp_sums[0] = l;
                alpha_val = alpha;
                beta_val = beta;
            }
            
            float alpha_b = simd_broadcast(alpha_val, 0);
            float beta_b = simd_broadcast(beta_val, 0);
            
            out_acc = out_acc * alpha_b + block_out * beta_b;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (warp == 0 && lane < vec4) {
        float denom = shared_warp_sums[0];
        float inv = (isfinite(denom) && abs(denom) > 1e-6f) ? (1.0f / denom) : 1.0f;
        float4 out = out_acc * inv;
        out[0] = isfinite(out[0]) ? out[0] : 0.0f;
        out[1] = isfinite(out[1]) ? out[1] : 0.0f;
        out[2] = isfinite(out[2]) ? out[2] : 0.0f;
        out[3] = isfinite(out[3]) ? out[3] : 0.0f;

        flash_store_from_float4(output, ((ulong)lane << 2), out); // INDEX64_OK
    }
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
inline void flash_decode_warp_tiled_m1_half2_mma(
    const threadgroup FlashDecodeVec2T* q2_shared,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device OutputStorageT* output,
    uint warp,
    uint lane,
    constant SdpaParams& params,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type* shared_warp_out,
    threadgroup float* shared_warp_scores
) {
    (void)shared_warp_scores;
    uint kv_len = params.kv_len;
    uint head_dim = params.head_dim;
    uint stride_k_s = params.stride_k_s;
    uint stride_v_s = params.stride_v_s;

    if (head_dim != 64 || stride_k_s != 64 || stride_v_s != 64) {
        return;
    }
    if (kv_len == 0 || kv_len > 65536) {
        return;
    }
    if ((KEYS_PER_WARP % 8u) != 0u) {
        return;
    }

    const uint qid = lane / 4u;
    const uint sm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint sn = ((qid & 2u) * 2u) + ((lane % 2u) * 2u);

    float m = -1e30f;
    float l = 0.0f;
    float alpha = 1.0f;
    float beta = 0.0f;
    float2 out_acc = float2(0.0f);
    float scale = params.scale;

    constexpr uint KEYS = WARPS * KEYS_PER_WARP;
    constexpr uint D_ITERS = 8u; // head_dim==64 and d0 step==8
    constexpr uint TILE_KEYS = 8u;

    for (uint base = 0; base < kv_len; base += KEYS) {
        float warp_m = -1e30f;
        float warp_l = 0.0f;
        float2 warp_out = float2(0.0f);
        float q0[D_ITERS];
        float q1[D_ITERS];

        #pragma unroll
        for (uint di = 0; di < D_ITERS; ++di) {
            uint d0 = di * 8u;
            q0[di] = flash_q2_scalar(q2_shared, d0 + sn);
            q1[di] = flash_q2_scalar(q2_shared, d0 + sn + 1u);
        }

        const uint tiles = KEYS_PER_WARP / TILE_KEYS;
        // K score tile ping-pong in registers.
        float score_cur0 = -1e30f;
        float score_cur1 = -1e30f;
        float score_next0 = -1e30f;
        float score_next1 = -1e30f;
        {
            uint j = 0u;
            simdgroup_matrix<float, 8, 8> A;
            simdgroup_matrix<float, 8, 8> B;
            simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0.0f);
            #pragma unroll
            for (uint di = 0; di < D_ITERS; ++di) {
                uint d0 = di * 8u;
                A.thread_elements()[0] = q0[di];
                A.thread_elements()[1] = q1[di];
                uint kv_idx0 = base + warp + (j + sn) * WARPS;
                uint kv_idx1 = base + warp + (j + sn + 1u) * WARPS;
                ulong k_comp_idx = (ulong)d0 + (ulong)sm; // INDEX64_OK
                float b0 = 0.0f;
                float b1 = 0.0f;
                if (kv_idx0 < kv_len) {
                    b0 = (float)flash_load_as_half(k_base, (ulong)kv_idx0 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                }
                if (kv_idx1 < kv_len) {
                    b1 = (float)flash_load_as_half(k_base, (ulong)kv_idx1 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                }
                B.thread_elements()[0] = b0;
                B.thread_elements()[1] = b1;
                simdgroup_multiply_accumulate(C, A, B, C);
            }
            score_cur0 = C.thread_elements()[0] * scale;
            score_cur1 = C.thread_elements()[1] * scale;
            score_cur0 = isfinite(score_cur0) ? score_cur0 : -1e30f;
            score_cur1 = isfinite(score_cur1) ? score_cur1 : -1e30f;
        }

        for (uint tile = 0; tile < tiles; ++tile) {
            uint j_base = tile * TILE_KEYS;
            uint j_next = j_base + TILE_KEYS;

            // Preload next K tile while consuming current K tile.
            if (j_next < KEYS_PER_WARP) {
                uint j = j_next;
                simdgroup_matrix<float, 8, 8> A;
                simdgroup_matrix<float, 8, 8> B;
                simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0.0f);

                #pragma unroll
                for (uint di = 0; di < D_ITERS; ++di) {
                    uint d0 = di * 8u;
                    A.thread_elements()[0] = q0[di];
                    A.thread_elements()[1] = q1[di];

                    uint kv_idx0 = base + warp + (j + sn) * WARPS;
                    uint kv_idx1 = base + warp + (j + sn + 1u) * WARPS;
                    ulong k_comp_idx = (ulong)d0 + (ulong)sm; // INDEX64_OK

                    float b0 = 0.0f;
                    float b1 = 0.0f;
                    if (kv_idx0 < kv_len) {
                        b0 = (float)flash_load_as_half(k_base, (ulong)kv_idx0 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                    }
                    if (kv_idx1 < kv_len) {
                        b1 = (float)flash_load_as_half(k_base, (ulong)kv_idx1 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                    }
                    B.thread_elements()[0] = b0;
                    B.thread_elements()[1] = b1;

                    simdgroup_multiply_accumulate(C, A, B, C);
                }
                score_next0 = C.thread_elements()[0] * scale;
                score_next1 = C.thread_elements()[1] * scale;
                score_next0 = isfinite(score_next0) ? score_next0 : -1e30f;
                score_next1 = isfinite(score_next1) ? score_next1 : -1e30f;
            }

            #pragma unroll
            for (uint t = 0; t < TILE_KEYS; ++t) {
                uint idx = j_base + t;
                uint kv_idx = base + warp + idx * WARPS;
                float score = flash_decode_tile_score_from_pair(score_cur0, score_cur1, t);
                float alpha_step = 1.0f;
                float beta_step = 0.0f;
                if (lane == 0 && kv_idx < kv_len) {
                    float m_new = max(warp_m, score);
                    alpha_step = metal::fast::exp(warp_m - m_new);
                    beta_step = metal::fast::exp(score - m_new);
                    warp_l = warp_l * alpha_step + beta_step;
                    warp_m = m_new;
                }
                float alpha_b = simd_broadcast(alpha_step, 0);
                float beta_b = simd_broadcast(beta_step, 0);
                if (kv_idx < kv_len) {
                    FlashDecodeVec2T vh = flash_load_as_half2(v_base, (ulong)kv_idx * (ulong)stride_v_s + ((ulong)lane << 1)); // INDEX64_OK
                    warp_out = warp_out * alpha_b + half2_to_float2(vh) * beta_b;
                }
            }
            score_cur0 = score_next0;
            score_cur1 = score_next1;
        }

        if (lane == 0) {
            shared_warp_max[warp] = warp_m;
            shared_warp_sums[warp] = warp_l;
        }
        shared_warp_out[warp * 32 + lane] = flash_pack_out2<TG_OUT_HALF>(warp_out);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp == 0) {
            float val = (lane < WARPS) ? shared_warp_max[lane] : -1e30f;
            float block_max = simd_max(val);

            float block_sum_part = 0.0f;
            if (lane < WARPS) {
                float wm = shared_warp_max[lane];
                float wl = shared_warp_sums[lane];
                block_sum_part = (wl > 0.0f) ? wl * metal::fast::exp(wm - block_max) : 0.0f;
            }
            float block_sum = simd_sum(block_sum_part);

            float2 block_out = float2(0.0f);
            for (uint widx = 0; widx < WARPS; ++widx) {
                float w_scale = metal::fast::exp(shared_warp_max[widx] - block_max);
                block_out += flash_unpack_out2<TG_OUT_HALF>(shared_warp_out[widx * 32 + lane]) * w_scale;
            }

            float alpha_val = 0.0f;
            float beta_val = 0.0f;
            if (lane == 0) {
                float m_new = max(m, block_max);
                alpha = metal::fast::exp(m - m_new);
                beta = metal::fast::exp(block_max - m_new);
                l = l * alpha + block_sum * beta;
                m = m_new;
                shared_warp_sums[0] = l;
                alpha_val = alpha;
                beta_val = beta;
            }

            float alpha_b = simd_broadcast(alpha_val, 0);
            float beta_b = simd_broadcast(beta_val, 0);
            out_acc = out_acc * alpha_b + block_out * beta_b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (warp == 0) {
        float denom = shared_warp_sums[0];
        float inv = (isfinite(denom) && abs(denom) > 1e-6f) ? (1.0f / denom) : 1.0f;
        float2 out = out_acc * inv;
        out[0] = isfinite(out[0]) ? out[0] : 0.0f;
        out[1] = isfinite(out[1]) ? out[1] : 0.0f;
        flash_store_from_float2(output, ((ulong)lane << 1), out); // INDEX64_OK
    }
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
inline void flash_decode_warp_tiled_m1_half4_mma(
    const threadgroup FlashDecodeVec4T* q4_shared,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device OutputStorageT* output,
    uint warp,
    uint lane,
    constant SdpaParams& params,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type* shared_warp_out,
    threadgroup float* shared_warp_scores
) {
    (void)shared_warp_scores;
    uint kv_len = params.kv_len;
    uint head_dim = params.head_dim;
    uint stride_k_s = params.stride_k_s;
    uint stride_v_s = params.stride_v_s;

    if (!((head_dim == 64) || (head_dim == 128))) {
        return;
    }
    if (stride_k_s != head_dim || stride_v_s != head_dim) {
        return;
    }
    if (kv_len == 0 || kv_len > 65536) {
        return;
    }
    if ((KEYS_PER_WARP % 8u) != 0u) {
        return;
    }

    const uint vec4 = head_dim >> 2;
    const uint qid = lane / 4u;
    const uint sm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint sn = ((qid & 2u) * 2u) + ((lane % 2u) * 2u);

    float m = -1e30f;
    float l = 0.0f;
    float alpha = 1.0f;
    float beta = 0.0f;
    float4 out_acc = float4(0.0f);
    float scale = params.scale;

    constexpr uint KEYS = WARPS * KEYS_PER_WARP;
    constexpr uint TILE_KEYS = 8u;
    for (uint base = 0; base < kv_len; base += KEYS) {
        float warp_m = -1e30f;
        float warp_l = 0.0f;
        float4 warp_out = float4(0.0f);
        float q0[16];
        float q1[16];
        const uint tiles = KEYS_PER_WARP / TILE_KEYS;

        if (head_dim == 64u) {
            #pragma unroll
            for (uint di = 0; di < 8u; ++di) {
                uint d0 = di * 8u;
                q0[di] = flash_q4_scalar(q4_shared, d0 + sn);
                q1[di] = flash_q4_scalar(q4_shared, d0 + sn + 1u);
            }
        } else {
            #pragma unroll
            for (uint di = 0; di < 16u; ++di) {
                uint d0 = di * 8u;
                q0[di] = flash_q4_scalar(q4_shared, d0 + sn);
                q1[di] = flash_q4_scalar(q4_shared, d0 + sn + 1u);
            }
        }

        // K score tile ping-pong in registers.
        float score_cur0 = -1e30f;
        float score_cur1 = -1e30f;
        float score_next0 = -1e30f;
        float score_next1 = -1e30f;
        {
            uint j = 0u;
            simdgroup_matrix<float, 8, 8> A;
            simdgroup_matrix<float, 8, 8> B;
            simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0.0f);

            if (head_dim == 64u) {
                #pragma unroll
                for (uint di = 0; di < 8u; ++di) {
                    uint d0 = di * 8u;
                    A.thread_elements()[0] = q0[di];
                    A.thread_elements()[1] = q1[di];

                    uint kv_idx0 = base + warp + (j + sn) * WARPS;
                    uint kv_idx1 = base + warp + (j + sn + 1u) * WARPS;
                    ulong k_comp_idx = (ulong)d0 + (ulong)sm; // INDEX64_OK

                    float b0 = 0.0f;
                    float b1 = 0.0f;
                    if (kv_idx0 < kv_len) {
                        b0 = (float)flash_load_as_half(k_base, (ulong)kv_idx0 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                    }
                    if (kv_idx1 < kv_len) {
                        b1 = (float)flash_load_as_half(k_base, (ulong)kv_idx1 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                    }
                    B.thread_elements()[0] = b0;
                    B.thread_elements()[1] = b1;

                    simdgroup_multiply_accumulate(C, A, B, C);
                }
            } else {
                #pragma unroll
                for (uint di = 0; di < 16u; ++di) {
                    uint d0 = di * 8u;
                    A.thread_elements()[0] = q0[di];
                    A.thread_elements()[1] = q1[di];

                    uint kv_idx0 = base + warp + (j + sn) * WARPS;
                    uint kv_idx1 = base + warp + (j + sn + 1u) * WARPS;
                    ulong k_comp_idx = (ulong)d0 + (ulong)sm; // INDEX64_OK

                    float b0 = 0.0f;
                    float b1 = 0.0f;
                    if (kv_idx0 < kv_len) {
                        b0 = (float)flash_load_as_half(k_base, (ulong)kv_idx0 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                    }
                    if (kv_idx1 < kv_len) {
                        b1 = (float)flash_load_as_half(k_base, (ulong)kv_idx1 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                    }
                    B.thread_elements()[0] = b0;
                    B.thread_elements()[1] = b1;

                    simdgroup_multiply_accumulate(C, A, B, C);
                }
            }
            score_cur0 = C.thread_elements()[0] * scale;
            score_cur1 = C.thread_elements()[1] * scale;
            score_cur0 = isfinite(score_cur0) ? score_cur0 : -1e30f;
            score_cur1 = isfinite(score_cur1) ? score_cur1 : -1e30f;
        }

        for (uint tile = 0; tile < tiles; ++tile) {
            uint j_base = tile * TILE_KEYS;
            uint j_next = j_base + TILE_KEYS;

            // Preload next tile into the alternate slot.
            if (j_next < KEYS_PER_WARP) {
                uint j = j_next;
                simdgroup_matrix<float, 8, 8> A;
                simdgroup_matrix<float, 8, 8> B;
                simdgroup_matrix<float, 8, 8> C = simdgroup_matrix<float, 8, 8>(0.0f);

                if (head_dim == 64u) {
                    #pragma unroll
                    for (uint di = 0; di < 8u; ++di) {
                        uint d0 = di * 8u;
                        A.thread_elements()[0] = q0[di];
                        A.thread_elements()[1] = q1[di];

                        uint kv_idx0 = base + warp + (j + sn) * WARPS;
                        uint kv_idx1 = base + warp + (j + sn + 1u) * WARPS;
                        ulong k_comp_idx = (ulong)d0 + (ulong)sm; // INDEX64_OK

                        float b0 = 0.0f;
                        float b1 = 0.0f;
                        if (kv_idx0 < kv_len) {
                            b0 = (float)flash_load_as_half(k_base, (ulong)kv_idx0 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                        }
                        if (kv_idx1 < kv_len) {
                            b1 = (float)flash_load_as_half(k_base, (ulong)kv_idx1 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                        }
                        B.thread_elements()[0] = b0;
                        B.thread_elements()[1] = b1;

                        simdgroup_multiply_accumulate(C, A, B, C);
                    }
                } else {
                    #pragma unroll
                    for (uint di = 0; di < 16u; ++di) {
                        uint d0 = di * 8u;
                        A.thread_elements()[0] = q0[di];
                        A.thread_elements()[1] = q1[di];

                        uint kv_idx0 = base + warp + (j + sn) * WARPS;
                        uint kv_idx1 = base + warp + (j + sn + 1u) * WARPS;
                        ulong k_comp_idx = (ulong)d0 + (ulong)sm; // INDEX64_OK

                        float b0 = 0.0f;
                        float b1 = 0.0f;
                        if (kv_idx0 < kv_len) {
                            b0 = (float)flash_load_as_half(k_base, (ulong)kv_idx0 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                        }
                        if (kv_idx1 < kv_len) {
                            b1 = (float)flash_load_as_half(k_base, (ulong)kv_idx1 * (ulong)stride_k_s + k_comp_idx); // INDEX64_OK
                        }
                        B.thread_elements()[0] = b0;
                        B.thread_elements()[1] = b1;

                        simdgroup_multiply_accumulate(C, A, B, C);
                    }
                }

                score_next0 = C.thread_elements()[0] * scale;
                score_next1 = C.thread_elements()[1] * scale;
                score_next0 = isfinite(score_next0) ? score_next0 : -1e30f;
                score_next1 = isfinite(score_next1) ? score_next1 : -1e30f;
            }

            #pragma unroll
            for (uint t = 0; t < TILE_KEYS; ++t) {
                uint idx = j_base + t;
                uint kv_idx = base + warp + idx * WARPS;
                float score = flash_decode_tile_score_from_pair(score_cur0, score_cur1, t);
                float alpha_step = 1.0f;
                float beta_step = 0.0f;
                if (lane == 0 && kv_idx < kv_len) {
                    float m_new = max(warp_m, score);
                    alpha_step = metal::fast::exp(warp_m - m_new);
                    beta_step = metal::fast::exp(score - m_new);
                    warp_l = warp_l * alpha_step + beta_step;
                    warp_m = m_new;
                }
                float alpha_b = simd_broadcast(alpha_step, 0);
                float beta_b = simd_broadcast(beta_step, 0);
                if (kv_idx < kv_len && lane < vec4) {
                    FlashDecodeVec4T vh = flash_load_as_half4(v_base, (ulong)kv_idx * (ulong)stride_v_s + ((ulong)lane << 2)); // INDEX64_OK
                    warp_out = warp_out * alpha_b + half4_to_float4(vh) * beta_b;
                }
            }
            score_cur0 = score_next0;
            score_cur1 = score_next1;
        }

        if (lane == 0) {
            shared_warp_max[warp] = warp_m;
            shared_warp_sums[warp] = warp_l;
        }
        shared_warp_out[warp * 32 + lane] = flash_pack_out4<TG_OUT_HALF>(warp_out);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (warp == 0) {
            float val = (lane < WARPS) ? shared_warp_max[lane] : -1e30f;
            float block_max = simd_max(val);

            float block_sum_part = 0.0f;
            if (lane < WARPS) {
                float wm = shared_warp_max[lane];
                float wl = shared_warp_sums[lane];
                block_sum_part = (wl > 0.0f) ? wl * metal::fast::exp(wm - block_max) : 0.0f;
            }
            float block_sum = simd_sum(block_sum_part);

            float4 block_out = float4(0.0f);
            if (lane < vec4) {
                for (uint widx = 0; widx < WARPS; ++widx) {
                    float w_scale = metal::fast::exp(shared_warp_max[widx] - block_max);
                    block_out += flash_unpack_out4<TG_OUT_HALF>(shared_warp_out[widx * 32 + lane]) * w_scale;
                }
            }

            float alpha_val = 0.0f;
            float beta_val = 0.0f;
            if (lane == 0) {
                float m_new = max(m, block_max);
                alpha = metal::fast::exp(m - m_new);
                beta = metal::fast::exp(block_max - m_new);
                l = l * alpha + block_sum * beta;
                m = m_new;

                shared_warp_sums[0] = l;
                alpha_val = alpha;
                beta_val = beta;
            }

            float alpha_b = simd_broadcast(alpha_val, 0);
            float beta_b = simd_broadcast(beta_val, 0);
            out_acc = out_acc * alpha_b + block_out * beta_b;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (warp == 0 && lane < vec4) {
        float denom = shared_warp_sums[0];
        float inv = (isfinite(denom) && abs(denom) > 1e-6f) ? (1.0f / denom) : 1.0f;
        float4 out = out_acc * inv;
        out[0] = isfinite(out[0]) ? out[0] : 0.0f;
        out[1] = isfinite(out[1]) ? out[1] : 0.0f;
        out[2] = isfinite(out[2]) ? out[2] : 0.0f;
        out[3] = isfinite(out[3]) ? out[3] : 0.0f;
        flash_store_from_float4(output, ((ulong)lane << 2), out); // INDEX64_OK
    }
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF, bool USE_MMA>
ALWAYS_INLINE void run_flash_decode_fused_half2_stage(
    const threadgroup FlashDecodeVec2T* q_vec,
    const device InputStorageT* k_ptr,
    const device InputStorageT* v_ptr,
    device OutputStorageT* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type* shared_warp_out,
    threadgroup float* shared_warp_scores
) {
    if (USE_MMA) {
        flash_decode_warp_tiled_m1_half2_mma<WARPS, KEYS_PER_WARP, TG_OUT_HALF>(
            q_vec,
            k_ptr,
            v_ptr,
            output_ptr,
            warp,
            lane,
            params,
            shared_warp_max,
            shared_warp_sums,
            shared_warp_out,
            shared_warp_scores
        );
    } else {
        flash_decode_warp_tiled_m1_half2<WARPS, KEYS_PER_WARP, TG_OUT_HALF>(
            q_vec,
            k_ptr,
            v_ptr,
            output_ptr,
            warp,
            lane,
            params,
            shared_warp_max,
            shared_warp_sums,
            shared_warp_out
        );
    }
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF, bool USE_MMA>
ALWAYS_INLINE void run_flash_decode_fused_half4_stage(
    const threadgroup FlashDecodeVec4T* q_vec,
    const device InputStorageT* k_ptr,
    const device InputStorageT* v_ptr,
    device OutputStorageT* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type* shared_warp_out,
    threadgroup float* shared_warp_scores
) {
    if (USE_MMA) {
        flash_decode_warp_tiled_m1_half4_mma<WARPS, KEYS_PER_WARP, TG_OUT_HALF>(
            q_vec,
            k_ptr,
            v_ptr,
            output_ptr,
            warp,
            lane,
            params,
            shared_warp_max,
            shared_warp_sums,
            shared_warp_out,
            shared_warp_scores
        );
    } else {
        flash_decode_warp_tiled_m1_half4<WARPS, KEYS_PER_WARP, TG_OUT_HALF>(
            q_vec,
            k_ptr,
            v_ptr,
            output_ptr,
            warp,
            lane,
            params,
            shared_warp_max,
            shared_warp_sums,
            shared_warp_out
        );
    }
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF, bool USE_MMA>
ALWAYS_INLINE void run_flash_decode_standalone_half2_stage(
    const device InputStorageT* q_ptr,
    const device InputStorageT* k_ptr,
    const device InputStorageT* v_ptr,
    device OutputStorageT* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup FlashDecodeVec2T* q_shared,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type* shared_warp_out,
    threadgroup float* shared_warp_scores
) {
    if (warp == 0) {
        q_shared[lane] = flash_load_as_half2(q_ptr, ((ulong)lane << 1)); // INDEX64_OK
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    run_flash_decode_fused_half2_stage<WARPS, KEYS_PER_WARP, TG_OUT_HALF, USE_MMA>(
        q_shared,
        k_ptr,
        v_ptr,
        output_ptr,
        params,
        warp,
        lane,
        shared_warp_max,
        shared_warp_sums,
        shared_warp_out,
        shared_warp_scores
    );
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF, bool USE_MMA, uint Q_VEC4>
ALWAYS_INLINE void run_flash_decode_standalone_half4_stage(
    const device InputStorageT* q_ptr,
    const device InputStorageT* k_ptr,
    const device InputStorageT* v_ptr,
    device OutputStorageT* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup FlashDecodeVec4T* q_shared,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type* shared_warp_out,
    threadgroup float* shared_warp_scores
) {
    if (warp == 0 && lane < Q_VEC4) {
        q_shared[lane] = flash_load_as_half4(q_ptr, ((ulong)lane << 2)); // INDEX64_OK
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    run_flash_decode_fused_half4_stage<WARPS, KEYS_PER_WARP, TG_OUT_HALF, USE_MMA>(
        q_shared,
        k_ptr,
        v_ptr,
        output_ptr,
        params,
        warp,
        lane,
        shared_warp_max,
        shared_warp_sums,
        shared_warp_out,
        shared_warp_scores
    );
}

#define FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF2(WARPS, TG_OUT_HALF, MAX_NAME, SUMS_NAME, OUT_NAME) \
    threadgroup float MAX_NAME[WARPS];                                                                  \
    threadgroup float SUMS_NAME[WARPS];                                                                 \
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type OUT_NAME[WARPS * 32]

#define FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF2_MMA(WARPS, KEYS_PER_WARP, TG_OUT_HALF, MAX_NAME, SUMS_NAME, OUT_NAME, SCORES_NAME) \
    threadgroup float MAX_NAME[WARPS];                                                                                                 \
    threadgroup float SUMS_NAME[WARPS];                                                                                                \
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type OUT_NAME[WARPS * 32];                                                         \
    threadgroup float SCORES_NAME[1]

#define FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF4(WARPS, TG_OUT_HALF, MAX_NAME, SUMS_NAME, OUT_NAME) \
    threadgroup float MAX_NAME[WARPS];                                                                  \
    threadgroup float SUMS_NAME[WARPS];                                                                 \
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type OUT_NAME[WARPS * 32]

#define FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF4_MMA(WARPS, KEYS_PER_WARP, TG_OUT_HALF, MAX_NAME, SUMS_NAME, OUT_NAME, SCORES_NAME) \
    threadgroup float MAX_NAME[WARPS];                                                                                                 \
    threadgroup float SUMS_NAME[WARPS];                                                                                                \
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type OUT_NAME[WARPS * 32];                                                         \
    threadgroup float SCORES_NAME[1]

#define FLASH_DECODE_DECLARE_Q_SHARED_HALF2(NAME) threadgroup FlashDecodeVec2T NAME[32]
#define FLASH_DECODE_DECLARE_Q_SHARED_HALF4(NAME) threadgroup FlashDecodeVec4T NAME[32]
