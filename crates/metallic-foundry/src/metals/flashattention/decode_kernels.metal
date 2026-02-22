template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
inline void flash_decode_warp_tiled_m1_half2(
    const threadgroup half2* q2_shared,
    const device half* k_base,
    const device half* v_base,
    device half* output,
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
                const device half2* k2 = (const device half2*)(k_base + kv_idx * stride_k_s);
                half2 qh = q2_shared[lane];
                half2 kh = k2[lane];
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
            const device half2* v2 = (const device half2*)(v_base + kv_idx * stride_v_s);
            warp_out += half2_to_float2(v2[lane]) * w;
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
        // Store as half2; output is contiguous [head_dim] for this head.
        device half2* out2 = (device half2*)output;
        out2[lane] = half2((half)out[0], (half)out[1]);
    }
}

// -----------------------------------------------------------------------------
// half4 path (supports head_dim==64 and head_dim==128).
// Intended for fused RoPE -> SDPA compound kernels where Q is naturally stored as half4.
// -----------------------------------------------------------------------------
template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
inline void flash_decode_warp_tiled_m1_half4(
    const threadgroup half4* q4_shared,
    const device half* k_base,
    const device half* v_base,
    device half* output,
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

    const uint vec4 = head_dim >> 2; // number of half4 per head

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
                const device half4* k4 = (const device half4*)(k_base + kv_idx * stride_k_s);
                half4 qh = q4_shared[lane];
                half4 kh = k4[lane];
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

        // Compute the block max in each warp (see half2 path above).
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
                const device half4* v4 = (const device half4*)(v_base + kv_idx * stride_v_s);
                warp_out += half4_to_float4(v4[lane]) * w;
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

        device half4* out4 = (device half4*)output;
        out4[lane] = half4((half)out[0], (half)out[1], (half)out[2], (half)out[3]);
    }
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
ALWAYS_INLINE void run_flash_decode_fused_half2_stage(
    const threadgroup half2* q_vec,
    const device half* k_ptr,
    const device half* v_ptr,
    device half* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type* shared_warp_out
) {
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

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
ALWAYS_INLINE void run_flash_decode_fused_half4_stage(
    const threadgroup half4* q_vec,
    const device half* k_ptr,
    const device half* v_ptr,
    device half* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type* shared_warp_out
) {
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

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF>
ALWAYS_INLINE void run_flash_decode_standalone_half2_stage(
    const device half* q_ptr,
    const device half* k_ptr,
    const device half* v_ptr,
    device half* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup half2* q_shared,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type* shared_warp_out
) {
    if (warp == 0) {
        q_shared[lane] = ((const device half2*)q_ptr)[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    run_flash_decode_fused_half2_stage<WARPS, KEYS_PER_WARP, TG_OUT_HALF>(
        q_shared,
        k_ptr,
        v_ptr,
        output_ptr,
        params,
        warp,
        lane,
        shared_warp_max,
        shared_warp_sums,
        shared_warp_out
    );
}

template<uint WARPS, uint KEYS_PER_WARP, bool TG_OUT_HALF, uint Q_VEC4>
ALWAYS_INLINE void run_flash_decode_standalone_half4_stage(
    const device half* q_ptr,
    const device half* k_ptr,
    const device half* v_ptr,
    device half* output_ptr,
    constant SdpaParams& params,
    uint warp,
    uint lane,
    threadgroup half4* q_shared,
    threadgroup float* shared_warp_max,
    threadgroup float* shared_warp_sums,
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type* shared_warp_out
) {
    if (warp == 0 && lane < Q_VEC4) {
        q_shared[lane] = ((const device half4*)q_ptr)[lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    run_flash_decode_fused_half4_stage<WARPS, KEYS_PER_WARP, TG_OUT_HALF>(
        q_shared,
        k_ptr,
        v_ptr,
        output_ptr,
        params,
        warp,
        lane,
        shared_warp_max,
        shared_warp_sums,
        shared_warp_out
    );
}

#define FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF2(WARPS, TG_OUT_HALF, MAX_NAME, SUMS_NAME, OUT_NAME) \
    threadgroup float MAX_NAME[WARPS];                                                                  \
    threadgroup float SUMS_NAME[WARPS];                                                                 \
    threadgroup typename FlashTgOut2<TG_OUT_HALF>::type OUT_NAME[WARPS * 32]

#define FLASH_DECODE_DECLARE_REDUCE_SHARED_HALF4(WARPS, TG_OUT_HALF, MAX_NAME, SUMS_NAME, OUT_NAME) \
    threadgroup float MAX_NAME[WARPS];                                                                  \
    threadgroup float SUMS_NAME[WARPS];                                                                 \
    threadgroup typename FlashTgOut4<TG_OUT_HALF>::type OUT_NAME[WARPS * 32]

#define FLASH_DECODE_DECLARE_Q_SHARED_HALF2(NAME) threadgroup half2 NAME[32]
#define FLASH_DECODE_DECLARE_Q_SHARED_HALF4(NAME) threadgroup half4 NAME[32]

