// ---- Split-K Prefill (FA1 completion) ----

// NOTE: `SdpaPrefillSplitKParams` is defined by the Rust stage and injected via `struct_defs`.

inline void splitk_tile_range(uint kv_len, uint split_k, uint split_id, thread uint& tile_begin, thread uint& tile_end) {
    uint tiles = (kv_len + 31) / 32;
    uint tiles_per_split = (tiles + split_k - 1) / split_k;
    tile_begin = split_id * tiles_per_split;
    tile_end = min(tiles, tile_begin + tiles_per_split);
}

template<uint WARPS>
inline void flash_prefill_splitk_part_d64(
    const device InputStorageT* q_base,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device float* partial_acc,
    device float* partial_m,
    device float* partial_l,
    constant SdpaPrefillSplitKParams& params,
    threadgroup FlashTileT* k_shared,
    threadgroup FlashTileT* v_shared,
    uint3 gid,
    uint3 tid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id
) {
    constexpr uint TILE_M = WARPS * 4;

    uint tile_m_idx = gid.x;
    uint head_idx = gid.y;
    uint split_id = gid.z;

    uint warp = simd_group_id;
    uint lane = simd_lane_id;
    if (warp >= WARPS) {
        return;
    }

    uint q_row_start = tile_m_idx * TILE_M;
    uint my_row_local_base = warp * 4;

    float2 acc[4];
    float m[4];
    float l[4];
    FlashVec2T q_reg[4];
    bool row_valid[4];
    for (int r = 0; r < 4; ++r) {
        acc[r] = float2(0.0f);
        m[r] = -1e30f;
        l[r] = 0.0f;

        uint q_idx = q_row_start + my_row_local_base + r;
        row_valid[r] = q_idx < params.q_len;
        if (row_valid[r]) {
            const device InputStorageT* q_ptr = q_base + q_idx * params.q_stride_m;
            q_reg[r] = FlashVec2T((FlashTileT)q_ptr[lane], (FlashTileT)q_ptr[lane + 32u]);
        } else {
            q_reg[r] = FlashVec2T((FlashTileT)0.0f, (FlashTileT)0.0f);
        }
    }

    uint tile_begin = 0;
    uint tile_end = 0;
    splitk_tile_range(params.kv_len, max((uint)1, params.split_k), split_id, tile_begin, tile_end);

    for (uint k_tile_idx = tile_begin; k_tile_idx < tile_end; ++k_tile_idx) {
        uint k_start = k_tile_idx * 32;
        uint stored_rows = min((uint)32, params.kv_len - k_start);

        uint linear_tid = tid.x;
        for (uint load_tid = linear_tid; load_tid < 256; load_tid += tptg.x) {
            load_tile(k_base + k_start * params.stride_k_s, k_shared, params.stride_k_s, stored_rows, load_tid);
            load_tile(v_base + k_start * params.stride_v_s, v_shared, params.stride_v_s, stored_rows, load_tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int r = 0; r < 4; ++r) {
            uint q_global = q_row_start + my_row_local_base + r;
            uint causal_limit = q_global + params.query_offset;
            if (!row_valid[r]) {
                continue;
            }

            float block_max = -1e30f;
            for (int k = 0; k < 32; ++k) {
                uint k_global = k_start + (uint)k;
                if (k_global >= params.kv_len) break;
                bool masked = (k_global > causal_limit);
                FlashTileT k0 = masked ? (FlashTileT)0.0f : k_shared[k * 64 + lane];
                FlashTileT k1 = masked ? (FlashTileT)0.0f : k_shared[k * 64 + lane + 32];
                float partial = (float)q_reg[r][0] * (float)k0 + (float)q_reg[r][1] * (float)k1;
                float score = simd_sum(partial) * params.scale;
                score = masked ? -1e30f : score;
                block_max = max(block_max, score);
            }

            float2 block_out = float2(0.0f);
            float block_sum = 0.0f;
            for (int k = 0; k < 32; ++k) {
                uint k_global = k_start + (uint)k;
                if (k_global >= params.kv_len) break;
                if (k_global > causal_limit) continue;

                FlashTileT k0 = k_shared[k * 64 + lane];
                FlashTileT k1 = k_shared[k * 64 + lane + 32];
                float partial = (float)q_reg[r][0] * (float)k0 + (float)q_reg[r][1] * (float)k1;
                float score = simd_sum(partial) * params.scale;

                float p_local = 0.0f;
                if (lane == 0) {
                    p_local = metal::fast::exp(score - block_max);
                    block_sum += p_local;
                }
                float p = simd_broadcast(p_local, 0);

                FlashTileT v0 = v_shared[k * 64 + lane];
                FlashTileT v1 = v_shared[k * 64 + lane + 32];
                block_out[0] += p * (float)v0;
                block_out[1] += p * (float)v1;
            }

            float alpha_local = 0.0f;
            float beta_local = 0.0f;
            if (lane == 0) {
                float m_prev = m[r];
                float l_prev = l[r];
                float m_new = max(m_prev, block_max);
                float alpha = metal::fast::exp(m_prev - m_new);
                float beta = metal::fast::exp(block_max - m_new);
                m[r] = m_new;
                l[r] = l_prev * alpha + block_sum * beta;
                alpha_local = alpha;
                beta_local = beta;
            }
            float alpha_b = simd_broadcast(alpha_local, 0);
            float beta_b = simd_broadcast(beta_local, 0);
            acc[r] = acc[r] * alpha_b + block_out * beta_b;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint q_tile_count = (params.q_len + TILE_M - 1) / TILE_M;
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx; // INDEX64_OK
    ulong split_linear = tile_linear * (ulong)max((uint)1, params.split_k) + (ulong)split_id; // INDEX64_OK
    ulong m_base = split_linear * (ulong)TILE_M; // INDEX64_OK
    ulong acc_base = m_base * (ulong)params.head_dim; // INDEX64_OK

    for (int r = 0; r < 4; ++r) {
        uint row_in_tile = my_row_local_base + (uint)r;
        uint q_global = q_row_start + row_in_tile;

        if (q_global >= params.q_len) {
            if (lane == 0) {
                partial_m[m_base + row_in_tile] = -1e30f;
                partial_l[m_base + row_in_tile] = 0.0f;
            }
            partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + lane] = 0.0f; // INDEX64_OK
            partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + lane + 32u] = 0.0f; // INDEX64_OK
            continue;
        }

        if (lane == 0) {
            partial_m[m_base + row_in_tile] = m[r];
            partial_l[m_base + row_in_tile] = l[r];
        }
        partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + lane] = acc[r][0]; // INDEX64_OK
        partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + lane + 32u] = acc[r][1]; // INDEX64_OK
    }
}

template<uint WARPS>
inline void flash_prefill_splitk_part_d128(
    const device InputStorageT* q_base,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device float* partial_acc,
    device float* partial_m,
    device float* partial_l,
    constant SdpaPrefillSplitKParams& params,
    threadgroup FlashTileT* k_shared,
    threadgroup FlashTileT* v_shared,
    uint3 gid,
    uint3 tid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id
) {
    constexpr uint TILE_M = WARPS * 4;

    uint tile_m_idx = gid.x;
    uint head_idx = gid.y;
    uint split_id = gid.z;

    uint warp = simd_group_id;
    uint lane = simd_lane_id;
    if (warp >= WARPS) {
        return;
    }

    uint q_row_start = tile_m_idx * TILE_M;
    uint my_row_local_base = warp * 4;

    float4 acc[4];
    float m[4];
    float l[4];
    FlashVec4T q_reg[4];
    bool row_valid[4];
    for (int r = 0; r < 4; ++r) {
        acc[r] = float4(0.0f);
        m[r] = -1e30f;
        l[r] = 0.0f;

        uint q_idx = q_row_start + my_row_local_base + r;
        row_valid[r] = q_idx < params.q_len;
        if (row_valid[r]) {
            const device InputStorageT* q_ptr = q_base + q_idx * params.q_stride_m;
            q_reg[r] = FlashVec4T(
                (FlashTileT)q_ptr[lane],
                (FlashTileT)q_ptr[lane + 32u],
                (FlashTileT)q_ptr[lane + 64u],
                (FlashTileT)q_ptr[lane + 96u]
            );
        } else {
            q_reg[r] = FlashVec4T((FlashTileT)0.0f);
        }
    }

    uint tile_begin = 0;
    uint tile_end = 0;
    splitk_tile_range(params.kv_len, max((uint)1, params.split_k), split_id, tile_begin, tile_end);

    for (uint k_tile_idx = tile_begin; k_tile_idx < tile_end; ++k_tile_idx) {
        uint k_start = k_tile_idx * 32;
        uint stored_rows = min((uint)32, params.kv_len - k_start);

        uint linear_tid = tid.x;
        for (uint load_tid = linear_tid; load_tid < 256; load_tid += tptg.x) {
            load_tile_d128(k_base + k_start * params.stride_k_s, k_shared, params.stride_k_s, stored_rows, load_tid);
            load_tile_d128(v_base + k_start * params.stride_v_s, v_shared, params.stride_v_s, stored_rows, load_tid);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int r = 0; r < 4; ++r) {
            uint q_global = q_row_start + my_row_local_base + r;
            uint causal_limit = q_global + params.query_offset;
            if (!row_valid[r]) {
                continue;
            }

            float block_max = -1e30f;
            for (int k = 0; k < 32; ++k) {
                uint k_global = k_start + (uint)k;
                if (k_global >= params.kv_len) break;
                bool masked = (k_global > causal_limit);
                FlashTileT k0 = masked ? (FlashTileT)0.0f : k_shared[k * 128 + lane];
                FlashTileT k1 = masked ? (FlashTileT)0.0f : k_shared[k * 128 + lane + 32];
                FlashTileT k2 = masked ? (FlashTileT)0.0f : k_shared[k * 128 + lane + 64];
                FlashTileT k3 = masked ? (FlashTileT)0.0f : k_shared[k * 128 + lane + 96];
                float partial =
                    (float)q_reg[r][0] * (float)k0 +
                    (float)q_reg[r][1] * (float)k1 +
                    (float)q_reg[r][2] * (float)k2 +
                    (float)q_reg[r][3] * (float)k3;
                float score = simd_sum(partial) * params.scale;
                score = masked ? -1e30f : score;
                block_max = max(block_max, score);
            }

            float4 block_out = float4(0.0f);
            float block_sum = 0.0f;
            for (int k = 0; k < 32; ++k) {
                uint k_global = k_start + (uint)k;
                if (k_global >= params.kv_len) break;
                if (k_global > causal_limit) continue;

                FlashTileT k0 = k_shared[k * 128 + lane];
                FlashTileT k1 = k_shared[k * 128 + lane + 32];
                FlashTileT k2 = k_shared[k * 128 + lane + 64];
                FlashTileT k3 = k_shared[k * 128 + lane + 96];
                float partial =
                    (float)q_reg[r][0] * (float)k0 +
                    (float)q_reg[r][1] * (float)k1 +
                    (float)q_reg[r][2] * (float)k2 +
                    (float)q_reg[r][3] * (float)k3;
                float score = simd_sum(partial) * params.scale;

                float p_local = 0.0f;
                if (lane == 0) {
                    p_local = metal::fast::exp(score - block_max);
                    block_sum += p_local;
                }
                float p = simd_broadcast(p_local, 0);

                FlashTileT v0 = v_shared[k * 128 + lane];
                FlashTileT v1 = v_shared[k * 128 + lane + 32];
                FlashTileT v2 = v_shared[k * 128 + lane + 64];
                FlashTileT v3 = v_shared[k * 128 + lane + 96];
                block_out[0] += p * (float)v0;
                block_out[1] += p * (float)v1;
                block_out[2] += p * (float)v2;
                block_out[3] += p * (float)v3;
            }

            float alpha_local = 0.0f;
            float beta_local = 0.0f;
            if (lane == 0) {
                float m_prev = m[r];
                float l_prev = l[r];
                float m_new = max(m_prev, block_max);
                float alpha = metal::fast::exp(m_prev - m_new);
                float beta = metal::fast::exp(block_max - m_new);
                m[r] = m_new;
                l[r] = l_prev * alpha + block_sum * beta;
                alpha_local = alpha;
                beta_local = beta;
            }
            float alpha_b = simd_broadcast(alpha_local, 0);
            float beta_b = simd_broadcast(beta_local, 0);
            acc[r] = acc[r] * alpha_b + block_out * beta_b;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint q_tile_count = (params.q_len + TILE_M - 1) / TILE_M;
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx; // INDEX64_OK
    ulong split_linear = tile_linear * (ulong)max((uint)1, params.split_k) + (ulong)split_id; // INDEX64_OK
    ulong m_base = split_linear * (ulong)TILE_M; // INDEX64_OK
    ulong acc_base = m_base * (ulong)params.head_dim; // INDEX64_OK

    for (int r = 0; r < 4; ++r) {
        uint row_in_tile = my_row_local_base + (uint)r;
        uint q_global = q_row_start + row_in_tile;

        if (q_global >= params.q_len) {
            if (lane == 0) {
                partial_m[m_base + row_in_tile] = -1e30f;
                partial_l[m_base + row_in_tile] = 0.0f;
            }
            ulong row_base = acc_base + (ulong)row_in_tile * (ulong)params.head_dim; // INDEX64_OK
            partial_acc[row_base + lane] = 0.0f;
            partial_acc[row_base + lane + 32u] = 0.0f;
            partial_acc[row_base + lane + 64u] = 0.0f;
            partial_acc[row_base + lane + 96u] = 0.0f;
            continue;
        }

        if (lane == 0) {
            partial_m[m_base + row_in_tile] = m[r];
            partial_l[m_base + row_in_tile] = l[r];
        }
        ulong row_base = acc_base + (ulong)row_in_tile * (ulong)params.head_dim; // INDEX64_OK
        partial_acc[row_base + lane] = acc[r][0];
        partial_acc[row_base + lane + 32u] = acc[r][1];
        partial_acc[row_base + lane + 64u] = acc[r][2];
        partial_acc[row_base + lane + 96u] = acc[r][3];
    }
}

template<uint WARPS>
inline void flash_prefill_splitk_reduce_d64(
    const device float* partial_acc,
    const device float* partial_m,
    const device float* partial_l,
    device OutputStorageT* output,
    constant SdpaPrefillSplitKParams& params,
    uint3 gid,
    uint3 tid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id
) {
    constexpr uint TILE_M = WARPS * 4;

    uint tile_m_idx = gid.x;
    uint head_idx = gid.y;

    uint warp = simd_group_id;
    uint lane = simd_lane_id;
    if (warp >= WARPS) {
        return;
    }

    uint q_row_start = tile_m_idx * TILE_M;
    uint my_row_local_base = warp * 4;

    uint q_tile_count = (params.q_len + TILE_M - 1) / TILE_M;
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx; // INDEX64_OK
    ulong base_split = tile_linear * (ulong)max((uint)1, params.split_k); // INDEX64_OK

    for (int r = 0; r < 4; ++r) {
        uint row_in_tile = my_row_local_base + (uint)r;
        uint q_global = q_row_start + row_in_tile;
        if (q_global >= params.q_len) {
            continue;
        }

        float m_max_local = -1e30f;
        float inv_l_local = 0.0f;

        if (lane == 0) {
            for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile; // INDEX64_OK
                float ms = partial_m[m_idx];
                m_max_local = max(m_max_local, ms);
            }

            float l_total = 0.0f;
            for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile; // INDEX64_OK
                float ms = partial_m[m_idx];
                float ls = partial_l[m_idx];
                if (!(ls > 0.0f)) continue;
                l_total += ls * metal::fast::exp(ms - m_max_local);
            }
            inv_l_local = (isfinite(l_total) && abs(l_total) > 1e-6f) ? (1.0f / l_total) : 0.0f;
        }

        float m_max = simd_broadcast(m_max_local, 0);
        float inv_l = simd_broadcast(inv_l_local, 0);

        float2 acc_sum = float2(0.0f);
        for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
            float w_local = 0.0f;
            if (lane == 0) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile; // INDEX64_OK
                float ms = partial_m[m_idx];
                w_local = metal::fast::exp(ms - m_max);
            }
            float w = simd_broadcast(w_local, 0);

            ulong acc_row_base =
                ((base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile) * (ulong)params.head_dim; // INDEX64_OK
            float a0 = partial_acc[acc_row_base + lane];
            float a1 = partial_acc[acc_row_base + lane + 32u];
            acc_sum[0] += a0 * w;
            acc_sum[1] += a1 * w;
        }

        float2 res = acc_sum * inv_l;
        device OutputStorageT* out_ptr = output + q_global * params.out_stride_m;
        metallic_store_output2(out_ptr, lane, lane + 32u, res);
    }
}

template<uint WARPS>
inline void flash_prefill_splitk_reduce_d128(
    const device float* partial_acc,
    const device float* partial_m,
    const device float* partial_l,
    device OutputStorageT* output,
    constant SdpaPrefillSplitKParams& params,
    uint3 gid,
    uint3 tid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id
) {
    constexpr uint TILE_M = WARPS * 4;

    uint tile_m_idx = gid.x;
    uint head_idx = gid.y;

    uint warp = simd_group_id;
    uint lane = simd_lane_id;
    if (warp >= WARPS) {
        return;
    }

    uint q_row_start = tile_m_idx * TILE_M;
    uint my_row_local_base = warp * 4;

    uint q_tile_count = (params.q_len + TILE_M - 1) / TILE_M;
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx; // INDEX64_OK
    ulong base_split = tile_linear * (ulong)max((uint)1, params.split_k); // INDEX64_OK

    for (int r = 0; r < 4; ++r) {
        uint row_in_tile = my_row_local_base + (uint)r;
        uint q_global = q_row_start + row_in_tile;
        if (q_global >= params.q_len) {
            continue;
        }

        float m_max_local = -1e30f;
        float inv_l_local = 0.0f;

        if (lane == 0) {
            for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile; // INDEX64_OK
                float ms = partial_m[m_idx];
                m_max_local = max(m_max_local, ms);
            }

            float l_total = 0.0f;
            for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile; // INDEX64_OK
                float ms = partial_m[m_idx];
                float ls = partial_l[m_idx];
                if (!(ls > 0.0f)) continue;
                l_total += ls * metal::fast::exp(ms - m_max_local);
            }
            inv_l_local = (isfinite(l_total) && abs(l_total) > 1e-6f) ? (1.0f / l_total) : 0.0f;
        }

        float m_max = simd_broadcast(m_max_local, 0);
        float inv_l = simd_broadcast(inv_l_local, 0);

        float4 acc_sum = float4(0.0f);
        for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
            float w_local = 0.0f;
            if (lane == 0) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile; // INDEX64_OK
                float ms = partial_m[m_idx];
                w_local = metal::fast::exp(ms - m_max);
            }
            float w = simd_broadcast(w_local, 0);

            ulong acc_row_base =
                ((base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile) * (ulong)params.head_dim; // INDEX64_OK
            float a0 = partial_acc[acc_row_base + lane];
            float a1 = partial_acc[acc_row_base + lane + 32u];
            float a2 = partial_acc[acc_row_base + lane + 64u];
            float a3 = partial_acc[acc_row_base + lane + 96u];
            acc_sum[0] += a0 * w;
            acc_sum[1] += a1 * w;
            acc_sum[2] += a2 * w;
            acc_sum[3] += a3 * w;
        }

        float4 res = acc_sum * inv_l;
        device OutputStorageT* out_ptr = output + q_global * params.out_stride_m;
        metallic_store_output4(
            out_ptr,
            lane,
            lane + 32u,
            lane + 64u,
            lane + 96u,
            res
        );
    }
}

template<uint WARPS>
ALWAYS_INLINE void run_sdpa_prefill_splitk_part_stage(
    const device InputStorageT* q,
    const device InputStorageT* k,
    const device InputStorageT* v,
    device float* partial_acc,
    device float* partial_m,
    device float* partial_l,
    constant SdpaPrefillSplitKParams& params,
    uint3 gid,
    uint3 lid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup FlashTileT* k_shared,
    threadgroup FlashTileT* v_shared
) {
    uint head_idx = gid.y;
    uint kv_head_idx = head_idx / params.group_size;
    const uint batch_idx = 0;

    ulong q_offset = batch_idx * params.q_stride_b + head_idx * params.q_stride_h;
    const device InputStorageT* q_ptr = q + q_offset;

    ulong k_offset = batch_idx * params.k_stride_b + kv_head_idx * params.k_stride_h;
    const device InputStorageT* k_ptr = k + k_offset;

    ulong v_offset = batch_idx * params.v_stride_b + kv_head_idx * params.v_stride_h;
    const device InputStorageT* v_ptr = v + v_offset;

    if (params.head_dim == 64) {
        flash_prefill_splitk_part_d64<WARPS>(
            q_ptr, k_ptr, v_ptr, partial_acc, partial_m, partial_l, params, k_shared, v_shared, gid, lid, tptg, simd_lane_id, simd_group_id
        );
    } else if (params.head_dim == 128) {
        flash_prefill_splitk_part_d128<WARPS>(
            q_ptr, k_ptr, v_ptr, partial_acc, partial_m, partial_l, params, k_shared, v_shared, gid, lid, tptg, simd_lane_id, simd_group_id
        );
    }
}

template<uint WARPS>
ALWAYS_INLINE void run_sdpa_prefill_splitk_reduce_stage(
    const device float* partial_acc,
    const device float* partial_m,
    const device float* partial_l,
    device OutputStorageT* output,
    constant SdpaPrefillSplitKParams& params,
    uint3 gid,
    uint3 lid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id
) {
    uint head_idx = gid.y;
    const uint batch_idx = 0;

    ulong out_offset = batch_idx * params.out_stride_b + head_idx * params.out_stride_h;
    device OutputStorageT* output_ptr = output + out_offset;

    if (params.head_dim == 64) {
        flash_prefill_splitk_reduce_d64<WARPS>(partial_acc, partial_m, partial_l, output_ptr, params, gid, lid, tptg, simd_lane_id, simd_group_id);
    } else if (params.head_dim == 128) {
        flash_prefill_splitk_reduce_d128<WARPS>(partial_acc, partial_m, partial_l, output_ptr, params, gid, lid, tptg, simd_lane_id, simd_group_id);
    }
}
