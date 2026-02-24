// NOTE: SdpaPrefillParams / SdpaPrefillSplitKParams are injected from Rust
// (`#[derive(MetalStruct)]`) through stage `struct_defs` for a single source of truth.


// Tiled prefill SDPA kernel for head_dim=64.
//
// Expected layouts (all strides are in elements, not bytes):
// - Q: per-head base pointer `q_base` points at the start of head `h` for batch `b`.
//      Rows are addressed as `q_base + q_row * q_stride_m`.
// - K/V: per-(kv)head base pointers `k_base`/`v_base` point at the start of the KV head for batch `b`.
//        Rows are addressed as `k_base + k_row * stride_k_s` / `v_base + k_row * stride_v_s`.
// - Output: per-head base pointer `output` points at the start of head `h` for batch `b`.
//           Rows are addressed as `output + q_row * out_stride_m`.
//
// Dispatch:
// - One threadgroup (256 threads = 8 simdgroups) per (TileM, head, batch).
// - `gid` is `[[threadgroup_position_in_grid]]`, so `gid.x` is the TileM index.
// - Each simdgroup processes 4 query rows; each lane holds 2 columns (lane and lane+32).
template<uint WARPS>
inline void flash_prefill_tiled_d64(
    const device InputStorageT* q_base,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device OutputStorageT* output,
    constant SdpaPrefillParams& params,
    threadgroup FlashTileT* k_shared,
    threadgroup FlashTileT* v_shared,
    uint3 gid, // threadgroup_position_in_grid
    uint3 tid, // thread_position_in_threadgroup
    uint3 tptg, // threads_per_threadgroup
    uint simd_lane_id,
    uint simd_group_id
) {
    // Grid: (ceil_div(M, 32), n_heads, batch)
    uint tile_m_idx = gid.x;
    
    // Each simdgroup processes 4 query rows.
    // Threadgroup size is WARPS simdgroups => TileM = WARPS * 4.
    constexpr uint TILE_M = WARPS * 4;
    uint q_row_start = tile_m_idx * TILE_M;
    
    uint warp = simd_group_id;
    uint lane = simd_lane_id;

    if (warp >= WARPS) {
        return;
    }
    
    uint my_row_local_base = warp * 4;
    
    // Accumulators for 4 rows, 2 columns per lane (lane, lane+32).
    float2 acc[4]; 
    for (int r=0; r<4; ++r) acc[r] = float2(0.0f);
    
    float m[4]; 
    float l[4];
    for (int r=0; r<4; ++r) { m[r] = -1e30f; l[r] = 0.0f; }
    
    // Cache Q in registers for this lane for each row.
    FlashVec2T q_reg[4]; 
    
    bool row_valid[4];
    
    // Load up to 4 query rows for this simdgroup.
    for (int r=0; r<4; ++r) {
        uint q_idx = q_row_start + my_row_local_base + r;
        
        row_valid[r] = q_idx < params.q_len;
        
        if (row_valid[r]) {
            const device InputStorageT* q_ptr = q_base + q_idx * params.q_stride_m;

            // Each lane loads its two columns: `lane` and `lane+32`.
            q_reg[r] = FlashVec2T(
                (FlashTileT)q_ptr[lane],
                (FlashTileT)q_ptr[lane + 32u]
            );
        } else {
            q_reg[r] = FlashVec2T((FlashTileT)0.0f, (FlashTileT)0.0f);
        }
    }
    
    uint kv_len = params.kv_len;
    
    for (uint k_tile_idx = 0; k_tile_idx * 32 < kv_len; ++k_tile_idx) {
        // 1) Cooperative load K/V tile into threadgroup memory.
        uint k_start = k_tile_idx * 32;
        uint stored_rows = min((uint)32, kv_len - k_start);
        
        // Tile loads are expressed as 256 independent vector-load "slots".
        // For smaller threadgroups (e.g. WARPS=4 => 128 threads), each thread covers multiple slots.
        uint linear_tid = tid.x;
        for (uint load_tid = linear_tid; load_tid < 256; load_tid += tptg.x) {
            load_tile(k_base + k_start * params.stride_k_s, k_shared, params.stride_k_s, stored_rows, load_tid);
            load_tile(v_base + k_start * params.stride_v_s, v_shared, params.stride_v_s, stored_rows, load_tid);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // 2) Compute attention for the 4 rows owned by this simdgroup against this K/V tile.
        for (int r=0; r<4; ++r) {
            uint q_global = q_row_start + my_row_local_base + r;
            // Causal masking:
            // - `query_offset` is the absolute position of query row 0 in the KV stream.
            // - absolute query position for this row is `query_offset + q_global`.
            // - allow keys `k_global <= abs_query`.
            uint causal_limit = q_global + params.query_offset;
            
            if (!row_valid[r]) {
                continue;
            }

            // FA1-style blockwise online softmax update:
            // 1) Find block max across keys in the tile.
            float block_max = -1e30f;
            for (int k=0; k<32; ++k) {
                uint k_global = k_start + k;
                if (k_global >= kv_len) break;

                bool masked = (k_global > causal_limit);
                FlashTileT k_val0 = masked ? (FlashTileT)0.0f : k_shared[k*64 + lane];
                FlashTileT k_val1 = masked ? (FlashTileT)0.0f : k_shared[k*64 + lane + 32];
                float partial = (float)q_reg[r][0] * (float)k_val0 + (float)q_reg[r][1] * (float)k_val1;
                float score = simd_sum(partial) * params.scale;
                score = masked ? -1e30f : score;
                block_max = max(block_max, score);
            }

            // 2) Compute block sumexp and block output for this tile.
            float2 block_out = float2(0.0f);
            float block_sum = 0.0f;
            for (int k=0; k<32; ++k) {
                uint k_global = k_start + k;
                if (k_global >= kv_len) break;
                if (k_global > causal_limit) continue;

                FlashTileT k_val0 = k_shared[k*64 + lane];
                FlashTileT k_val1 = k_shared[k*64 + lane + 32];
                float partial = (float)q_reg[r][0] * (float)k_val0 + (float)q_reg[r][1] * (float)k_val1;
                float score = simd_sum(partial) * params.scale;

                // `score` is identical across the simdgroup. Compute exp once (lane 0) and broadcast.
                float p_local = 0.0f;
                if (lane == 0) {
                    p_local = metal::fast::exp(score - block_max);
                    block_sum += p_local;
                }
                float p = simd_broadcast(p_local, 0);

                FlashTileT v_val0 = v_shared[k*64 + lane];
                FlashTileT v_val1 = v_shared[k*64 + lane + 32];
                block_out[0] += p * (float)v_val0;
                block_out[1] += p * (float)v_val1;
            }

            // 3) Update running (m, l, acc) once per tile.
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
    
    // 3) Normalize and store.
    for (int r=0; r<4; ++r) {
        if (!row_valid[r]) continue;

        float inv_l_local = 0.0f;
        if (lane == 0) {
            float denom = l[r];
            inv_l_local = (isfinite(denom) && abs(denom) > 1e-6f) ? (1.0f / denom) : 0.0f;
        }
        float inv_l = simd_broadcast(inv_l_local, 0);
        float2 res = acc[r] * inv_l;
        
        uint q_global = q_row_start + my_row_local_base + r;
        
        // Store this lane's two columns.
        device OutputStorageT* out_ptr = output + q_global * params.out_stride_m;
        metallic_store_output2(out_ptr, lane, lane + 32u, res);
    }
}

// Tiled prefill SDPA kernel for head_dim=128.
template<uint WARPS>
inline void flash_prefill_tiled_d128(
    const device InputStorageT* q_base,
    const device InputStorageT* k_base,
    const device InputStorageT* v_base,
    device OutputStorageT* output,
    constant SdpaPrefillParams& params,
    threadgroup FlashTileT* k_shared,
    threadgroup FlashTileT* v_shared,
    uint3 gid,
    uint3 tid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id
) {
    uint tile_m_idx = gid.x;
    constexpr uint TILE_M = WARPS * 4;
    uint q_row_start = tile_m_idx * TILE_M;

    uint warp = simd_group_id;
    uint lane = simd_lane_id;

    if (warp >= WARPS) {
        return;
    }
    uint my_row_local_base = warp * 4;

    float4 acc[4];
    for (int r = 0; r < 4; ++r) {
        acc[r] = float4(0.0f);
    }

    float m[4];
    float l[4];
    for (int r = 0; r < 4; ++r) {
        m[r] = -1e30f;
        l[r] = 0.0f;
    }

    FlashVec4T q_reg[4];
    bool row_valid[4];
    for (int r = 0; r < 4; ++r) {
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

    uint kv_len = params.kv_len;
    for (uint k_tile_idx = 0; k_tile_idx * 32 < kv_len; ++k_tile_idx) {
        uint k_start = k_tile_idx * 32;
        uint stored_rows = min((uint)32, kv_len - k_start);

        uint linear_tid = tid.x;
        for (uint load_tid = linear_tid; load_tid < 256; load_tid += tptg.x) {
            load_tile_d128(k_base + k_start * params.stride_k_s, k_shared, params.stride_k_s, stored_rows, load_tid);
            load_tile_d128(v_base + k_start * params.stride_v_s, v_shared, params.stride_v_s, stored_rows, load_tid);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int r = 0; r < 4; ++r) {
            if (!row_valid[r]) {
                continue;
            }

            uint q_global = q_row_start + my_row_local_base + r;
            uint causal_limit = q_global + params.query_offset;

            float block_max = -1e30f;
            for (int k = 0; k < 32; ++k) {
                uint k_global = k_start + k;
                if (k_global >= kv_len) break;

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
                uint k_global = k_start + k;
                if (k_global >= kv_len) break;
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

    for (int r = 0; r < 4; ++r) {
        if (!row_valid[r]) continue;

        float inv_l_local = 0.0f;
        if (lane == 0) {
            float denom = l[r];
            inv_l_local = (isfinite(denom) && abs(denom) > 1e-6f) ? (1.0f / denom) : 0.0f;
        }
        float inv_l = simd_broadcast(inv_l_local, 0);
        float4 res = acc[r] * inv_l;

        uint q_global = q_row_start + my_row_local_base + r;
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
ALWAYS_INLINE void run_sdpa_prefill_stage(
    const device InputStorageT* q,
    const device InputStorageT* k,
    const device InputStorageT* v,
    device OutputStorageT* output,
    constant SdpaPrefillParams& params,
    uint3 gid,
    uint3 lid,
    uint3 tptg,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup FlashTileT* k_shared,
    threadgroup FlashTileT* v_shared
) {
    uint head_idx = gid.y;
    uint batch_idx = gid.z;
    uint kv_head_idx = head_idx / params.group_size;

    ulong q_offset = batch_idx * params.q_stride_b + head_idx * params.q_stride_h;
    const device InputStorageT* q_ptr = q + q_offset;

    ulong k_offset = batch_idx * params.k_stride_b + kv_head_idx * params.k_stride_h;
    const device InputStorageT* k_ptr = k + k_offset;

    ulong v_offset = batch_idx * params.v_stride_b + kv_head_idx * params.v_stride_h;
    const device InputStorageT* v_ptr = v + v_offset;

    ulong out_offset = batch_idx * params.out_stride_b + head_idx * params.out_stride_h;
    device OutputStorageT* output_ptr = output + out_offset;

    if (params.head_dim == 64) {
        flash_prefill_tiled_d64<WARPS>(q_ptr, k_ptr, v_ptr, output_ptr, params, k_shared, v_shared, gid, lid, tptg, simd_lane_id, simd_group_id);
    } else if (params.head_dim == 128) {
        flash_prefill_tiled_d128<WARPS>(q_ptr, k_ptr, v_ptr, output_ptr, params, k_shared, v_shared, gid, lid, tptg, simd_lane_id, simd_group_id);
    }
}

#define SDPA_PREFILL_DECLARE_SHARED(NAME_K, NAME_V) \
    threadgroup FlashTileT NAME_K[32 * 128];              \
    threadgroup FlashTileT NAME_V[32 * 128]
