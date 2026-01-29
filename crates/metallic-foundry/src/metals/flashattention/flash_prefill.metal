#ifndef METALLIC_FLASH_PREFILL_METAL
#define METALLIC_FLASH_PREFILL_METAL

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

// Cooperative tile load for a [TileN=32, D=64] block.
// - `src` must already point at the first element of the tile (row 0, col 0).
// - `row_stride` is in elements (half), not bytes.
// - `limit` is the number of valid rows in this tile (<= 32); remaining rows are zero-filled.
// - `tid` is the linear thread index in the threadgroup [0, 255].
inline void load_tile(
    const device half* src,
    threadgroup half* dst,
    uint row_stride,
    uint limit,
    uint tid
) {
    // Each thread loads 8 consecutive half elements (16 bytes).
    // 256 threads * 8 half = 2048 half = 32 * 64.
    uint row_in_tile = tid / 8;
    uint col_in_tile = (tid % 8) * 8;
    
    if (row_in_tile < 32) {
        bool active = row_in_tile < limit;
        
        const device ulong* src_u = (const device ulong*)(src + row_in_tile * row_stride + col_in_tile);
        threadgroup ulong* dst_u = (threadgroup ulong*)(dst + row_in_tile * 64 + col_in_tile);
        if (active) {
            // Copy 16 bytes via 2x 64-bit loads/stores.
            // `col_in_tile` is a multiple of 8 half elements, so this is 16-byte aligned.
            dst_u[0] = src_u[0];
            dst_u[1] = src_u[1];
        } else {
            dst_u[0] = 0;
            dst_u[1] = 0;
        }
    }
}

// Cooperative tile load for a [TileN=32, D=128] block.
// 256 threads * 16 half = 4096 half = 32 * 128.
inline void load_tile_d128(
    const device half* src,
    threadgroup half* dst,
    uint row_stride,
    uint limit,
    uint tid
) {
    uint row_in_tile = tid / 8;
    uint col_in_tile = (tid % 8) * 16;

    if (row_in_tile < 32) {
        bool active = row_in_tile < limit;

        const device ulong* src_u = (const device ulong*)(src + row_in_tile * row_stride + col_in_tile);
        threadgroup ulong* dst_u = (threadgroup ulong*)(dst + row_in_tile * 128 + col_in_tile);
        if (active) {
            // Copy 32 bytes via 4x 64-bit loads/stores.
            dst_u[0] = src_u[0];
            dst_u[1] = src_u[1];
            dst_u[2] = src_u[2];
            dst_u[3] = src_u[3];
        } else {
            dst_u[0] = 0;
            dst_u[1] = 0;
            dst_u[2] = 0;
            dst_u[3] = 0;
        }
    }
}

// NOTE: This struct must match the Rust `SdpaPrefillParams` layout exactly.
#ifndef METALLIC_SDPA_PREFILL_PARAMS_DEFINED
#define METALLIC_SDPA_PREFILL_PARAMS_DEFINED
struct SdpaPrefillParams {
    uint kv_len;
    uint head_dim;
    float scale;
    uint stride_k_s;
    uint stride_v_s;
    uint query_offset;
    
    // Strides
    uint q_stride_b;
    uint q_stride_h;
    uint k_stride_b;
    uint k_stride_h;
    uint v_stride_b;
    uint v_stride_h;
    uint out_stride_b;
    uint out_stride_h;
    
    uint q_stride_m;
    uint out_stride_m;
    uint group_size;
    uint q_len; // Query sequence length (M)
};
#endif

// Split-K prefill params (used only by the Split-K kernels below).
// NOTE: This struct must match the Rust `SdpaPrefillSplitKParams` layout exactly.
#ifndef METALLIC_SDPA_PREFILL_SPLITK_PARAMS_DEFINED
#define METALLIC_SDPA_PREFILL_SPLITK_PARAMS_DEFINED
struct SdpaPrefillSplitKParams {
    uint kv_len;
    uint head_dim;
    float scale;
    uint stride_k_s;
    uint stride_v_s;
    uint query_offset;

    uint q_stride_b;
    uint q_stride_h;
    uint k_stride_b;
    uint k_stride_h;
    uint v_stride_b;
    uint v_stride_h;
    uint out_stride_b;
    uint out_stride_h;

    uint q_stride_m;
    uint out_stride_m;
    uint group_size;
    uint q_len;

    uint n_heads;
    uint split_k;
};
#endif


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
    const device half* q_base,
    const device half* k_base,
    const device half* v_base,
    device half* output,
    constant SdpaPrefillParams& params,
    threadgroup half* k_shared,
    threadgroup half* v_shared,
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
    half2 q_reg[4]; 
    
    bool row_valid[4];
    
    // Load up to 4 query rows for this simdgroup.
    for (int r=0; r<4; ++r) {
        uint q_idx = q_row_start + my_row_local_base + r;
        
        row_valid[r] = q_idx < params.q_len;
        
        if (row_valid[r]) {
            const device half* q_ptr = q_base + q_idx * params.q_stride_m; 
            
            // Each lane loads its two columns: `lane` and `lane+32`.
            half v0 = q_ptr[lane];
            half v1 = q_ptr[lane+32];
            q_reg[r] = half2(v0, v1);
        } else {
            q_reg[r] = half2(0.0h, 0.0h);
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
                half k_val0 = masked ? half(0.0h) : k_shared[k*64 + lane];
                half k_val1 = masked ? half(0.0h) : k_shared[k*64 + lane + 32];
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

                half k_val0 = k_shared[k*64 + lane];
                half k_val1 = k_shared[k*64 + lane + 32];
                float partial = (float)q_reg[r][0] * (float)k_val0 + (float)q_reg[r][1] * (float)k_val1;
                float score = simd_sum(partial) * params.scale;

                // `score` is identical across the simdgroup. Compute exp once (lane 0) and broadcast.
                float p_local = 0.0f;
                if (lane == 0) {
                    p_local = metal::fast::exp(score - block_max);
                    block_sum += p_local;
                }
                float p = simd_broadcast(p_local, 0);

                half v_val0 = v_shared[k*64 + lane];
                half v_val1 = v_shared[k*64 + lane + 32];
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
        device half* out_ptr = output + q_global * params.out_stride_m;
        out_ptr[lane] = (half)res[0];
        out_ptr[lane+32] = (half)res[1];
    }
}

// Tiled prefill SDPA kernel for head_dim=128.
template<uint WARPS>
inline void flash_prefill_tiled_d128(
    const device half* q_base,
    const device half* k_base,
    const device half* v_base,
    device half* output,
    constant SdpaPrefillParams& params,
    threadgroup half* k_shared,
    threadgroup half* v_shared,
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

    half4 q_reg[4];
    bool row_valid[4];
    for (int r = 0; r < 4; ++r) {
        uint q_idx = q_row_start + my_row_local_base + r;
        row_valid[r] = q_idx < params.q_len;
        if (row_valid[r]) {
            const device half* q_ptr = q_base + q_idx * params.q_stride_m;
            half v0 = q_ptr[lane];
            half v1 = q_ptr[lane + 32];
            half v2 = q_ptr[lane + 64];
            half v3 = q_ptr[lane + 96];
            q_reg[r] = half4(v0, v1, v2, v3);
        } else {
            q_reg[r] = half4(0.0h);
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
                half k0 = masked ? half(0.0h) : k_shared[k * 128 + lane];
                half k1 = masked ? half(0.0h) : k_shared[k * 128 + lane + 32];
                half k2 = masked ? half(0.0h) : k_shared[k * 128 + lane + 64];
                half k3 = masked ? half(0.0h) : k_shared[k * 128 + lane + 96];

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

                half k0 = k_shared[k * 128 + lane];
                half k1 = k_shared[k * 128 + lane + 32];
                half k2 = k_shared[k * 128 + lane + 64];
                half k3 = k_shared[k * 128 + lane + 96];

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

                half v0 = v_shared[k * 128 + lane];
                half v1 = v_shared[k * 128 + lane + 32];
                half v2 = v_shared[k * 128 + lane + 64];
                half v3 = v_shared[k * 128 + lane + 96];
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
        device half* out_ptr = output + q_global * params.out_stride_m;
        out_ptr[lane] = (half)res[0];
        out_ptr[lane + 32] = (half)res[1];
        out_ptr[lane + 64] = (half)res[2];
        out_ptr[lane + 96] = (half)res[3];
    }
}

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
    const device half* q_base,
    const device half* k_base,
    const device half* v_base,
    device float* partial_acc,
    device float* partial_m,
    device float* partial_l,
    constant SdpaPrefillSplitKParams& params,
    threadgroup half* k_shared,
    threadgroup half* v_shared,
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
    half2 q_reg[4];
    bool row_valid[4];
    for (int r = 0; r < 4; ++r) {
        acc[r] = float2(0.0f);
        m[r] = -1e30f;
        l[r] = 0.0f;

        uint q_idx = q_row_start + my_row_local_base + r;
        row_valid[r] = q_idx < params.q_len;
        if (row_valid[r]) {
            const device half* q_ptr = q_base + q_idx * params.q_stride_m;
            q_reg[r] = half2(q_ptr[lane], q_ptr[lane + 32]);
        } else {
            q_reg[r] = half2(0.0h, 0.0h);
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
                half k0 = masked ? half(0.0h) : k_shared[k * 64 + lane];
                half k1 = masked ? half(0.0h) : k_shared[k * 64 + lane + 32];
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

                half k0 = k_shared[k * 64 + lane];
                half k1 = k_shared[k * 64 + lane + 32];
                float partial = (float)q_reg[r][0] * (float)k0 + (float)q_reg[r][1] * (float)k1;
                float score = simd_sum(partial) * params.scale;

                float p_local = 0.0f;
                if (lane == 0) {
                    p_local = metal::fast::exp(score - block_max);
                    block_sum += p_local;
                }
                float p = simd_broadcast(p_local, 0);

                half v0 = v_shared[k * 64 + lane];
                half v1 = v_shared[k * 64 + lane + 32];
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
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx;
    ulong split_linear = tile_linear * (ulong)max((uint)1, params.split_k) + (ulong)split_id;
    ulong m_base = split_linear * (ulong)TILE_M;
    ulong acc_base = m_base * (ulong)params.head_dim;

    for (int r = 0; r < 4; ++r) {
        uint row_in_tile = my_row_local_base + (uint)r;
        uint q_global = q_row_start + row_in_tile;

        if (q_global >= params.q_len) {
            if (lane == 0) {
                partial_m[m_base + row_in_tile] = -1e30f;
                partial_l[m_base + row_in_tile] = 0.0f;
            }
            partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + (ulong)lane] = 0.0f;
            partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + (ulong)lane + 32ul] = 0.0f;
            continue;
        }

        if (lane == 0) {
            partial_m[m_base + row_in_tile] = m[r];
            partial_l[m_base + row_in_tile] = l[r];
        }
        partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + (ulong)lane] = acc[r][0];
        partial_acc[acc_base + (ulong)row_in_tile * (ulong)params.head_dim + (ulong)lane + 32ul] = acc[r][1];
    }
}

template<uint WARPS>
inline void flash_prefill_splitk_part_d128(
    const device half* q_base,
    const device half* k_base,
    const device half* v_base,
    device float* partial_acc,
    device float* partial_m,
    device float* partial_l,
    constant SdpaPrefillSplitKParams& params,
    threadgroup half* k_shared,
    threadgroup half* v_shared,
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
    half4 q_reg[4];
    bool row_valid[4];
    for (int r = 0; r < 4; ++r) {
        acc[r] = float4(0.0f);
        m[r] = -1e30f;
        l[r] = 0.0f;

        uint q_idx = q_row_start + my_row_local_base + r;
        row_valid[r] = q_idx < params.q_len;
        if (row_valid[r]) {
            const device half* q_ptr = q_base + q_idx * params.q_stride_m;
            q_reg[r] = half4(q_ptr[lane], q_ptr[lane + 32], q_ptr[lane + 64], q_ptr[lane + 96]);
        } else {
            q_reg[r] = half4(0.0h);
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
                half k0 = masked ? half(0.0h) : k_shared[k * 128 + lane];
                half k1 = masked ? half(0.0h) : k_shared[k * 128 + lane + 32];
                half k2 = masked ? half(0.0h) : k_shared[k * 128 + lane + 64];
                half k3 = masked ? half(0.0h) : k_shared[k * 128 + lane + 96];
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

                half k0 = k_shared[k * 128 + lane];
                half k1 = k_shared[k * 128 + lane + 32];
                half k2 = k_shared[k * 128 + lane + 64];
                half k3 = k_shared[k * 128 + lane + 96];
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

                half v0 = v_shared[k * 128 + lane];
                half v1 = v_shared[k * 128 + lane + 32];
                half v2 = v_shared[k * 128 + lane + 64];
                half v3 = v_shared[k * 128 + lane + 96];
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
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx;
    ulong split_linear = tile_linear * (ulong)max((uint)1, params.split_k) + (ulong)split_id;
    ulong m_base = split_linear * (ulong)TILE_M;
    ulong acc_base = m_base * (ulong)params.head_dim;

    for (int r = 0; r < 4; ++r) {
        uint row_in_tile = my_row_local_base + (uint)r;
        uint q_global = q_row_start + row_in_tile;

        if (q_global >= params.q_len) {
            if (lane == 0) {
                partial_m[m_base + row_in_tile] = -1e30f;
                partial_l[m_base + row_in_tile] = 0.0f;
            }
            ulong row_base = acc_base + (ulong)row_in_tile * (ulong)params.head_dim;
            partial_acc[row_base + (ulong)lane] = 0.0f;
            partial_acc[row_base + (ulong)lane + 32ul] = 0.0f;
            partial_acc[row_base + (ulong)lane + 64ul] = 0.0f;
            partial_acc[row_base + (ulong)lane + 96ul] = 0.0f;
            continue;
        }

        if (lane == 0) {
            partial_m[m_base + row_in_tile] = m[r];
            partial_l[m_base + row_in_tile] = l[r];
        }
        ulong row_base = acc_base + (ulong)row_in_tile * (ulong)params.head_dim;
        partial_acc[row_base + (ulong)lane] = acc[r][0];
        partial_acc[row_base + (ulong)lane + 32ul] = acc[r][1];
        partial_acc[row_base + (ulong)lane + 64ul] = acc[r][2];
        partial_acc[row_base + (ulong)lane + 96ul] = acc[r][3];
    }
}

template<uint WARPS>
inline void flash_prefill_splitk_reduce_d64(
    const device float* partial_acc,
    const device float* partial_m,
    const device float* partial_l,
    device half* output,
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
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx;
    ulong base_split = tile_linear * (ulong)max((uint)1, params.split_k);

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
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile;
                float ms = partial_m[m_idx];
                m_max_local = max(m_max_local, ms);
            }

            float l_total = 0.0f;
            for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile;
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
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile;
                float ms = partial_m[m_idx];
                w_local = metal::fast::exp(ms - m_max);
            }
            float w = simd_broadcast(w_local, 0);

            ulong acc_row_base =
                ((base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile) * (ulong)params.head_dim;
            float a0 = partial_acc[acc_row_base + (ulong)lane];
            float a1 = partial_acc[acc_row_base + (ulong)lane + 32ul];
            acc_sum[0] += a0 * w;
            acc_sum[1] += a1 * w;
        }

        float2 res = acc_sum * inv_l;
        device half* out_ptr = output + q_global * params.out_stride_m;
        out_ptr[lane] = (half)res[0];
        out_ptr[lane + 32] = (half)res[1];
    }
}

template<uint WARPS>
inline void flash_prefill_splitk_reduce_d128(
    const device float* partial_acc,
    const device float* partial_m,
    const device float* partial_l,
    device half* output,
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
    ulong tile_linear = (ulong)head_idx * (ulong)q_tile_count + (ulong)tile_m_idx;
    ulong base_split = tile_linear * (ulong)max((uint)1, params.split_k);

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
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile;
                float ms = partial_m[m_idx];
                m_max_local = max(m_max_local, ms);
            }

            float l_total = 0.0f;
            for (uint s = 0; s < max((uint)1, params.split_k); ++s) {
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile;
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
                ulong m_idx = (base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile;
                float ms = partial_m[m_idx];
                w_local = metal::fast::exp(ms - m_max);
            }
            float w = simd_broadcast(w_local, 0);

            ulong acc_row_base =
                ((base_split + (ulong)s) * (ulong)TILE_M + (ulong)row_in_tile) * (ulong)params.head_dim;
            float a0 = partial_acc[acc_row_base + (ulong)lane];
            float a1 = partial_acc[acc_row_base + (ulong)lane + 32ul];
            float a2 = partial_acc[acc_row_base + (ulong)lane + 64ul];
            float a3 = partial_acc[acc_row_base + (ulong)lane + 96ul];
            acc_sum[0] += a0 * w;
            acc_sum[1] += a1 * w;
            acc_sum[2] += a2 * w;
            acc_sum[3] += a3 * w;
        }

        float4 res = acc_sum * inv_l;
        device half* out_ptr = output + q_global * params.out_stride_m;
        out_ptr[lane] = (half)res[0];
        out_ptr[lane + 32] = (half)res[1];
        out_ptr[lane + 64] = (half)res[2];
        out_ptr[lane + 96] = (half)res[3];
    }
}

#endif
