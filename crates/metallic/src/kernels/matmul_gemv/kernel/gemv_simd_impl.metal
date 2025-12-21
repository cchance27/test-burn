#pragma once
#include <metal_stdlib>
#include "common_defs.metal"
using namespace metal;

constant float GEMV_RMSNORM_EPS = 1e-6f;

ALWAYS_INLINE float gemv_compute_inv_rms(
    const device half *vector_x,
    const uint K,
    const uint lane_id,
    const uint warp_id,
    threadgroup float *tg_inv_rms,
    float epsilon
) {
    if (K == 0u) {
        if (warp_id == 0u && lane_id == 0u) {
            tg_inv_rms[0] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return tg_inv_rms[0];
    }

    float sum = 0.0f;
    if (warp_id == 0u) {
        uint k = lane_id * 4u;
        const uint stride = 32u * 4u;
        for (; k + 4u <= K; k += stride) {
            half4 hv = *(const device half4 *)(vector_x + k);
            float4 fv = float4(hv);
            sum += dot(fv, fv);
        }
        for (; k < K; k += 32u) {
            float v = (float)vector_x[k];
            sum += v * v;
        }
        sum = simd_sum(sum);
        if (lane_id == 0u) {
            tg_inv_rms[0] = rsqrt(sum / (float)K + epsilon);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tg_inv_rms[0];
}

ALWAYS_INLINE float gemv_compute_inv_rms(
    const device half *vector_x,
    const uint K,
    const uint lane_id,
    const uint warp_id,
    threadgroup float *tg_inv_rms
) {
    return gemv_compute_inv_rms(vector_x, K, lane_id, warp_id, tg_inv_rms, GEMV_RMSNORM_EPS);
}


// =================================================================================================
// Generic SIMD GEMV Template
// =================================================================================================

// Policy Interface Concept:
// struct Policy {
//     struct Params { ... };
//     static constant uint FAST_K_CHUNK_SIZE;
//     static constant uint SAFE_K_CHUNK_SIZE;
//
//     template<uint HEADS>
//     void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS]);
//
//     void load_x_fast(const device half* x, uint k_base);
//     void load_x_safe(const device half* x, uint k_base, uint K);
//     float compute_dot(uint head_idx, bool fast_mode);
//     void advance_pointers(uint k_step);
// };

// =================================================================================================
// Epilogue Policies
// =================================================================================================

struct DefaultEpilogue {
    template<uint HEADS>
    static void apply(
        float acc[HEADS],
        uint lane_id,
        uint logical_col,
        const uint N[HEADS],
        const device half *bias[HEADS],
        const uint has_bias_flags[HEADS],
        const float alpha,
        const float beta,
        const device half *residual,
        device half *result_y[HEADS]
    ) {
        for (uint h = 0; h < HEADS; ++h) {
            // Mask out inactive heads/columns
            // Note: logical_col < N[h] check should be done here or passed in.
            if (logical_col >= N[h]) continue;

            float val = acc[h];
            
            // Full warp reduction
            val += simd_shuffle_xor(val, 16u);
            val += simd_shuffle_xor(val, 8u);
            val += simd_shuffle_xor(val, 4u);
            val += simd_shuffle_xor(val, 2u);
            val += simd_shuffle_xor(val, 1u);
            
            if (lane_id == 0) {
                float val_acc = val;
                if (has_bias_flags[h] && bias[h]) {
                    val_acc += (float)bias[h][logical_col];
                }

                // Standard Gemv Epilogue (Alpha/Beta/Residual)
                // Note: Generic template used alpha/beta only for standard gemv. 
                // fused/multi-head usually ignores alpha/beta or sets 1/0.
                // We'll use the helper.
                
                // Limitation: gemv_epilogue_value helper takes `residual_ptr`.
                // In multi-head, we usually don't support residual per head unless passed as array.
                // The template passed `residual` as single ptr. 
                // We'll assume residual applies to head 0 or we pass nullptr for >1 heads (as handled in original code).
                
                const device half* res_ptr = (HEADS == 1) ? residual : (const device half*)nullptr;
                
                const float val_computed = gemv_epilogue_value<false>(
                    val_acc,
                    (const device half*)nullptr, // Bias already added manually above
                    res_ptr,
                    alpha,
                    beta,
                    logical_col
                );
                result_y[h][logical_col] = (half)val_computed;
            }
        }
    }
};

// =================================================================================================
// Generic SIMD GEMV Template
// =================================================================================================

template <typename Policy, uint HEADS, uint COLS_PER_TG, bool HasBias, typename EpiloguePolicy = DefaultEpilogue>
void run_simd_gemv_template(
    typename Policy::Params params,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    GemvParams gemv_params
) {
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    
    // Batch Handling
    const uint batch_idx = gid.z;
    if (batch_idx >= gemv_params.batch) return;

    // Apply Batch Strides to Inputs/Outputs
    const device half *curr_x = vector_x + (ulong)batch_idx * gemv_params.stride_x;
    
    // Apply Batch Strides to Result Y (Array of pointers)
    // Note: We need local copy of pointers to offset them
    device half *curr_y[HEADS];
    for(uint h=0; h<HEADS; ++h) {
        curr_y[h] = result_y[h] + (ulong)batch_idx * gemv_params.stride_y;
    }

    // Each Warp processes 1 Logical Output Column.
    const uint logical_col = gid.x * COLS_PER_TG + warp_id;
    
    bool head_active[HEADS];
    for (uint h = 0; h < HEADS; ++h) {
        head_active[h] = (logical_col < N[h]);
    }

    // Initialize Policy
    Policy policy;
    policy.template init<HEADS>(params, gid, lid, logical_col, K, N, gemv_params, batch_idx);
    
    // Accumulators
    float acc[HEADS];
    for (uint h = 0; h < HEADS; ++h) acc[h] = 0.0f;

    uint k_base = 0;

    // 1. Fast Path Loop (No Bounds Checks)
    while (k_base + Policy::FAST_K_CHUNK_SIZE <= K) {
        policy.load_x_fast(curr_x, k_base);

        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) {
                acc[h] += policy.compute_dot(h, true); 
            }
        }
        
        policy.template advance_pointers<HEADS>(Policy::FAST_K_CHUNK_SIZE);
        k_base += Policy::FAST_K_CHUNK_SIZE;
    }

    // 2. Safe/Tail Loop
    while (k_base < K) { 
        policy.load_x_safe(curr_x, k_base, K);
        
        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) {
                acc[h] += policy.compute_dot(h, false); 
            }
        }
        
        policy.template advance_pointers<HEADS>(Policy::SAFE_K_CHUNK_SIZE);
        k_base += Policy::SAFE_K_CHUNK_SIZE;
    }

    // Handle residual batch offset if present
    const device half *curr_residual = nullptr;
    if (residual) {
        // Assume residual has same shape as Y [Batch, N] -> stride_y
        curr_residual = residual + (ulong)batch_idx * gemv_params.stride_y; 
    }

    // 3. Final Reduction & Epilogue delegated to Policy
    EpiloguePolicy::template apply<HEADS>(
        acc, lane_id, logical_col, N, bias, has_bias_flags, alpha, beta, curr_residual, curr_y
    );
}

// =================================================================================================
// FP16 Policy
// =================================================================================================

struct SimdGemvPolicyF16 {
    struct Params {
        const device half *matrix;
    };
    
    // F16: Unrolls 1x float4 load per thread (8 halves total per thread).
    // Warp = 32 threads. Total stride = 32 * 8 = 256.
    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256; // Fallback to single float4 load per thread (8 halves) stride

    uint lane_id;
    uint k_thread_offset;
    
    // Per-head pointers
    const device half *ptr_a[8]; 
    
    // Loaded X vectors
    float4 xv_f32_0_lo, xv_f32_0_hi;
    
    // For safe mode (single vector)
    float4 xv_safe_lo, xv_safe_hi;
    
    // Internal state for safe mode limit handling
    uint safe_n; // Number of valid elements this thread can read (0..8)

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        k_thread_offset = lane_id * 8u;
        
        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for(uint h=0; h<HEADS; ++h) {
             ptr_a[h] = p.matrix + matrix_batch_offset + (ulong)(logical_col + h) * gp.stride_w + k_thread_offset;
        }
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
         uint k_0 = k_base + k_thread_offset;
         
         float4 xv_raw_0 = *(const device float4*)(vector_x + k_0);
         
         half4 xv_lo_0 = as_type<half4>(xv_raw_0.xy); half4 xv_hi_0 = as_type<half4>(xv_raw_0.zw);
         
         xv_f32_0_lo = float4(xv_lo_0); xv_f32_0_hi = float4(xv_hi_0);
    }
    
    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        // Stride is 256 (SAFE_K_CHUNK_SIZE). 
        // We load 8 halves (lane * 8).
        uint k = k_base + k_thread_offset;
        
        float4 xv_raw = float4(0.0f);
        if (k + 8 <= K) {
             xv_raw = *(const device float4*)(vector_x + k);
             safe_n = 8;
        } else {
             safe_n = 0;
             for (uint i=0; i<8 && k+i < K; ++i) {
                 ((thread half*)&xv_raw)[i] = vector_x[k+i];
                 safe_n++;
             }
        }
        
        half4 xv_lo = as_type<half4>(xv_raw.xy); half4 xv_hi = as_type<half4>(xv_raw.zw);
        xv_safe_lo = float4(xv_lo); xv_safe_hi = float4(xv_hi);
    }

    float compute_dot(uint h, bool fast_mode) {
        if (fast_mode) {
             // Load Weights (Unrolled) - Offset embedded in ptr_a
             float4 w_raw_0 = *(const device float4*)(ptr_a[h]);
             half4 w_lo_0 = as_type<half4>(w_raw_0.xy); half4 w_hi_0 = as_type<half4>(w_raw_0.zw);
             return dot(xv_f32_0_lo, float4(w_lo_0)) + dot(xv_f32_0_hi, float4(w_hi_0));
        } else {
             // Safe Load Weights
             float4 w_raw = float4(0.0f);
             if (safe_n == 8) {
                 w_raw = *(const device float4*)(ptr_a[h]);
             } else {
                 for (uint i=0; i<safe_n; ++i) {
                     ((thread half*)&w_raw)[i] = (ptr_a[h])[i];
                 }
             }
             
             half4 w_lo = as_type<half4>(w_raw.xy); half4 w_hi = as_type<half4>(w_raw.zw);
             return dot(xv_safe_lo, float4(w_lo)) + dot(xv_safe_hi, float4(w_hi));
        }
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for(uint h=0; h<HEADS; ++h) ptr_a[h] += k_step;
    }
};

// =================================================================================================
// FP16 Policy (Strided / Non-Transposed)
// =================================================================================================

struct SimdGemvPolicyF16Strided {
    struct Params {
        const device half *matrix;
    };
    
    // Scalar loads are expensive, so we reduce unrolling slightly or keep matched?
    // Matching F16 (16 elems) creates 16 scalar loads. 
    // Let's stick to SAFE_K_CHUNK_SIZE which processes 8 elems (1 float4 of X).
    // To match template loop structure, FAST_K should be multiple of SAFE_K.
    // We'll set FAST = SAFE for simplicity and to minimize register spills with scalar loads.
    static constant uint FAST_K_CHUNK_SIZE = 256; 
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint k_thread_offset;
    
    const device half *ptr_a[8]; 
    uint stride_a[8];
    
    float4 xv_safe_lo, xv_safe_hi;
    uint safe_n;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        k_thread_offset = lane_id * 8u;
        
        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for(uint h=0; h<HEADS; ++h) {
             stride_a[h] = gp.stride_w; 
             ptr_a[h] = p.matrix + matrix_batch_offset + (ulong)(logical_col + h) + (ulong)k_thread_offset * gp.stride_w;
        }
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
         // Fallback to safe load logic since FAST = SAFE
         load_x_safe(vector_x, k_base, 0xFFFFFFFF);
    }
    
    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k = k_base + k_thread_offset;
        
        float4 xv_raw = float4(0.0f);
        if (k + 8 <= K) {
             xv_raw = *(const device float4*)(vector_x + k);
             safe_n = 8;
        } else {
             safe_n = 0;
             for (uint i=0; i<8 && k+i < K; ++i) {
                 ((thread half*)&xv_raw)[i] = vector_x[k+i];
                 safe_n++;
             }
        }
        
        half4 xv_lo = as_type<half4>(xv_raw.xy); half4 xv_hi = as_type<half4>(xv_raw.zw);
        xv_safe_lo = float4(xv_lo); xv_safe_hi = float4(xv_hi);
    }

    float compute_dot(uint h, bool fast_mode) {
        // Scalar strided loads
        half tmp[8] = {0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h, 0.0h};
        
        uint n_limit = (fast_mode) ? 8 : safe_n;
        const device half *p = ptr_a[h];
        uint s = stride_a[h];
        
        // Manual unroll for 8 elements
        if (n_limit > 0) { tmp[0] = *p; p += s; }
        if (n_limit > 1) { tmp[1] = *p; p += s; }
        if (n_limit > 2) { tmp[2] = *p; p += s; }
        if (n_limit > 3) { tmp[3] = *p; p += s; }
        
        if (n_limit > 4) { tmp[4] = *p; p += s; }
        if (n_limit > 5) { tmp[5] = *p; p += s; }
        if (n_limit > 6) { tmp[6] = *p; p += s; }
        if (n_limit > 7) { tmp[7] = *p; p += s; }
        
        half4 w_lo = half4(tmp[0], tmp[1], tmp[2], tmp[3]);
        half4 w_hi = half4(tmp[4], tmp[5], tmp[6], tmp[7]);
        
        return dot(xv_safe_lo, float4(w_lo)) + dot(xv_safe_hi, float4(w_hi));
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for(uint h=0; h<HEADS; ++h) ptr_a[h] += (ulong)k_step * stride_a[h];
    }
};

// =================================================================================================
// FP16 Policy (RMSNorm fused)
// =================================================================================================

struct SimdGemvPolicyF16Rmsnorm {
    struct Params {
        const device half *matrix;
        const device half *gamma;
        float inv_rms;
    };

    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint k_thread_offset;

    const device half *ptr_a[8];
    const device half *gamma_ptr;
    float inv_rms;

    float4 xv_f32_0_lo, xv_f32_0_hi;
    float4 xv_safe_lo, xv_safe_hi;
    uint safe_n;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        k_thread_offset = lane_id * 8u;
        gamma_ptr = p.gamma;
        inv_rms = p.inv_rms;

        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for (uint h = 0; h < HEADS; ++h) {
            ptr_a[h] = p.matrix + matrix_batch_offset + (ulong)(logical_col + h) * gp.stride_w + k_thread_offset;
        }
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
        uint k_0 = k_base + k_thread_offset;

        float4 xv_raw_0 = *(const device float4*)(vector_x + k_0);
        float4 gv_raw_0 = *(const device float4*)(gamma_ptr + k_0);

        half4 xv_lo_0 = as_type<half4>(xv_raw_0.xy); half4 xv_hi_0 = as_type<half4>(xv_raw_0.zw);
        half4 gv_lo_0 = as_type<half4>(gv_raw_0.xy); half4 gv_hi_0 = as_type<half4>(gv_raw_0.zw);

        float inv = inv_rms;
        xv_f32_0_lo = float4(xv_lo_0) * float4(gv_lo_0) * inv;
        xv_f32_0_hi = float4(xv_hi_0) * float4(gv_hi_0) * inv;
    }

    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k = k_base + k_thread_offset;

        float4 xv_raw = float4(0.0f);
        float4 gv_raw = float4(0.0f);
        if (k + 8 <= K) {
            xv_raw = *(const device float4*)(vector_x + k);
            gv_raw = *(const device float4*)(gamma_ptr + k);
            safe_n = 8;
        } else {
            safe_n = 0;
            for (uint i = 0; i < 8 && k + i < K; ++i) {
                ((thread half*)&xv_raw)[i] = vector_x[k + i];
                ((thread half*)&gv_raw)[i] = gamma_ptr[k + i];
                safe_n++;
            }
        }

        half4 xv_lo = as_type<half4>(xv_raw.xy); half4 xv_hi = as_type<half4>(xv_raw.zw);
        half4 gv_lo = as_type<half4>(gv_raw.xy); half4 gv_hi = as_type<half4>(gv_raw.zw);
        float inv = inv_rms;
        xv_safe_lo = float4(xv_lo) * float4(gv_lo) * inv;
        xv_safe_hi = float4(xv_hi) * float4(gv_hi) * inv;
    }

    float compute_dot(uint h, bool fast_mode) {
        if (fast_mode) {
            float4 w_raw_0 = *(const device float4*)(ptr_a[h]);
            half4 w_lo_0 = as_type<half4>(w_raw_0.xy); half4 w_hi_0 = as_type<half4>(w_raw_0.zw);
            return dot(xv_f32_0_lo, float4(w_lo_0)) + dot(xv_f32_0_hi, float4(w_hi_0));
        }

        float4 w_raw = float4(0.0f);
        if (safe_n == 8) {
            w_raw = *(const device float4*)(ptr_a[h]);
        } else {
            for (uint i = 0; i < safe_n; ++i) {
                ((thread half*)&w_raw)[i] = (ptr_a[h])[i];
            }
        }

        half4 w_lo = as_type<half4>(w_raw.xy); half4 w_hi = as_type<half4>(w_raw.zw);
        return dot(xv_safe_lo, float4(w_lo)) + dot(xv_safe_hi, float4(w_hi));
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) ptr_a[h] += k_step;
    }
};

struct SimdGemvPolicyF16StridedRmsnorm {
    struct Params {
        const device half *matrix;
        const device half *gamma;
        float inv_rms;
    };

    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint k_thread_offset;

    const device half *ptr_a[8];
    uint stride_a[8];
    const device half *gamma_ptr;
    float inv_rms;

    float4 xv_safe_lo, xv_safe_hi;
    uint safe_n;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        k_thread_offset = lane_id * 8u;
        gamma_ptr = p.gamma;
        inv_rms = p.inv_rms;

        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for (uint h = 0; h < HEADS; ++h) {
            stride_a[h] = gp.stride_w;
            ptr_a[h] = p.matrix + matrix_batch_offset + (ulong)(logical_col + h) + (ulong)k_thread_offset * stride_a[h];
        }
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
        load_x_safe(vector_x, k_base, 0xFFFFFFFF);
    }

    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k = k_base + k_thread_offset;

        float4 xv_raw = float4(0.0f);
        float4 gv_raw = float4(0.0f);
        
        if (k + 8 <= K) {
            xv_raw = *(const device float4*)(vector_x + k);
            gv_raw = *(const device float4*)(gamma_ptr + k);
            safe_n = 8;
        } else {
            safe_n = 0;
            for (uint i = 0; i < 8 && k + i < K; ++i) {
                ((thread half*)&xv_raw)[i] = vector_x[k + i];
                ((thread half*)&gv_raw)[i] = gamma_ptr[k + i];
                safe_n++;
            }
        }

        half4 xv_lo = as_type<half4>(xv_raw.xy); half4 xv_hi = as_type<half4>(xv_raw.zw);
        half4 gv_lo = as_type<half4>(gv_raw.xy); half4 gv_hi = as_type<half4>(gv_raw.zw);
        float inv = inv_rms;
        
        xv_safe_lo = float4(xv_lo) * float4(gv_lo) * inv;
        xv_safe_hi = float4(xv_hi) * float4(gv_hi) * inv;
    }

    float compute_dot(uint h, bool fast_mode) {
        uint n_limit = (fast_mode) ? 8 : safe_n;
        uint s = stride_a[h];
        
        half4 hw_lo = half4(0.0h);
        half4 hw_hi = half4(0.0h);
        
        const device half *p2 = ptr_a[h];
        if (n_limit > 0) { hw_lo[0] = *p2; p2 += s; }
        if (n_limit > 1) { hw_lo[1] = *p2; p2 += s; }
        if (n_limit > 2) { hw_lo[2] = *p2; p2 += s; }
        if (n_limit > 3) { hw_lo[3] = *p2; p2 += s; }
        
        if (n_limit > 4) { hw_hi[0] = *p2; p2 += s; }
        if (n_limit > 5) { hw_hi[1] = *p2; p2 += s; }
        if (n_limit > 6) { hw_hi[2] = *p2; p2 += s; }
        if (n_limit > 7) { hw_hi[3] = *p2; p2 += s; }
        
        return dot(xv_safe_lo, float4(hw_lo)) + dot(xv_safe_hi, float4(hw_hi));
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) ptr_a[h] += (ulong)k_step * stride_a[h];
    }
};

// =================================================================================================
// FP16 Canonical Policy (k-block-major layout)
// =================================================================================================

struct SimdGemvPolicyF16Canonical {
    struct Params {
        const device half **data;
        uint weights_per_block;
    };

    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint block_in_group;
    uint sub_offset;
    uint sub_lane;

    const device half *ptr_w[8];
    uint stride_w[8];

    float4 xv0, xv1;
    uint block_idx_0, block_idx_1, total_blocks;
    uint weights_per_block;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        block_in_group = lane_id / 8u;
        sub_lane = lane_id % 8u;
        sub_offset = sub_lane * 4u;
        weights_per_block = p.weights_per_block;

        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for (uint h = 0; h < HEADS; ++h) {
            stride_w[h] = N[h] * weights_per_block;
            ptr_w[h] = p.data[h] + matrix_batch_offset + logical_col * weights_per_block + (block_in_group * stride_w[h]);
        }
        total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
        uint k_idx_0 = k_base + block_in_group * weights_per_block + sub_offset;
        uint k_idx_1 = k_idx_0 + 4u * weights_per_block;

        xv0 = float4(*(const device half4*)(vector_x + k_idx_0));
        xv1 = float4(*(const device half4*)(vector_x + k_idx_1));

        block_idx_0 = (k_base / weights_per_block) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
    }

    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k_idx_0 = k_base + block_in_group * weights_per_block + sub_offset;
        uint k_idx_1 = k_idx_0 + 4u * weights_per_block;

        xv0 = float4(0.0f);
        xv1 = float4(0.0f);

        if (k_idx_0 + 4u <= K) {
            xv0 = float4(*(const device half4*)(vector_x + k_idx_0));
        }
        if (k_idx_1 + 4u <= K) {
            xv1 = float4(*(const device half4*)(vector_x + k_idx_1));
        }

        block_idx_0 = (k_base / weights_per_block) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
        total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    }

    float compute_dot(uint h, bool fast_mode) {
        float partial = 0.0f;

        if (fast_mode || block_idx_0 < total_blocks) {
            const device half *w_ptr = ptr_w[h] + sub_offset;
            half4 w_vec = *(const device half4*)(w_ptr);
            partial += dot(xv0, float4(w_vec));
        }

        if (fast_mode || block_idx_1 < total_blocks) {
            const device half *w_ptr = ptr_w[h] + (4u * stride_w[h]) + sub_offset;
            half4 w_vec = *(const device half4*)(w_ptr);
            partial += dot(xv1, float4(w_vec));
        }

        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);

        if (sub_lane == 0u) return partial;
        return 0.0f;
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) {
            ptr_w[h] += stride_w[h] * 8u;
        }
    }
};


// =================================================================================================
// FP16 Canonical Policy (RMSNorm fused)
// =================================================================================================

struct SimdGemvPolicyF16CanonicalRmsnorm {
    struct Params {
        const device half **data;
        const device half *gamma;
        float inv_rms;
        uint weights_per_block;
    };

    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint block_in_group;
    uint sub_offset;
    uint sub_lane;

    const device half *ptr_w[8];
    uint stride_w[8];
    const device half *gamma_ptr;
    float inv_rms;

    float4 xv0, xv1;
    uint block_idx_0, block_idx_1, total_blocks;
    uint weights_per_block;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        block_in_group = lane_id / 8u;
        sub_lane = lane_id % 8u;
        sub_offset = sub_lane * 4u;
        gamma_ptr = p.gamma;
        inv_rms = p.inv_rms;
        weights_per_block = p.weights_per_block;

        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for (uint h = 0; h < HEADS; ++h) {
            stride_w[h] = N[h] * weights_per_block;
            ptr_w[h] = p.data[h] + matrix_batch_offset + logical_col * weights_per_block + (block_in_group * stride_w[h]);
        }
        total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
        uint k_idx_0 = k_base + block_in_group * weights_per_block + sub_offset;
        uint k_idx_1 = k_idx_0 + 4u * weights_per_block;

        half4 x0 = *(const device half4*)(vector_x + k_idx_0);
        half4 x1 = *(const device half4*)(vector_x + k_idx_1);
        half4 g0 = *(const device half4*)(gamma_ptr + k_idx_0);
        half4 g1 = *(const device half4*)(gamma_ptr + k_idx_1);

        float inv = inv_rms;
        xv0 = float4(x0) * float4(g0) * inv;
        xv1 = float4(x1) * float4(g1) * inv;

        block_idx_0 = (k_base / weights_per_block) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
    }

    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k_idx_0 = k_base + block_in_group * weights_per_block + sub_offset;
        uint k_idx_1 = k_idx_0 + 4u * weights_per_block;

        xv0 = float4(0.0f);
        xv1 = float4(0.0f);

        if (k_idx_0 + 4u <= K) {
            half4 x0 = *(const device half4*)(vector_x + k_idx_0);
            half4 g0 = *(const device half4*)(gamma_ptr + k_idx_0);
            xv0 = float4(x0) * float4(g0) * inv_rms;
        }
        if (k_idx_1 + 4u <= K) {
            half4 x1 = *(const device half4*)(vector_x + k_idx_1);
            half4 g1 = *(const device half4*)(gamma_ptr + k_idx_1);
            xv1 = float4(x1) * float4(g1) * inv_rms;
        }

        block_idx_0 = (k_base / weights_per_block) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
        total_blocks = (K + weights_per_block - 1u) / weights_per_block;
    }

    float compute_dot(uint h, bool fast_mode) {
        float partial = 0.0f;

        if (fast_mode || block_idx_0 < total_blocks) {
            const device half *w_ptr = ptr_w[h] + sub_offset;
            half4 w_vec = *(const device half4*)(w_ptr);
            partial += dot(xv0, float4(w_vec));
        }

        if (fast_mode || block_idx_1 < total_blocks) {
            const device half *w_ptr = ptr_w[h] + (4u * stride_w[h]) + sub_offset;
            half4 w_vec = *(const device half4*)(w_ptr);
            partial += dot(xv1, float4(w_vec));
        }

        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);

        if (sub_lane == 0u) return partial;
        return 0.0f;
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) {
            ptr_w[h] += (stride_w[h] * 8u);
        }
    }
};


// =================================================================================================
// Q8 Policy
// =================================================================================================

struct SimdGemvPolicyQ8 {
    struct Params {
        const device uchar **data;
        const device uchar **scale_bytes;
        uint weights_per_block;
    };

    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint block_in_group;
    uint sub_offset;
    uint sub_lane;
    
    // Per-head pointers (Max 3)
    const device uchar *ptr_q[8];
    const device uchar *ptr_s[8];
    uint stride_q[8];
    uint stride_s[8];
    
    // Loaded X vectors
    float4 xv0, xv1;
    
    // State for safe mode
    uint k_curr_0, k_curr_1;
    uint k_limit;
    uint block_idx_0, block_idx_1, total_blocks;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        block_in_group = lane_id / 8u;
        sub_lane = lane_id % 8u;
        sub_offset = sub_lane * 4u;

        const ulong q_batch_offset = (ulong)batch_idx * gp.stride_a;
        const ulong s_batch_offset = (ulong)batch_idx * gp.stride_scale;
        
        for (uint h = 0; h < HEADS; ++h) {
            stride_q[h] = N[h] * 32u;
            stride_s[h] = N[h] * 2u;

            // Absorb block_in_group offset
            ptr_q[h] = p.data[h] + q_batch_offset + logical_col * 32u + (block_in_group * stride_q[h]);
            ptr_s[h] = p.scale_bytes[h] + s_batch_offset + logical_col * 2u + (block_in_group * stride_s[h]);
        }
        total_blocks = (K + 31u) / 32u;
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
        // Q8 logic: Load `xv0` at `k_base ...` and `xv1` at `k_base + 128 ...`
        
        uint k_idx_0 = k_base + block_in_group * 32u + sub_offset;
        uint k_idx_1 = k_idx_0 + 128u;
        
        xv0 = float4(*(const device half4*)(vector_x + k_idx_0));
        xv1 = float4(*(const device half4*)(vector_x + k_idx_1));
        
        block_idx_0 = (k_base / 32u) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
    }

    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k_idx_0 = k_base + block_in_group * 32u + sub_offset;
        uint k_idx_1 = k_idx_0 + 128u;
        
        xv0 = float4(0.0f);
        xv1 = float4(0.0f);

        if (k_idx_0 + 4 <= K) {
             xv0 = float4(*(const device half4*)(vector_x + k_idx_0));
        }
        if (k_idx_1 + 4 <= K) {
             xv1 = float4(*(const device half4*)(vector_x + k_idx_1));
        }
        
        block_idx_0 = (k_base / 32u) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
        total_blocks = (K + 31u) / 32u; // Recalc? No const.
    }

    float compute_dot(uint h, bool fast_mode) {
        // Same logic for fast/safe because checks are on `block_idx` vs `total_blocks`.
        // In Fast Mode, `block_idx` should be valid.
        
        float partial = 0.0f;

        // Part 0
        if (fast_mode || block_idx_0 < total_blocks) {
             const device uchar *s_ptr = ptr_s[h];
             const device uchar *q_ptr = ptr_q[h];
             
             ushort s_bits = *(const device ushort*)s_ptr;
             float scale = (float)as_type<half>(s_bits);
             uchar4 q_bytes = *(const device uchar4*)(q_ptr + sub_offset);
             float4 w_vec = float4(char4(q_bytes));
             
             partial += dot(xv0, w_vec) * scale;
        }
        
        // Part 1
        if (fast_mode || block_idx_1 < total_blocks) {
             // Offset 4 blocks (4 * stride)
             const device uchar *s_ptr = ptr_s[h] + (4u * stride_s[h]);
             const device uchar *q_ptr = ptr_q[h] + (4u * stride_q[h]);

             ushort s_bits = *(const device ushort*)s_ptr;
             float scale = (float)as_type<half>(s_bits);
             uchar4 q_bytes = *(const device uchar4*)(q_ptr + sub_offset);
             float4 w_vec = float4(char4(q_bytes));

             partial += dot(xv1, w_vec) * scale;
        }

        // Intra-warp shuffle reduction
        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);
        
        if (sub_lane == 0) return partial;
        return 0.0f;
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for(uint h=0; h<HEADS; ++h) {
            ptr_q[h] += stride_q[h] * 8u;
            ptr_s[h] += stride_s[h] * 8u;
        }
    }
};

// =================================================================================================
// Q8 Policy (RMSNorm fused)
// =================================================================================================

struct SimdGemvPolicyQ8Rmsnorm {
    struct Params {
        const device uchar **data;
        const device uchar **scale_bytes;
        const device half *gamma;
        uint weights_per_block;
        float inv_rms;
    };

    static constant uint FAST_K_CHUNK_SIZE = 256;
    static constant uint SAFE_K_CHUNK_SIZE = 256;

    uint lane_id;
    uint block_in_group;
    uint sub_offset;
    uint sub_lane;

    const device uchar *ptr_q[8];
    const device uchar *ptr_s[8];
    uint stride_q[8];
    uint stride_s[8];

    const device half *gamma_ptr;
    float inv_rms;

    float4 xv0, xv1;
    uint k_curr_0, k_curr_1;
    uint block_idx_0, block_idx_1, total_blocks;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx) {
        lane_id = lid.x & 31u;
        block_in_group = lane_id / 8u;
        sub_lane = lane_id % 8u;
        sub_offset = sub_lane * 4u;
        gamma_ptr = p.gamma;
        inv_rms = p.inv_rms;

        const ulong q_batch_offset = (ulong)batch_idx * gp.stride_a;
        const ulong s_batch_offset = (ulong)batch_idx * gp.stride_scale;

        for (uint h = 0; h < HEADS; ++h) {
            stride_q[h] = N[h] * 32u;
            stride_s[h] = N[h] * 2u;

            ptr_q[h] = p.data[h] + q_batch_offset + logical_col * 32u + (block_in_group * stride_q[h]);
            ptr_s[h] = p.scale_bytes[h] + s_batch_offset + logical_col * 2u + (block_in_group * stride_s[h]);
        }
        total_blocks = (K + 31u) / 32u;
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
        uint k_idx_0 = k_base + block_in_group * 32u + sub_offset;
        uint k_idx_1 = k_idx_0 + 128u;

        xv0 = float4(*(const device half4*)(vector_x + k_idx_0));
        xv1 = float4(*(const device half4*)(vector_x + k_idx_1));

        float4 g0 = float4(*(const device half4*)(gamma_ptr + k_idx_0));
        float4 g1 = float4(*(const device half4*)(gamma_ptr + k_idx_1));
        float inv = inv_rms;
        xv0 = xv0 * g0 * inv;
        xv1 = xv1 * g1 * inv;

        block_idx_0 = (k_base / 32u) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
    }

    void load_x_safe(const device half* vector_x, uint k_base, uint K) {
        uint k_idx_0 = k_base + block_in_group * 32u + sub_offset;
        uint k_idx_1 = k_idx_0 + 128u;

        xv0 = float4(0.0f);
        xv1 = float4(0.0f);
        float4 g0 = float4(0.0f);
        float4 g1 = float4(0.0f);

        if (k_idx_0 + 4 <= K) {
            xv0 = float4(*(const device half4*)(vector_x + k_idx_0));
            g0 = float4(*(const device half4*)(gamma_ptr + k_idx_0));
        } else {
            half4 xv_half = half4(0.0h);
            half4 gv_half = half4(0.0h);
            for (uint i = 0; i < 4 && k_idx_0 + i < K; ++i) {
                ((thread half*)&xv_half)[i] = vector_x[k_idx_0 + i];
                ((thread half*)&gv_half)[i] = gamma_ptr[k_idx_0 + i];
            }
            xv0 = float4(xv_half);
            g0 = float4(gv_half);
        }
        if (k_idx_1 + 4 <= K) {
            xv1 = float4(*(const device half4*)(vector_x + k_idx_1));
            g1 = float4(*(const device half4*)(gamma_ptr + k_idx_1));
        } else {
            half4 xv_half = half4(0.0h);
            half4 gv_half = half4(0.0h);
            for (uint i = 0; i < 4 && k_idx_1 + i < K; ++i) {
                ((thread half*)&xv_half)[i] = vector_x[k_idx_1 + i];
                ((thread half*)&gv_half)[i] = gamma_ptr[k_idx_1 + i];
            }
            xv1 = float4(xv_half);
            g1 = float4(gv_half);
        }

        float inv = inv_rms;
        xv0 = xv0 * g0 * inv;
        xv1 = xv1 * g1 * inv;

        block_idx_0 = (k_base / 32u) + block_in_group;
        block_idx_1 = block_idx_0 + 4u;
        total_blocks = (K + 31u) / 32u;
    }

    float compute_dot(uint h, bool fast_mode) {
        float partial = 0.0f;

        if (fast_mode || block_idx_0 < total_blocks) {
            const device uchar *s_ptr = ptr_s[h];
            const device uchar *q_ptr = ptr_q[h];

            ushort s_bits = *(const device ushort*)s_ptr;
            float scale = (float)as_type<half>(s_bits);
            uchar4 q_bytes = *(const device uchar4*)(q_ptr + sub_offset);
            float4 w_vec = float4(char4(q_bytes));

            partial += dot(xv0, w_vec) * scale;
        }

        if (fast_mode || block_idx_1 < total_blocks) {
            const device uchar *s_ptr = ptr_s[h] + (4u * stride_s[h]);
            const device uchar *q_ptr = ptr_q[h] + (4u * stride_q[h]);

            ushort s_bits = *(const device ushort*)s_ptr;
            float scale = (float)as_type<half>(s_bits);
            uchar4 q_bytes = *(const device uchar4*)(q_ptr + sub_offset);
            float4 w_vec = float4(char4(q_bytes));

            partial += dot(xv1, w_vec) * scale;
        }

        partial += simd_shuffle_xor(partial, 4u);
        partial += simd_shuffle_xor(partial, 2u);
        partial += simd_shuffle_xor(partial, 1u);

        if (sub_lane == 0) return partial;
        return 0.0f;
    }

    template<uint HEADS>
    void advance_pointers(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) {
            ptr_q[h] += stride_q[h] * 8u;
            ptr_s[h] += stride_s[h] * 8u;
        }
    }
};

// =================================================================================================
// Wrappers
// =================================================================================================

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16::Params p = { matrix };
    run_simd_gemv_template<SimdGemvPolicyF16, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv_strided(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16Strided::Params p = { matrix };
    run_simd_gemv_template<SimdGemvPolicyF16Strided, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv_rmsnorm(
    const device half *matrix,
    const device half *vector_x,
    const device half *gamma,
    const float inv_rms,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16Rmsnorm::Params p = { matrix, gamma, inv_rms };
    run_simd_gemv_template<SimdGemvPolicyF16Rmsnorm, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv_rmsnorm_cols8(
    const device half *matrix,
    const device half *vector_x,
    const device half *gamma,
    const float inv_rms,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16Rmsnorm::Params p = { matrix, gamma, inv_rms };
    run_simd_gemv_template<SimdGemvPolicyF16Rmsnorm, HEADS, 8, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv_rmsnorm_strided(
    const device half *matrix,
    const device half *vector_x,
    const device half *gamma,
    const float inv_rms,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16StridedRmsnorm::Params p = { matrix, gamma, inv_rms };
    run_simd_gemv_template<SimdGemvPolicyF16StridedRmsnorm, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv_cols2(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16::Params p = { matrix };
    run_simd_gemv_template<SimdGemvPolicyF16, HEADS, 2, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_gemv_cols8(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyF16::Params p = { matrix };
    run_simd_gemv_template<SimdGemvPolicyF16, HEADS, 8, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_canonical_gemv_cols2(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    const device half *data_arr[HEADS] = { matrix };
    SimdGemvPolicyF16Canonical::Params p = { (const device half **)data_arr, weights_per_block };
    run_simd_gemv_template<SimdGemvPolicyF16Canonical, HEADS, 2, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_canonical_gemv_cols8(
    const device half *matrix,
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    const device half *data_arr[HEADS] = { matrix };
    SimdGemvPolicyF16Canonical::Params p = { (const device half **)data_arr, weights_per_block };
    run_simd_gemv_template<SimdGemvPolicyF16Canonical, HEADS, 8, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_f16_canonical_gemv_rmsnorm(
    const device half *matrix,
    const device half *vector_x,
    const device half *gamma,
    const float inv_rms,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    const device half *data_arr[HEADS] = { matrix };
    SimdGemvPolicyF16CanonicalRmsnorm::Params p = { (const device half **)data_arr, gamma, inv_rms, weights_per_block };
    run_simd_gemv_template<SimdGemvPolicyF16CanonicalRmsnorm, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_q8_gemv(
    const device uchar *data[HEADS],
    const device uchar *scale_bytes[HEADS],
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    // Cast arrays to const device uchar** if needed, or pass array directly.
    // The wrapper expects `const device uchar* data[HEADS]`.
    // The struct Params has `const device uchar **data`. 
    // They are compatible if we pass the array name (decays to pointer).
    
    SimdGemvPolicyQ8::Params p = { 
        (const device uchar**)data, 
        (const device uchar**)scale_bytes, 
        weights_per_block 
    };
    
    run_simd_gemv_template<SimdGemvPolicyQ8, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_q8_gemv_cols2(
    const device uchar *data[HEADS],
    const device uchar *scale_bytes[HEADS],
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyQ8::Params p = {
        (const device uchar**)data,
        (const device uchar**)scale_bytes,
        weights_per_block
    };

    run_simd_gemv_template<SimdGemvPolicyQ8, HEADS, 2, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_q8_gemv_cols8(
    const device uchar *data[HEADS],
    const device uchar *scale_bytes[HEADS],
    const device half *vector_x,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyQ8::Params p = {
        (const device uchar**)data,
        (const device uchar**)scale_bytes,
        weights_per_block
    };

    run_simd_gemv_template<SimdGemvPolicyQ8, HEADS, 8, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}

template <uint HEADS, bool HasBias>
void run_simd_q8_gemv_rmsnorm(
    const device uchar *data[HEADS],
    const device uchar *scale_bytes[HEADS],
    const device half *vector_x,
    const device half *gamma,
    const float inv_rms,
    device half *result_y[HEADS],
    const uint N[HEADS],
    const uint K,
    const uint weights_per_block,
    const device half *bias[HEADS],
    const uint has_bias_flags[HEADS],
    const float alpha,
    const float beta,
    const device half *residual,
    uint3 gid,
    uint3 lid,
    uint gp_k, uint gp_n, uint gp_blocks_per_k, uint gp_weights_per_block, uint gp_batch, uint gp_stride_x, uint gp_stride_y, uint gp_stride_a, uint gp_stride_w, uint gp_stride_scale
) {
    GemvParams gemv_params = { gp_k, gp_n, gp_blocks_per_k, gp_weights_per_block, gp_batch, gp_stride_x, gp_stride_y, gp_stride_a, gp_stride_w, gp_stride_scale };
    SimdGemvPolicyQ8Rmsnorm::Params p = {
        (const device uchar**)data,
        (const device uchar**)scale_bytes,
        gamma,
        weights_per_block,
        inv_rms
    };

    run_simd_gemv_template<SimdGemvPolicyQ8Rmsnorm, HEADS, 4, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid, gemv_params
    );
}
