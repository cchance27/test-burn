#include <metal_stdlib>
using namespace metal;


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

template <typename Policy, uint HEADS, bool HasBias>
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
    uint3 lid
) {
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    
    // Each Warp processes 1 Logical Output Column.
    const uint logical_col = gid.x * 4u + warp_id;
    
    bool head_active[HEADS];
    for (uint h = 0; h < HEADS; ++h) {
        head_active[h] = (logical_col < N[h]);
    }

    // Initialize Policy
    Policy policy;
    policy.template init<HEADS>(params, gid, lid, logical_col, K, N);

    // Accumulators
    float acc[HEADS];
    for (uint h = 0; h < HEADS; ++h) acc[h] = 0.0f;

    uint k_base = 0;

    // 1. Fast Path Loop (No Bounds Checks)
    // Runs while we have full chunks available
    while (k_base + Policy::FAST_K_CHUNK_SIZE <= K) {
        
        policy.load_x_fast(vector_x, k_base);

        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) {
                acc[h] += policy.compute_dot(h, true); // true = fast mode
            }
        }
        
        policy.advance_pointers(Policy::FAST_K_CHUNK_SIZE);
        k_base += Policy::FAST_K_CHUNK_SIZE;
    }

    // 2. Safe/Tail Loop (With Bounds Checks)
    // Runs the remaining elements using the safe stride
    while (k_base < K) { // Iterate until done, policy handles detailed stepping/masking if needed
         
        // Note: The safe chunk size might be smaller or same. 
        // Here we assume the policy might handle partial tiles or we just step by SAFE_K_CHUNK_SIZE
        // and let load_x_safe handle boundaries.
        
        policy.load_x_safe(vector_x, k_base, K);
        
        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) {
                acc[h] += policy.compute_dot(h, false); // false = safe mode
            }
        }
        
        policy.advance_pointers(Policy::SAFE_K_CHUNK_SIZE);
        k_base += Policy::SAFE_K_CHUNK_SIZE;
    }

    // 3. Final Reduction & Epilogue
    for (uint h = 0; h < HEADS; ++h) {
        if (!head_active[h]) continue;
        
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

            if (HEADS == 1) {
                const float val_computed = gemv_epilogue_value<false>(
                    val_acc,
                    (const device half*)nullptr,
                    residual,
                    alpha,
                    beta,
                    logical_col
                );
                result_y[h][logical_col] = (half)val_computed;
            } else {
                const float val_computed = gemv_epilogue_value<false>(
                    val_acc,
                    (const device half*)nullptr,
                    (const device half*)nullptr,
                    1.0f,
                    0.0f,
                    logical_col
                );
                result_y[h][logical_col] = (half)val_computed;
            }
        }
    }
}

// =================================================================================================
// FP16 Policy
// =================================================================================================

struct SimdGemvPolicyF16 {
    struct Params {
        const device half *matrix;
    };
    
    // F16: Unrolls 2x float4 loads per thread (16 halves total per thread).
    // Warp = 32 threads. Total stride = 32 * 16 = 512.
    static constant uint FAST_K_CHUNK_SIZE = 512;
    static constant uint SAFE_K_CHUNK_SIZE = 256; // Fallback to single float4 load per thread (8 halves) stride

    uint lane_id;
    uint k_thread_offset;
    
    // Per-head pointers
    const device half *ptr_a[3]; 
    
    // Loaded X vectors
    float4 xv_f32_0_lo, xv_f32_0_hi;
    float4 xv_f32_1_lo, xv_f32_1_hi;
    
    // For safe mode (single vector)
    float4 xv_safe_lo, xv_safe_hi;
    
    // Internal state for safe mode limit handling
    uint safe_n; // Number of valid elements this thread can read (0..8)

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS]) {
        lane_id = lid.x & 31u;
        k_thread_offset = lane_id * 8u;
        
        for(uint h=0; h<HEADS; ++h) {
            // Absorb thread offset into base pointer
             ptr_a[h] = p.matrix + (ulong)logical_col * K + k_thread_offset;
        }
    }

    void load_x_fast(const device half* vector_x, uint k_base) {
         uint k_0 = k_base + k_thread_offset;
         uint k_1 = k_0 + 256u; 
         
         float4 xv_raw_0 = *(const device float4*)(vector_x + k_0);
         float4 xv_raw_1 = *(const device float4*)(vector_x + k_1);
         
         half4 xv_lo_0 = as_type<half4>(xv_raw_0.xy); half4 xv_hi_0 = as_type<half4>(xv_raw_0.zw);
         half4 xv_lo_1 = as_type<half4>(xv_raw_1.xy); half4 xv_hi_1 = as_type<half4>(xv_raw_1.zw);
         
         xv_f32_0_lo = float4(xv_lo_0); xv_f32_0_hi = float4(xv_hi_0);
         xv_f32_1_lo = float4(xv_lo_1); xv_f32_1_hi = float4(xv_hi_1);
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
             float4 w_raw_1 = *(const device float4*)(ptr_a[h] + 256u);  
             
             half4 w_lo_0 = as_type<half4>(w_raw_0.xy); half4 w_hi_0 = as_type<half4>(w_raw_0.zw);
             half4 w_lo_1 = as_type<half4>(w_raw_1.xy); half4 w_hi_1 = as_type<half4>(w_raw_1.zw);
             
             float sum0 = dot(xv_f32_0_lo, float4(w_lo_0)) + dot(xv_f32_0_hi, float4(w_hi_0));
             float sum1 = dot(xv_f32_1_lo, float4(w_lo_1)) + dot(xv_f32_1_hi, float4(w_hi_1));
             return sum0 + sum1;
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

    void advance_pointers(uint k_step) {
        for(int h=0; h<3; ++h) ptr_a[h] += k_step;
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
    const device uchar *ptr_q[3];
    const device uchar *ptr_s[3];
    uint stride_q[3];
    uint stride_s[3];
    
    // Loaded X vectors
    float4 xv0, xv1;
    
    // State for safe mode
    uint k_curr_0, k_curr_1;
    uint k_limit;
    uint block_idx_0, block_idx_1, total_blocks;

    template<uint HEADS>
    void init(Params p, uint3 gid, uint3 lid, uint logical_col, uint K, const uint N[HEADS]) {
        lane_id = lid.x & 31u;
        block_in_group = lane_id / 8u;
        sub_lane = lane_id % 8u;
        sub_offset = sub_lane * 4u;
        
        for (uint h = 0; h < HEADS; ++h) {
            stride_q[h] = N[h] * 32u;
            stride_s[h] = N[h] * 2u;

            // Absorb block_in_group offset
            ptr_q[h] = p.data[h] + logical_col * 32u + (block_in_group * stride_q[h]);
            ptr_s[h] = p.scale_bytes[h] + logical_col * 2u + (block_in_group * stride_s[h]);
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

    void advance_pointers(uint k_step) {
        // Step is 256 elements = 8 blocks.
        // Pointers advance by stride * 8.
        for(uint h=0; h<3; ++h) {
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
    uint3 lid
) {
    SimdGemvPolicyF16::Params p = { matrix };
    run_simd_gemv_template<SimdGemvPolicyF16, HEADS, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid
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
    uint3 lid
) {
    // Cast arrays to const device uchar** if needed, or pass array directly.
    // The wrapper expects `const device uchar* data[HEADS]`.
    // The struct Params has `const device uchar **data`. 
    // They are compatible if we pass the array name (decays to pointer).
    
    SimdGemvPolicyQ8::Params p = { 
        (const device uchar**)data, 
        (const device uchar**)scale_bytes, 
        weights_per_block 
    };
    
    run_simd_gemv_template<SimdGemvPolicyQ8, HEADS, HasBias>(
        p, vector_x, result_y, N, K, bias, has_bias_flags, alpha, beta, residual, gid, lid
    );
}

// Specialization for Policy Loop:
// We need to resolve the `init` needing `N`.
// I will update the Template to call `policy.init(..., N)`.
// And update F16 to ignore it.

// Re-writing template segment specifically for this file content...

template <typename Policy, uint HEADS, bool HasBias>
void run_simd_gemv_template_final( // Renamed to avoid confusion in thought process, will use real name in file
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
    uint3 lid
) {
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    const uint logical_col = gid.x * 4u + warp_id;
    
    bool head_active[HEADS];
    for (uint h = 0; h < HEADS; ++h) head_active[h] = (logical_col < N[h]);

    Policy policy;
    // Pass N to init
    policy.init(params, gid, lid, logical_col, K, N);

    float acc[HEADS];
    for (uint h = 0; h < HEADS; ++h) acc[h] = 0.0f;

    uint k_base = 0;
    while (k_base + Policy::FAST_K_CHUNK_SIZE <= K) {
        policy.load_x_fast(vector_x, k_base);
        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) acc[h] += policy.compute_dot(h, true);
        }
        policy.advance_pointers(Policy::FAST_K_CHUNK_SIZE);
        k_base += Policy::FAST_K_CHUNK_SIZE;
    }

    while (k_base < K) {
        policy.load_x_safe(vector_x, k_base, K);
        for (uint h = 0; h < HEADS; ++h) {
            if (head_active[h]) acc[h] += policy.compute_dot(h, false);
        }
        policy.advance_pointers(Policy::SAFE_K_CHUNK_SIZE);
        k_base += Policy::SAFE_K_CHUNK_SIZE;
    }

    for (uint h = 0; h < HEADS; ++h) {
        if (!head_active[h]) continue;
        float val = acc[h];
        val += simd_shuffle_xor(val, 16u);
        val += simd_shuffle_xor(val, 8u);
        val += simd_shuffle_xor(val, 4u);
        val += simd_shuffle_xor(val, 2u);
        val += simd_shuffle_xor(val, 1u);
        
        if (lane_id == 0) {
            float val_acc = val;
            if (has_bias_flags[h] && bias[h]) val_acc += (float)bias[h][logical_col];

            if (HEADS == 1) {
                const float val_computed = gemv_epilogue_value<false>(val_acc, (const device half*)nullptr, residual, alpha, beta, logical_col);
                result_y[h][logical_col] = (half)val_computed;
            } else {
                const float val_computed = gemv_epilogue_value<false>(val_acc, (const device half*)nullptr, (const device half*)nullptr, 1.0f, 0.0f, logical_col);
                result_y[h][logical_col] = (half)val_computed;
            }
        }
    }
}
