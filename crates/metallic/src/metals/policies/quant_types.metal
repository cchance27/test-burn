#ifndef METALLIC_QUANT_TYPES_METAL
#define METALLIC_QUANT_TYPES_METAL

#include <metal_stdlib>
using namespace metal;

// =================================================================================================
// F16 Quantization Policy
// =================================================================================================

struct QuantF16 {
    struct Params {
        const device half **data;
        const device half *gamma;
        float inv_rms;
        uint weights_per_block;
    };

    const device half *ptr_w[8];
    uint stride_w[8];
    const device half *gamma_ptr;
    float inv_rms;
    uint weights_per_block;
    uint block_in_group;
    uint sub_offset;

    template<uint HEADS>
    void init(Params p, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx, uint _block_in_group, uint _sub_offset) {
        weights_per_block = p.weights_per_block;
        block_in_group = _block_in_group;
        sub_offset = _sub_offset;
        gamma_ptr = p.gamma;
        inv_rms = p.inv_rms;
        
        const ulong matrix_batch_offset = (ulong)batch_idx * gp.stride_a;

        for (uint h = 0; h < HEADS; ++h) {
            stride_w[h] = N[h] * weights_per_block;
            ptr_w[h] = p.data[h] + matrix_batch_offset + logical_col * weights_per_block + (block_in_group * stride_w[h]);
        }
    }

    template<uint HEADS>
    void advance(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) {
            ptr_w[h] += (stride_w[h] * 8u);
        }
    }

    // Load input vector and apply generic normalization (if gamma present)
    float4 load_input(const device half *vector_x, uint k_idx) {
        half4 x = *(const device half4 *)(vector_x + k_idx);
        if (gamma_ptr != nullptr) {
            half4 g = *(const device half4 *)(gamma_ptr + k_idx);
            return float4(x) * float4(g) * inv_rms;
        }
        return float4(x);
    }
    
    // Check bounds and load safely
    float4 load_input_safe(const device half *vector_x, uint k_idx, uint K) {
        if (k_idx + 4u <= K) {
            return load_input(vector_x, k_idx);
        } else {
             // Partial load
            half4 xv_half = half4(0.0h);
            for (uint i = 0u; i < 4u && k_idx + i < K; ++i) {
                ((thread half *)&xv_half)[i] = vector_x[k_idx + i];
            }
            float4 res = float4(xv_half);
            
            if (gamma_ptr != nullptr) {
                half4 gv_half = half4(0.0h);
                for (uint i = 0u; i < 4u && k_idx + i < K; ++i) {
                    ((thread half *)&gv_half)[i] = gamma_ptr[k_idx + i];
                }
                res = res * float4(gv_half) * inv_rms;
            }
            return res;
        }
    }

    // Compute dot product with a loaded vector register
    float dot_part(uint h, float4 x_reg, bool second_block) {
        uint offset = second_block ? (4u * stride_w[h]) : 0;
        const device half *w_ptr = ptr_w[h] + offset + sub_offset;
        half4 w_vec = *(const device half4 *)(w_ptr);
        return dot(x_reg, float4(w_vec));
    }
};

// =================================================================================================
// Q8 Quantization Policy
// =================================================================================================

struct QuantQ8 {
    struct Params {
        const device uchar **data;
        const device uchar **scale_bytes;
        const device half *gamma;
        float inv_rms;
        uint weights_per_block; 
    };

    const device uchar *ptr_q[8];
    const device uchar *ptr_s[8];
    uint stride_q[8];
    uint stride_s[8];
    const device half *gamma_ptr;
    float inv_rms;
    uint weights_per_block;
    
    uint block_in_group;
    uint sub_offset;

    template<uint HEADS>
    void init(Params p, uint3 lid, uint logical_col, uint K, const uint N[HEADS], GemvParams gp, uint batch_idx, uint _block_in_group, uint _sub_offset) {
        weights_per_block = p.weights_per_block;
        block_in_group = _block_in_group;
        sub_offset = _sub_offset;
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
    }

    template<uint HEADS>
    void advance(uint k_step) {
        for (uint h = 0; h < HEADS; ++h) {
            ptr_q[h] += stride_q[h] * 8u;
            ptr_s[h] += stride_s[h] * 8u;
        }
    }

    // Reuse same logic (could technically reuse code via base class but structs are simpler here)
    float4 load_input(const device half *vector_x, uint k_idx) {
        half4 x = *(const device half4 *)(vector_x + k_idx);
        if (gamma_ptr != nullptr) {
            half4 g = *(const device half4 *)(gamma_ptr + k_idx);
            return float4(x) * float4(g) * inv_rms;
        }
        return float4(x);
    }

    float4 load_input_safe(const device half *vector_x, uint k_idx, uint K) {
         if (k_idx + 4u <= K) {
            return load_input(vector_x, k_idx);
        } else {
             // Partial load
            half4 xv_half = half4(0.0h);
            for (uint i = 0u; i < 4u && k_idx + i < K; ++i) {
                ((thread half *)&xv_half)[i] = vector_x[k_idx + i];
            }
            float4 res = float4(xv_half);
            
            if (gamma_ptr != nullptr) {
                half4 gv_half = half4(0.0h);
                for (uint i = 0u; i < 4u && k_idx + i < K; ++i) {
                    ((thread half *)&gv_half)[i] = gamma_ptr[k_idx + i];
                }
                res = res * float4(gv_half) * inv_rms;
            }
            return res;
        }
    }

    float dot_part(uint h, float4 x_reg, bool second_block) {
        uint q_offset = second_block ? (4u * stride_q[h]) : 0;
        uint s_offset = second_block ? (4u * stride_s[h]) : 0;
        
        const device uchar *q_ptr = ptr_q[h] + q_offset;
        const device uchar *s_ptr = ptr_s[h] + s_offset;

        ushort s_bits = *(const device ushort *)s_ptr;
        float scale = (float)as_type<half>(s_bits);
        uchar4 q_bytes = *(const device uchar4 *)(q_ptr + sub_offset);
        float4 w_vec = float4(char4(q_bytes));

        return dot(x_reg, w_vec) * scale;
    }
};

#endif // METALLIC_QUANT_TYPES_METAL
