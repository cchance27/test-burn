#include "base.metal"

#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 1
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_Q8_DEFINED
#define POLICY_Q8_DEFINED

/**
 * Policy implementation for GGUF Q8_0 (8-bit quantized) weights.
 * 
 * Format:
 * - Weights are stored as signed chars (int8).
 * - Each block of 32 weights has a singular half-precision scale factor.
 * - This policy expects split buffers (weights and scales separated) for 
 *   compatibility with standard GEMM/GEMV tiling.
 */
struct PolicyQ8 {
    static constant bool HAS_SCALE = true;
    /**
     * Load the block scale (FP16).
     * 
     * Each block is 32 elements. Scales are stored contiguously in the scales buffer.
     */
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        // Load 2-byte scale (half) bits and cast to half.
        // We use ushort bits to avoid potential alignment issues on direct half pointer casts.
        const device uchar *s_ptr = scales + block_idx * 2;
        ushort bits = (ushort)s_ptr[0] | ((ushort)s_ptr[1] << 8);
        return as_type<half>(bits);
    }

    /**
     * Compute dot product with dequantized weights.
     * Encapsulates the load-and-dot pattern to match PolicyF16's optimized interface.
     */
    template<int N>
    static ALWAYS_INLINE float dot(
        const device uchar *ptr, 
        ulong offset, 
        float scale,
        float4 xv_lo,
        float4 xv_hi
    ) {
        thread float w[N];
        load_weights<N>(ptr, offset, w);
        
        float res = 0.0f;
        if constexpr (N >= 4) {
           res += metal::dot(xv_lo, float4(w[0], w[1], w[2], w[3]));
        }
        if constexpr (N >= 8) {
           res += metal::dot(xv_hi, float4(w[4], w[5], w[6], w[7]));
        }
        return scale * res;
    }

    /**
     * Load and expand Q8 weights to floats (pre-scaled).
     * 
     * Note: This method loads raw integer values. Scaling is typically applied 
     * by the kernel after loading.
     */
    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr, 
        ulong offset, 
        thread float results[N]
    ) {
        const device char *w_ptr = (const device char *)ptr;
        
        if constexpr (N == 8) {
            // Optimized: Load 8 bytes (chars) using two 32-bit (uchar4) transactions.
            // Split into two uchar4 loads since uchar8 is not a standard Metal type.
            uchar4 raw0 = *(const device uchar4*)(ptr + offset);
            uchar4 raw1 = *(const device uchar4*)(ptr + offset + 4);
            
            // Dequantize: Q8 values are signed bytes.
            results[0] = (float)(char)raw0.x;
            results[1] = (float)(char)raw0.y;
            results[2] = (float)(char)raw0.z;
            results[3] = (float)(char)raw0.w;
            results[4] = (float)(char)raw1.x;
            results[5] = (float)(char)raw1.y;
            results[6] = (float)(char)raw1.z;
            results[7] = (float)(char)raw1.w;
        } else if constexpr (N == 4) {
            // Optimized: Load 4 bytes using a single 32-bit transaction.
            uchar4 raw = *(const device uchar4*)(ptr + offset);
            results[0] = (float)(char)raw.x;
            results[1] = (float)(char)raw.y;
            results[2] = (float)(char)raw.z;
            results[3] = (float)(char)raw.w;
        } else {
            // Fallback for non-vectorized loads.
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                results[i] = (float)w_ptr[offset + i];
            }
        }
    }
};

#endif // POLICY_Q8_DEFINED
