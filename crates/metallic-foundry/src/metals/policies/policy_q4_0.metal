
#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 1
#undef METALLIC_POLICY_HAS_AFFINE
#define METALLIC_POLICY_HAS_AFFINE 0
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_Q4_0_DEFINED
#define POLICY_Q4_0_DEFINED

/**
 * Policy implementation for GGUF Q4_0 (4-bit quantized) weights.
 * 
 * Format:
 * - Weights are stored as packed 4-bit nibbles (2 per byte).
 * - Each block of 32 weights has a singular half-precision scale factor.
 * - This policy expects split buffers (weights and scales separated).
 */
struct PolicyQ4_0 {
    static constant bool HAS_SCALE = true;
    static constant bool HAS_AFFINE = false;
    static constant uint WEIGHTS_PER_BLOCK = 32u;
    static constant uint SCALE_BYTES = 2u;

    /**
     * Load the block scale (FP16).
     */
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        const device uchar *s_ptr = scales + block_idx * 2;
        ushort bits = (ushort)s_ptr[0] | ((ushort)s_ptr[1] << 8);
        return as_type<half>(bits);
    }

    static ALWAYS_INLINE half load_affine(const device uchar *scales, ulong block_idx) {
        (void)scales;
        (void)block_idx;
        return 0.0h;
    }

    /**
     * Compute dot product with dequantized weights.
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
     * Load and expand Q4_0 weights to floats (pre-scaled).
     */
    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr, 
        ulong offset, 
        thread float results[N]
    ) {
        if constexpr (N == 8 && true) {
            // Optimized vectorized load for sequential 8 elements (4 bytes)
            const device uchar4 *w_ptr = (const device uchar4 *)(ptr + (offset / 2));
            uchar4 raw = *w_ptr;
            
            results[0] = (float)((int)(raw.x & 0x0F) - 8);
            results[1] = (float)((int)(raw.x >> 4) - 8);
            results[2] = (float)((int)(raw.y & 0x0F) - 8);
            results[3] = (float)((int)(raw.y >> 4) - 8);
            results[4] = (float)((int)(raw.z & 0x0F) - 8);
            results[5] = (float)((int)(raw.z >> 4) - 8);
            results[6] = (float)((int)(raw.w & 0x0F) - 8);
            results[7] = (float)((int)(raw.w >> 4) - 8);
        } else {
            for (int i = 0; i < N; ++i) {
                ulong curr_idx = offset + i;
                uchar b = ptr[curr_idx / 2];
                if (curr_idx % 2 == 0) {
                    results[i] = (float)((int)(b & 0x0F) - 8);
                } else {
                    results[i] = (float)((int)(b >> 4) - 8);
                }
            }
        }
    }
};

#endif // POLICY_Q4_0_DEFINED
