
#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 0
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_U32_DEFINED
#define POLICY_U32_DEFINED

/**
 * Policy implementation for U32 (uint) tensors.
 * 
 * Performs direct memory loads and casts them to float for generic kernel compatibility.
 */
struct PolicyU32 {
    static constant bool HAS_SCALE = false;
    static constant uint WEIGHTS_PER_BLOCK = 1u;

    // U32 does not use scales.
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        return 1.0h;
    }

    /**
     * Load and expand U32 weights to floats.
     */
    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr,
        ulong offset, 
        thread float results[N]
    ) {
        const device uint *w_ptr = (const device uint *)ptr;
        
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            results[i] = (float)w_ptr[offset + i];
        }
    }

    // Specialized dot product for U32 (rarely used for compute, but needed for trait completeness)
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
};

#endif // POLICY_U32_DEFINED
