
#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 0
#undef METALLIC_POLICY_HAS_AFFINE
#define METALLIC_POLICY_HAS_AFFINE 0
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_F32_DEFINED
#define POLICY_F32_DEFINED

/**
 * Policy implementation for native FP32 weights.
 *
 * This policy preserves dense FP32 storage and converts to float math directly.
 */
struct PolicyF32 {
    static constant bool HAS_SCALE = false;
    static constant bool HAS_AFFINE = false;
    static constant uint WEIGHTS_PER_BLOCK = 1u;
    static constant uint SCALE_BYTES = 0u;

    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        (void)scales;
        (void)block_idx;
        return 1.0h;
    }

    static ALWAYS_INLINE half load_affine(const device uchar *scales, ulong block_idx) {
        (void)scales;
        (void)block_idx;
        return 0.0h;
    }

    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr,
        ulong offset,
        thread float results[N]
    ) {
        const device float *w_ptr = (const device float *)ptr;
        if constexpr (N == 8) {
            float4 lo = *(const device float4 *)(w_ptr + offset + 0u);
            float4 hi = *(const device float4 *)(w_ptr + offset + 4u);
            results[0] = lo.x; results[1] = lo.y;
            results[2] = lo.z; results[3] = lo.w;
            results[4] = hi.x; results[5] = hi.y;
            results[6] = hi.z; results[7] = hi.w;
        } else if constexpr (N == 4) {
            float4 w = *(const device float4 *)(w_ptr + offset);
            results[0] = w.x; results[1] = w.y;
            results[2] = w.z; results[3] = w.w;
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                results[i] = w_ptr[offset + (ulong)i];
            }
        }
    }

    template<int N>
    static ALWAYS_INLINE float dot(
        const device uchar *ptr,
        ulong offset,
        float scale,
        float4 xv_lo,
        float4 xv_hi
    ) {
        const device float *w_ptr = (const device float *)ptr;
        if constexpr (N == 8) {
            float4 w_lo = *(const device float4 *)(w_ptr + offset + 0u);
            float4 w_hi = *(const device float4 *)(w_ptr + offset + 4u);
            return scale * (metal::dot(xv_lo, w_lo) + metal::dot(xv_hi, w_hi));
        } else if constexpr (N == 4) {
            float4 w = *(const device float4 *)(w_ptr + offset);
            return scale * metal::dot(xv_lo, w);
        } else {
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
    }
};

#endif // POLICY_F32_DEFINED
