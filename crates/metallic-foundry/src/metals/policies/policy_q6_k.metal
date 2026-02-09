
#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 1
#undef METALLIC_POLICY_HAS_AFFINE
#define METALLIC_POLICY_HAS_AFFINE 0
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_Q6_K_DEFINED
#define POLICY_Q6_K_DEFINED

/**
 * Policy implementation for Foundry's decoded Q6_K layout:
 * - weights buffer stores signed int8 values for each logical weight.
 * - scales buffer stores fp16 scale per 16-weight group.
 */
struct PolicyQ6K {
    static constant bool HAS_SCALE = true;
    static constant bool HAS_AFFINE = false;
    static constant uint WEIGHTS_PER_BLOCK = 16u;
    static constant uint SCALE_BYTES = 2u;

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

    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr,
        ulong offset,
        thread float results[N]
    ) {
        const device char *w_ptr = (const device char *)ptr;

        if constexpr (N == 8) {
            uchar4 raw0 = *(const device uchar4*)(ptr + offset);
            uchar4 raw1 = *(const device uchar4*)(ptr + offset + 4);
            results[0] = (float)(char)raw0.x;
            results[1] = (float)(char)raw0.y;
            results[2] = (float)(char)raw0.z;
            results[3] = (float)(char)raw0.w;
            results[4] = (float)(char)raw1.x;
            results[5] = (float)(char)raw1.y;
            results[6] = (float)(char)raw1.z;
            results[7] = (float)(char)raw1.w;
        } else if constexpr (N == 4) {
            uchar4 raw = *(const device uchar4*)(ptr + offset);
            results[0] = (float)(char)raw.x;
            results[1] = (float)(char)raw.y;
            results[2] = (float)(char)raw.z;
            results[3] = (float)(char)raw.w;
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                results[i] = (float)w_ptr[offset + i];
            }
        }
    }
};

#endif // POLICY_Q6_K_DEFINED
