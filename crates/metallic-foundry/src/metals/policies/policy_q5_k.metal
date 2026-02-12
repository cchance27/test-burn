
#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 1
#undef METALLIC_POLICY_HAS_AFFINE
#define METALLIC_POLICY_HAS_AFFINE 1
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_Q5_K_DEFINED
#define POLICY_Q5_K_DEFINED

/**
 * Policy implementation for Foundry's decoded Q5_K layout:
 * - weights buffer stores uint8 values in [0,31] for each logical weight.
 * - scales buffer stores fp32 scale + fp32 affine per 16-weight group.
 */
struct PolicyQ5K {
    static constant bool HAS_SCALE = true;
    static constant bool HAS_AFFINE = true;
    static constant uint WEIGHTS_PER_BLOCK = 16u;
    static constant uint SCALE_BYTES = 8u;

    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        const device uchar *s_ptr = scales + block_idx * SCALE_BYTES;
        uint bits = (uint)s_ptr[0]
            | ((uint)s_ptr[1] << 8)
            | ((uint)s_ptr[2] << 16)
            | ((uint)s_ptr[3] << 24);
        return half(as_type<float>(bits));
    }

    static ALWAYS_INLINE half load_affine(const device uchar *scales, ulong block_idx) {
        const device uchar *s_ptr = scales + block_idx * SCALE_BYTES + 4;
        uint bits = (uint)s_ptr[0]
            | ((uint)s_ptr[1] << 8)
            | ((uint)s_ptr[2] << 16)
            | ((uint)s_ptr[3] << 24);
        return half(as_type<float>(bits));
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
        const device uchar *w_ptr = ptr;

        if constexpr (N == 8) {
            // Batched prefill (M > 1) can hit offsets that are not 4-byte aligned.
            // Guard vector loads to avoid undefined behavior on unaligned uchar4 casts.
            if ((offset & 0x3ul) == 0ul) {
                uchar4 raw0 = *(const device uchar4*)(w_ptr + offset);
                uchar4 raw1 = *(const device uchar4*)(w_ptr + offset + 4);
                results[0] = (float)raw0.x;
                results[1] = (float)raw0.y;
                results[2] = (float)raw0.z;
                results[3] = (float)raw0.w;
                results[4] = (float)raw1.x;
                results[5] = (float)raw1.y;
                results[6] = (float)raw1.z;
                results[7] = (float)raw1.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    results[i] = (float)w_ptr[offset + (ulong)i];
                }
            }
        } else if constexpr (N == 4) {
            if ((offset & 0x3ul) == 0ul) {
                uchar4 raw = *(const device uchar4*)(w_ptr + offset);
                results[0] = (float)raw.x;
                results[1] = (float)raw.y;
                results[2] = (float)raw.z;
                results[3] = (float)raw.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    results[i] = (float)w_ptr[offset + (ulong)i];
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                results[i] = (float)w_ptr[offset + (ulong)i];
            }
        }
    }
};

#endif // POLICY_Q5_K_DEFINED
