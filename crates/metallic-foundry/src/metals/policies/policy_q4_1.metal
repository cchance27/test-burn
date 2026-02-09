
#undef METALLIC_POLICY_HAS_SCALE
#define METALLIC_POLICY_HAS_SCALE 1
#undef METALLIC_POLICY_HAS_AFFINE
#define METALLIC_POLICY_HAS_AFFINE 1
#undef METALLIC_POLICY_WEIGHTS_FP16
#define METALLIC_POLICY_WEIGHTS_FP16 0

#ifndef POLICY_Q4_1_DEFINED
#define POLICY_Q4_1_DEFINED

/**
 * Policy implementation for GGUF Q4_1 (4-bit affine quantized) weights.
 *
 * Format:
 * - Weights are packed 4-bit values (2 per byte), logical range [0, 15].
 * - Scales plane stores fp16 `d` and fp16 `m` per 32-weight block.
 * - Dequantization is affine: w = d * q + m.
 */
struct PolicyQ4_1 {
    static constant bool HAS_SCALE = true;
    static constant bool HAS_AFFINE = true;
    static constant uint WEIGHTS_PER_BLOCK = 32u;
    static constant uint SCALE_BYTES = 4u;

    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        const device uchar *s_ptr = scales + block_idx * SCALE_BYTES;
        ushort bits = (ushort)s_ptr[0] | ((ushort)s_ptr[1] << 8);
        return as_type<half>(bits);
    }

    static ALWAYS_INLINE half load_affine(const device uchar *scales, ulong block_idx) {
        const device uchar *s_ptr = scales + block_idx * SCALE_BYTES + 2;
        ushort bits = (ushort)s_ptr[0] | ((ushort)s_ptr[1] << 8);
        return as_type<half>(bits);
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
        if constexpr (N == 8 && true) {
            const device uchar4 *w_ptr = (const device uchar4 *)(ptr + (offset / 2));
            uchar4 raw = *w_ptr;

            results[0] = (float)(raw.x & 0x0F);
            results[1] = (float)(raw.x >> 4);
            results[2] = (float)(raw.y & 0x0F);
            results[3] = (float)(raw.y >> 4);
            results[4] = (float)(raw.z & 0x0F);
            results[5] = (float)(raw.z >> 4);
            results[6] = (float)(raw.w & 0x0F);
            results[7] = (float)(raw.w >> 4);
        } else {
            for (int i = 0; i < N; ++i) {
                ulong curr_idx = offset + i;
                uchar b = ptr[curr_idx / 2];
                if (curr_idx % 2 == 0) {
                    results[i] = (float)(b & 0x0F);
                } else {
                    results[i] = (float)(b >> 4);
                }
            }
        }
    }
};

#endif // POLICY_Q4_1_DEFINED
