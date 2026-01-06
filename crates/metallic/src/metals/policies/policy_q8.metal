#include "base.metal"

#ifndef POLICY_Q8_DEFINED
#define POLICY_Q8_DEFINED
struct PolicyQ8 {
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        // Load 2-byte scale (half)
        const device uchar *s_ptr = scales + block_idx * 2;
        ushort bits = (ushort)s_ptr[0] | ((ushort)s_ptr[1] << 8);
        return as_type<half>(bits);
    }

    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr, 
        ulong offset, 
        thread float results[N]
    ) {
        const device char *w_ptr = (const device char *)ptr;
        
        if constexpr (N == 8) {
            // Vectorized Q8 load: Load 8 bytes (chars) and dequantize to float
            // Split into two uchar4 loads since uchar8 is not a standard Metal type
            uchar4 raw0 = *(const device uchar4*)(ptr + offset);
            uchar4 raw1 = *(const device uchar4*)(ptr + offset + 4);
            
            // Dequantize: Q8 values are signed chars centered at 0
            results[0] = (float)(char)raw0.x;
            results[1] = (float)(char)raw0.y;
            results[2] = (float)(char)raw0.z;
            results[3] = (float)(char)raw0.w;
            results[4] = (float)(char)raw1.x;
            results[5] = (float)(char)raw1.y;
            results[6] = (float)(char)raw1.z;
            results[7] = (float)(char)raw1.w;
        } else if constexpr (N == 4) {
            // Load 4 bytes
            uchar4 raw = *(const device uchar4*)(ptr + offset);
            results[0] = (float)(char)raw.x;
            results[1] = (float)(char)raw.y;
            results[2] = (float)(char)raw.z;
            results[3] = (float)(char)raw.w;
        } else {
            // Scalar fallback
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                results[i] = (float)w_ptr[offset + i];
            }
        }
    }
};
#endif // POLICY_Q8_DEFINED
