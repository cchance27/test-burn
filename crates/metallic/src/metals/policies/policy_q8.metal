#include "base.metal"

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
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            results[i] = (float)w_ptr[offset + i];
        }
    }
};
