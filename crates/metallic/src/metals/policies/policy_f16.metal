#include "base.metal"

#ifndef POLICY_F16_DEFINED
#define POLICY_F16_DEFINED
struct PolicyF16 {
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        return 1.0h;
    }

    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr, 
        ulong offset, 
        thread float results[N]
    ) {
        const device half *w_ptr = (const device half *)ptr;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            results[i] = (float)w_ptr[offset + i];
        }
    }
};
#endif // POLICY_F16_DEFINED
