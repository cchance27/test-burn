#include "base.metal"

#ifndef POLICY_F16_DEFINED
#define POLICY_F16_DEFINED

/**
 * Policy implementation for standard FP16 weights.
 * 
 * Since FP16 is not quantized (uncompressed), this policy simply performs 
 * straight memory loads and casts them to the internal float representation.
 */
struct PolicyF16 {
    // F16 does not use block scales. Always returns unity.
    static ALWAYS_INLINE half load_scale(const device uchar *scales, ulong block_idx) {
        return 1.0h;
    }

    /**
     * Load and expand F16 weights to floats.
     * 
     * Handles vectorized loads for N=4 and N=8 to maximize memory bandwidth utilization.
     */
    template<int N>
    static ALWAYS_INLINE void load_weights(
        const device uchar *ptr, 
        ulong offset, 
        thread float results[N]
    ) {
        // Internal cast to half pointers for standard F16 access
        const device half *w_ptr = (const device half *)ptr;
        
        if constexpr (N == 8) {
            // Optimized: Load 16 bytes (8 halves) via float4 cast.
            // This ensures a single 128-bit memory transaction.
            float4 raw = *(const device float4*)(w_ptr + offset);
            half4 lo = as_type<half4>(raw.xy);
            half4 hi = as_type<half4>(raw.zw);
            results[0] = (float)lo.x; results[1] = (float)lo.y;
            results[2] = (float)lo.z; results[3] = (float)lo.w;
            results[4] = (float)hi.x; results[5] = (float)hi.y;
            results[6] = (float)hi.z; results[7] = (float)hi.w;
        } else if constexpr (N == 4) {
            // Optimized: Load 8 bytes (4 halves) via half4 cast.
            half4 h = *(const device half4*)(w_ptr + offset);
            results[0] = (float)h.x; results[1] = (float)h.y;
            results[2] = (float)h.z; results[3] = (float)h.w;
        } else {
            // General purpose unrolled loop for non-power-of-two or small loads.
            #pragma unroll
            for (int i = 0; i < N; ++i) {
                results[i] = (float)w_ptr[offset + i];
            }
        }
    }
};

#endif // POLICY_F16_DEFINED
