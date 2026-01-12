#ifndef POLICY_BASE_METAL
#define POLICY_BASE_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/*
 * Common interface for Quantization Policy implementations.
 * 
 * A Policy structure must provide static methods for loading scales and dequantizing weights.
 * This decoupled approach allows kernels (like GEMM or GEMV) to remain agnostic of the underlying
 * quantization format (F16, Q8_0, etc.).
 */

/**
 * Interface Requirements:
 * 
 * struct Policy {
 *     // Load a singular scale factor for a given block.
 *     // - scales: Pointer to the scales buffer.
 *     // - block_idx: Logical block index within the weight matrix.
 *     static half load_scale(const device uchar *scales, ulong block_idx);
 *
 *     // Load and dequantize N weights at once into thread-local results.
 *     // - ptr: Pointer to the quantized data buffer.
 *     // - offset: Starting index for the load (in terms of quantized units).
 *     // - results: Pre-allocated thread-local array to store dequantized float values.
 *     template<int N>
 *     static ALWAYS_INLINE void load_weights(
 *         const device uchar *ptr, 
 *         ulong offset, 
 *         thread float results[N]
 *     );
 * };
 */

#endif // POLICY_BASE_METAL
