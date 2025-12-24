#ifndef POLICY_BASE_METAL
#define POLICY_BASE_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE __attribute__((always_inline))
#endif

/// Common interface for Loader Policies.
/// Specific policies (F16, Q8_0, etc.) must implement these methods.
///
/// Implementations must provide:
///   static half load_scale(const device uchar *scales, ulong block_idx)
///
///   template<int N>
///   static ALWAYS_INLINE void load_weights(
///       const device uchar *ptr, 
///       ulong offset, 
///       thread float results[N]
///   );

#endif // POLICY_BASE_METAL
