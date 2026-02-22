#ifndef METALLIC_ACTIVATIONS_METAL
#define METALLIC_ACTIVATIONS_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef ALWAYS_INLINE
#define ALWAYS_INLINE inline __attribute__((always_inline))
#endif

// =============================================================================
// Activation Components
// =============================================================================
// Each component implements:
//   static float apply(float x)
// =============================================================================

struct ActivationNone {
    ALWAYS_INLINE static float apply(float x) { return x; }
    ALWAYS_INLINE static float2 apply(float2 x) { return x; }
    ALWAYS_INLINE static float4 apply(float4 x) { return x; }
};

struct ActivationSiLU {
    ALWAYS_INLINE static float apply(float x) {
        return x / (1.0f + metal::fast::exp(-x));
    }
    ALWAYS_INLINE static float2 apply(float2 x) {
        return x / (1.0f + metal::fast::exp(-x));
    }
    ALWAYS_INLINE static float4 apply(float4 x) {
        return x / (1.0f + metal::fast::exp(-x));
    }
};

struct ActivationReLU {
    ALWAYS_INLINE static float apply(float x) {
        return max(0.0f, x);
    }
    ALWAYS_INLINE static float2 apply(float2 x) {
        return max(0.0f, x);
    }
    ALWAYS_INLINE static float4 apply(float4 x) {
        return max(0.0f, x);
    }
};

struct ActivationGELU {
    ALWAYS_INLINE static float apply(float x) {
        return 0.5f * x * (1.0f + metal::fast::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    ALWAYS_INLINE static float2 apply(float2 x) {
        return 0.5f * x * (1.0f + metal::fast::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
    ALWAYS_INLINE static float4 apply(float4 x) {
        return 0.5f * x * (1.0f + metal::fast::tanh(0.79788456f * (x + 0.044715f * x * x * x)));
    }
};

#endif // METALLIC_ACTIVATIONS_METAL
