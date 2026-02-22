#ifndef METALLIC_STREAMING_SOFTMAX_METAL
#define METALLIC_STREAMING_SOFTMAX_METAL

#include <metal_stdlib>

using namespace metal;

// =============================================================================
// Streaming Softmax (Online) primitives
// =============================================================================
//
// This implements a numerically stable online softmax update:
//   m_new = max(m_prev, x)
//   l_new = l_prev * exp(m_prev - m_new) + exp(x - m_new)
//
// This is the core primitive used by FlashAttention-style kernels to avoid
// materializing the full score matrix.
//

struct StreamingSoftmaxState {
    float m; // running max
    float l; // running sum exp
};

METAL_FUNC inline StreamingSoftmaxState streaming_softmax_init() {
    StreamingSoftmaxState s;
    s.m = -1e30f;
    s.l = 0.0f;
    return s;
}

// Update softmax state with a new scalar `x`.
// Returns (state, exp(x - m_new), exp(m_prev - m_new)).
METAL_FUNC inline void streaming_softmax_update(
    thread StreamingSoftmaxState& state,
    float x,
    thread float& out_scale_x,
    thread float& out_scale_prev
) {
    float m_prev = state.m;
    float m_new = max(m_prev, x);

    // Use fast exp; callers can switch to precise exp if they need higher accuracy.
    float scale_x = metal::fast::exp(x - m_new);
    float scale_prev = metal::fast::exp(m_prev - m_new);

    state.l = state.l * scale_prev + scale_x;
    state.m = m_new;

    out_scale_x = scale_x;
    out_scale_prev = scale_prev;
}

#endif // METALLIC_STREAMING_SOFTMAX_METAL
