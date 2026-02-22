#include <metal_stdlib>
using namespace metal;

// Buffers:
// - ring:  [window_len] u32 (most-recent token ring; written sequentially)
// - pairs: [window_len * 2] u32 packed pairs: (tok, count) per slot
// - meta:  [4] u32: { head, filled, window_len, _reserved }

// NOTE: `RepetitionStateParams` is injected by the Rust `MetalStruct` definition.

inline void dec_pair(device uint* pairs, uint window_len, uint tok) {
    const uint slots = window_len;
    for (uint i = 0; i < slots; ++i) {
        const uint base = i * 2u;
        const uint t = pairs[base + 0u];
        const uint c = pairs[base + 1u];
        if (c == 0u) {
            continue;
        }
        if (t == tok) {
            const uint nc = c - 1u;
            pairs[base + 1u] = nc;
            if (nc == 0u) {
                pairs[base + 0u] = 0u;
            }
            return;
        }
    }
}

inline void inc_pair(device uint* pairs, uint window_len, uint tok) {
    const uint slots = window_len;
    uint empty_base = 0xffffffffu;
    for (uint i = 0; i < slots; ++i) {
        const uint base = i * 2u;
        const uint c = pairs[base + 1u];
        if (c == 0u) {
            if (empty_base == 0xffffffffu) empty_base = base;
            continue;
        }
        const uint t = pairs[base + 0u];
        if (t == tok) {
            pairs[base + 1u] = c + 1u;
            return;
        }
    }
    if (empty_base != 0xffffffffu) {
        pairs[empty_base + 0u] = tok;
        pairs[empty_base + 1u] = 1u;
    }
}

inline void push_token(
    device uint* ring,
    device uint* pairs,
    device uint* meta,
    uint tok
) {
    const uint window_len = meta[2];
    if (window_len == 0u) return;

    uint head = meta[0];
    uint filled = meta[1];

    if (filled >= window_len) {
        const uint old = ring[head];
        dec_pair(pairs, window_len, old);
    } else {
        filled += 1u;
        meta[1] = filled;
    }

    ring[head] = tok;
    inc_pair(pairs, window_len, tok);

    head += 1u;
    if (head >= window_len) head = 0u;
    meta[0] = head;
}

kernel void repetition_state_init_u32(
    device uint* ring [[buffer(0)]],
    device uint* pairs [[buffer(1)]],
    device uint* meta [[buffer(2)]],
    const device uint* tokens [[buffer(3)]],
    constant RepetitionStateParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) return;
    const uint window_len = params.window_len;

    meta[0] = 0u;
    meta[1] = 0u;
    meta[2] = window_len;
    meta[3] = 0u;

    for (uint i = 0; i < window_len; ++i) {
        ring[i] = 0u;
    }
    for (uint i = 0; i < window_len; ++i) {
        const uint base = i * 2u;
        pairs[base + 0u] = 0u;
        pairs[base + 1u] = 0u;
    }

    const uint token_len = params.token_len;
    if (token_len == 0u || window_len == 0u) {
        return;
    }

    const uint take = (token_len < window_len) ? token_len : window_len;
    const uint start = token_len - take;

    // Fill ring in chronological order and update pairs.
    for (uint i = 0; i < take; ++i) {
        const uint tok = tokens[start + i];
        push_token(ring, pairs, meta, tok);
    }
}

kernel void repetition_state_ingest_u32(
    device uint* ring [[buffer(0)]],
    device uint* pairs [[buffer(1)]],
    device uint* meta [[buffer(2)]],
    const device uint* tokens [[buffer(3)]],
    constant RepetitionStateParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0u) return;
    const uint token_len = params.token_len;
    for (uint i = 0; i < token_len; ++i) {
        push_token(ring, pairs, meta, tokens[i]);
    }
}

kernel void repetition_state_update_from_token_u32(
    device uint* ring [[buffer(0)]],
    device uint* pairs [[buffer(1)]],
    device uint* meta [[buffer(2)]],
    const device uint* token_buf [[buffer(3)]],
    constant RepetitionStateParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    (void)params;
    if (gid != 0u) return;
    push_token(ring, pairs, meta, token_buf[0]);
}
