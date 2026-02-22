#include <metal_stdlib>
using namespace metal;

kernel void apply_repetition_penalty_f16(
    device half* logits [[buffer(0)]],
    // Packed pairs: [tok_0, count_0, tok_1, count_1, ...]
    const device uint* recent_pairs [[buffer(1)]],
    constant RepetitionPenaltyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const float penalty = params.repeat_penalty;
    const float presence_penalty = params.presence_penalty;
    const float frequency_penalty = params.frequency_penalty;
    const bool has_repeat = (penalty > 1.0f) && isfinite(penalty);
    const bool has_presence = (presence_penalty > 0.0f) && isfinite(presence_penalty);
    const bool has_frequency = (frequency_penalty > 0.0f) && isfinite(frequency_penalty);
    if (!(has_repeat || has_presence || has_frequency)) {
        return;
    }

    // Number of (tok,count) pairs.
    const uint pair_len = params.recent_len;
    if (gid >= pair_len) {
        return;
    }

    const uint base = gid * 2u;
    const uint tok = recent_pairs[base + 0u];
    const uint count = recent_pairs[base + 1u];
    if (tok >= params.vocab_size) {
        return;
    }
    if (count == 0u) {
        return;
    }

    float v = static_cast<float>(logits[tok]);
    if (has_repeat) {
        const float scale = powr(penalty, static_cast<float>(count));
        v = (v > 0.0f) ? (v / scale) : (v * scale);
    }
    // Additive penalties:
    // - presence_penalty applies once if token appears at all
    // - frequency_penalty scales with count in the window
    if (has_presence) {
        v -= presence_penalty;
    }
    if (has_frequency) {
        v -= frequency_penalty * static_cast<float>(count);
    }
    logits[tok] = static_cast<half>(v);
}

