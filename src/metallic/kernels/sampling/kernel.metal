#include <metal_stdlib>
using namespace metal;

struct SamplerConfig {
    uint vocab_size;
    uint top_k;
    float top_p;
    float temperature;
    uint rng_seed;
};

struct SamplerResult {
    uint selected;
    uint fallback;
    uint used_fallback;
    uint padding;
};

static inline uint next_state(uint state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state * 1664525u + 1013904223u;
}

static inline float uniform_from_state(uint state) {
    return (float)(state & 0xFFFFFFFFu) / 4294967295.0f;
}

static inline void initialize_result(thread SamplerResult &result, uint fallback_idx) {
    result.selected = fallback_idx;
    result.fallback = fallback_idx;
    result.used_fallback = 1u;
    result.padding = 0u;
}

static inline void finalize_result(
    thread SamplerResult &result,
    uint selected_idx,
    uint fallback_idx,
    bool used_fallback) {
    result.selected = selected_idx;
    result.fallback = fallback_idx;
    result.used_fallback = used_fallback ? 1u : 0u;
    result.padding = 0u;
}

static inline float clamp_exp(float value) {
    if (value > 1e10f) {
        return 1e10f;
    }
    if (value < 1e-10f) {
        return 0.0f;
    }
    return value;
}

template <typename Loader>
static inline void sample_impl(
    Loader loader,
    SamplerConfig cfg,
    device float *shortlist_values,
    device uint *shortlist_indices,
    device SamplerResult *out_result) {
    SamplerResult result;
    uint fallback_idx = 0u;
    float fallback_val = -INFINITY;
    bool fallback_found = false;

    for (uint i = 0u; i < cfg.vocab_size; ++i) {
        float val = loader(i);
        if (!isfinite(val)) {
            continue;
        }
        if (!fallback_found || val > fallback_val || (val == fallback_val && i > fallback_idx)) {
            fallback_idx = i;
            fallback_val = val;
            fallback_found = true;
        }
    }

    if (cfg.vocab_size == 0u || cfg.top_k == 0u || cfg.temperature <= 0.0f || !isfinite(cfg.temperature)) {
        initialize_result(result, fallback_found ? fallback_idx : 0u);
        out_result[0] = result;
        return;
    }

    uint effective_top_k = cfg.top_k;
    if (effective_top_k < 1u) {
        effective_top_k = 1u;
    }
    if (effective_top_k > cfg.vocab_size) {
        effective_top_k = cfg.vocab_size;
    }

    uint shortlist_len = 0u;
    for (uint i = 0u; i < cfg.vocab_size; ++i) {
        float scaled_val = loader(i) / cfg.temperature;
        if (!isfinite(scaled_val)) {
            continue;
        }

        uint insert_pos = 0u;
        while (insert_pos < shortlist_len && shortlist_values[insert_pos] > scaled_val) {
            ++insert_pos;
        }

        if (shortlist_len < effective_top_k) {
            for (uint j = shortlist_len; j > insert_pos; --j) {
                shortlist_values[j] = shortlist_values[j - 1u];
                shortlist_indices[j] = shortlist_indices[j - 1u];
            }
            shortlist_values[insert_pos] = scaled_val;
            shortlist_indices[insert_pos] = i;
            ++shortlist_len;
        } else if (insert_pos < effective_top_k) {
            for (uint j = effective_top_k - 1u; j > insert_pos; --j) {
                shortlist_values[j] = shortlist_values[j - 1u];
                shortlist_indices[j] = shortlist_indices[j - 1u];
            }
            shortlist_values[insert_pos] = scaled_val;
            shortlist_indices[insert_pos] = i;
        }
    }

    if (shortlist_len == 0u) {
        initialize_result(result, fallback_found ? fallback_idx : 0u);
        out_result[0] = result;
        return;
    }

    float max_val = shortlist_values[0];
    float total = 0.0f;
    bool has_positive = false;
    for (uint i = 0u; i < shortlist_len; ++i) {
        float exp_val = clamp_exp(exp(shortlist_values[i] - max_val));
        shortlist_values[i] = exp_val;
        total += exp_val;
        has_positive = has_positive || (exp_val > 0.0f);
    }

    if (!has_positive || total <= 0.0f || !isfinite(total)) {
        initialize_result(result, fallback_found ? fallback_idx : shortlist_indices[0]);
        out_result[0] = result;
        return;
    }

    float normalized_top_p = isfinite(cfg.top_p) ? clamp(cfg.top_p, 0.0f, 1.0f) : 1.0f;
    uint cutoff = shortlist_len - 1u;
    if (normalized_top_p <= 0.0f) {
        cutoff = 0u;
    } else if (normalized_top_p < 1.0f) {
        float cum = 0.0f;
        float threshold = normalized_top_p * total;
        for (uint i = 0u; i < shortlist_len; ++i) {
            cum += shortlist_values[i];
            cutoff = i;
            if (cum >= threshold || !isfinite(cum)) {
                break;
            }
        }
    }

    if (cutoff + 1u < shortlist_len) {
        shortlist_len = cutoff + 1u;
    }

    float shortlist_total = 0.0f;
    for (uint i = 0u; i < shortlist_len; ++i) {
        shortlist_total += shortlist_values[i];
    }

    if (shortlist_total <= 0.0f || !isfinite(shortlist_total)) {
        initialize_result(result, shortlist_indices[0]);
        out_result[0] = result;
        return;
    }

    for (uint i = 0u; i < shortlist_len; ++i) {
        shortlist_values[i] /= shortlist_total;
    }

    uint state = cfg.rng_seed | 1u;
    state = next_state(state);
    float r = uniform_from_state(state);
    float acc = 0.0f;
    for (uint i = 0u; i < shortlist_len; ++i) {
        acc += shortlist_values[i];
        if (r <= acc || !isfinite(acc) || i == shortlist_len - 1u) {
            finalize_result(result, shortlist_indices[i], fallback_found ? fallback_idx : shortlist_indices[i], false);
            out_result[0] = result;
            return;
        }
    }

    finalize_result(
        result,
        shortlist_indices[shortlist_len - 1u],
        fallback_found ? fallback_idx : shortlist_indices[shortlist_len - 1u],
        false);
    out_result[0] = result;
}

struct F32Loader {
    device const float *ptr;
    float operator()(uint idx) const {
        return ptr[idx];
    }
};

struct F16Loader {
    device const half *ptr;
    float operator()(uint idx) const {
        return (float)ptr[idx];
    }
};

kernel void sample_top_k_top_p_f32(
    device const float *logits [[buffer(0)]],
    device float *shortlist_values [[buffer(1)]],
    device uint *shortlist_indices [[buffer(2)]],
    device SamplerResult *out_result [[buffer(3)]],
    constant SamplerConfig &cfg [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }

    F32Loader loader{logits};
    sample_impl(loader, cfg, shortlist_values, shortlist_indices, out_result);
}

kernel void sample_top_k_top_p_f16(
    device const half *logits [[buffer(0)]],
    device float *shortlist_values [[buffer(1)]],
    device uint *shortlist_indices [[buffer(2)]],
    device SamplerResult *out_result [[buffer(3)]],
    constant SamplerConfig &cfg [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }

    F16Loader loader{logits};
    sample_impl(loader, cfg, shortlist_values, shortlist_indices, out_result);
}
