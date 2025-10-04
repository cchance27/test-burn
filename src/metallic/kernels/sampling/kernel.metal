#include <metal_stdlib>
using namespace metal;

constant uint MAX_TOP_K = 256;

struct SamplingParams {
    uint vocab_size;
    uint top_k;
    float top_p;
    float temperature;
    uint random_u32;
    uint _padding;
};

kernel void sample_top_k_top_p_f32(
    device const float* logits [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant SamplingParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) {
        return;
    }

    uint vocab_size = params.vocab_size;
    if (vocab_size == 0) {
        output[0] = 0u;
        return;
    }

    float temperature = params.temperature;
    uint requested_top_k = params.top_k;

    uint fallback_idx = 0u;
    bool fallback_found = false;
    float fallback_val = -INFINITY;
    for (uint i = 0; i < vocab_size; ++i) {
        float val = logits[i];
        if (isfinite(val) && (!fallback_found || val > fallback_val || (val == fallback_val && i > fallback_idx))) {
            fallback_idx = i;
            fallback_val = val;
            fallback_found = true;
        }
    }

    if (!isfinite(temperature) || temperature <= 0.0f) {
        output[0] = fallback_found ? fallback_idx : 0u;
        return;
    }

    if (requested_top_k == 0u) {
        output[0] = fallback_found ? fallback_idx : 0u;
        return;
    }

    uint effective_top_k = requested_top_k;
    if (effective_top_k < 1u) {
        effective_top_k = 1u;
    }
    if (effective_top_k > vocab_size) {
        effective_top_k = vocab_size;
    }
    if (effective_top_k > MAX_TOP_K) {
        output[0] = fallback_found ? fallback_idx : 0u;
        return;
    }

    thread float scaled[MAX_TOP_K];
    thread uint indices[MAX_TOP_K];
    uint count = 0u;
    float inv_temp = 1.0f / temperature;

    for (uint i = 0; i < vocab_size; ++i) {
        float scaled_val = logits[i] * inv_temp;
        if (!isfinite(scaled_val)) {
            continue;
        }

        uint insert_pos = 0u;
        while (insert_pos < count && scaled[insert_pos] > scaled_val) {
            ++insert_pos;
        }

        if (count < effective_top_k) {
            for (uint j = count; j > insert_pos; --j) {
                scaled[j] = scaled[j - 1];
                indices[j] = indices[j - 1];
            }
            scaled[insert_pos] = scaled_val;
            indices[insert_pos] = i;
            ++count;
        } else if (insert_pos < effective_top_k) {
            for (uint j = effective_top_k - 1u; j > insert_pos; --j) {
                scaled[j] = scaled[j - 1u];
                indices[j] = indices[j - 1u];
            }
            scaled[insert_pos] = scaled_val;
            indices[insert_pos] = i;
        }
    }

    if (count == 0u) {
        output[0] = fallback_found ? fallback_idx : 0u;
        return;
    }

    float max_val = scaled[0];
    float total = 0.0f;
    bool has_positive = false;
    for (uint i = 0; i < count; ++i) {
        float val = scaled[i];
        float exp_val = isfinite(val) ? exp(val - max_val) : 0.0f;
        if (exp_val > 1e10f) {
            exp_val = 1e10f;
        } else if (exp_val < 1e-10f) {
            exp_val = 0.0f;
        }
        scaled[i] = exp_val;
        total += exp_val;
        has_positive = has_positive || exp_val > 0.0f;
    }

    if (!has_positive || total <= 0.0f || !isfinite(total)) {
        output[0] = fallback_found ? fallback_idx : 0u;
        return;
    }

    float normalized_top_p = params.top_p;
    if (!isfinite(normalized_top_p)) {
        normalized_top_p = 1.0f;
    } else {
        normalized_top_p = clamp(normalized_top_p, 0.0f, 1.0f);
    }

    uint cutoff = count - 1u;
    if (normalized_top_p <= 0.0f) {
        cutoff = 0u;
    } else if (normalized_top_p < 1.0f) {
        float cumulative = 0.0f;
        float threshold = normalized_top_p * total;
        for (uint i = 0; i < count; ++i) {
            cumulative += scaled[i];
            cutoff = i;
            if (cumulative >= threshold || !isfinite(cumulative)) {
                break;
            }
        }
    }

    float shortlist_total = 0.0f;
    for (uint i = 0; i <= cutoff; ++i) {
        shortlist_total += scaled[i];
    }

    if (shortlist_total <= 0.0f || !isfinite(shortlist_total)) {
        output[0] = indices[0];
        return;
    }

    for (uint i = 0; i <= cutoff; ++i) {
        scaled[i] /= shortlist_total;
    }

    float random_value = static_cast<float>(params.random_u32) / 4294967295.0f;
    float acc = 0.0f;
    for (uint i = 0; i <= cutoff; ++i) {
        acc += scaled[i];
        if (random_value <= acc || !isfinite(acc)) {
            output[0] = indices[i];
            return;
        }
    }

    output[0] = indices[cutoff];
}
