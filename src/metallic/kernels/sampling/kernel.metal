#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint MAX_TOP_K = 256;
constant uint THREADGROUP_SIZE = 32;

struct SamplingParams {
    uint vocab_size;
    uint top_k;
    uint random_u32;
    uint threadgroup_count;
    float top_p;
    float temperature;
    uint _padding0;
    uint _padding1;
};

kernel void sample_top_k_top_p_f32(
    device const float* logits [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device float* partial_vals [[buffer(2)]],
    device uint* partial_indices [[buffer(3)]],
    device uint* partial_counts [[buffer(4)]],
    device float* fallback_vals [[buffer(5)]],
    device uint* fallback_indices [[buffer(6)]],
    device uint* fallback_flags [[buffer(7)]],
    device atomic_uint* completion_counter [[buffer(8)]],
    constant SamplingParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]])
{
    uint vocab_size = params.vocab_size;
    if (vocab_size == 0u) {
        output[0] = 0u;
        return;
    }

    uint threadgroup_index = tg_pos.x;
    uint threadgroup_count = max(params.threadgroup_count, 1u);
    if (threadgroup_index >= threadgroup_count) {
        return;
    }

    uint total_threads = threadgroup_count * THREADGROUP_SIZE;
    uint global_thread = threadgroup_index * THREADGROUP_SIZE + tid;

    uint requested_top_k = params.top_k;
    float temperature = params.temperature;
    bool skip_sampling = false;
    if (!isfinite(temperature) || temperature <= 0.0f) {
        skip_sampling = true;
    }
    if (requested_top_k == 0u) {
        skip_sampling = true;
    }

    uint effective_top_k = requested_top_k;
    if (effective_top_k < 1u) {
        effective_top_k = 1u;
    }
    if (effective_top_k > vocab_size) {
        effective_top_k = vocab_size;
    }
    if (effective_top_k > MAX_TOP_K) {
        effective_top_k = MAX_TOP_K;
        skip_sampling = true;
    }

    threadgroup float shared_vals[THREADGROUP_SIZE * MAX_TOP_K];
    threadgroup uint shared_indices[THREADGROUP_SIZE * MAX_TOP_K];
    threadgroup uint shared_counts[THREADGROUP_SIZE];
    threadgroup float shared_fallback_vals[THREADGROUP_SIZE];
    threadgroup uint shared_fallback_indices[THREADGROUP_SIZE];
    threadgroup uint shared_fallback_flags[THREADGROUP_SIZE];

    thread float local_vals[MAX_TOP_K];
    thread uint local_indices[MAX_TOP_K];
    uint local_count = 0u;
    bool local_fallback_found = false;
    float local_fallback_val = -INFINITY;
    uint local_fallback_idx = 0u;
    float inv_temp = skip_sampling ? 0.0f : 1.0f / temperature;

    for (uint index = global_thread; index < vocab_size; index += total_threads) {
        float logit = logits[index];
        if (isfinite(logit) && (!local_fallback_found || logit > local_fallback_val ||
                                (logit == local_fallback_val && index > local_fallback_idx))) {
            local_fallback_found = true;
            local_fallback_val = logit;
            local_fallback_idx = index;
        }

        if (skip_sampling) {
            continue;
        }

        float scaled_val = logit * inv_temp;
        if (!isfinite(scaled_val)) {
            continue;
        }

        uint insert_pos = 0u;
        while (insert_pos < local_count && local_vals[insert_pos] > scaled_val) {
            ++insert_pos;
        }

        if (local_count < effective_top_k) {
            for (uint j = local_count; j > insert_pos; --j) {
                local_vals[j] = local_vals[j - 1u];
                local_indices[j] = local_indices[j - 1u];
            }
            local_vals[insert_pos] = scaled_val;
            local_indices[insert_pos] = index;
            ++local_count;
        } else if (insert_pos < effective_top_k) {
            for (uint j = effective_top_k - 1u; j > insert_pos; --j) {
                local_vals[j] = local_vals[j - 1u];
                local_indices[j] = local_indices[j - 1u];
            }
            local_vals[insert_pos] = scaled_val;
            local_indices[insert_pos] = index;
        }
    }

    shared_counts[tid] = local_count;
    shared_fallback_vals[tid] = local_fallback_val;
    shared_fallback_indices[tid] = local_fallback_idx;
    shared_fallback_flags[tid] = local_fallback_found ? 1u : 0u;

    uint shared_base = tid * MAX_TOP_K;
    for (uint i = 0u; i < local_count; ++i) {
        shared_vals[shared_base + i] = local_vals[i];
        shared_indices[shared_base + i] = local_indices[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        bool fallback_found = false;
        float fallback_val = -INFINITY;
        uint fallback_idx = 0u;
        for (uint t = 0u; t < THREADGROUP_SIZE; ++t) {
            if (shared_fallback_flags[t] == 0u) {
                continue;
            }
            float candidate_val = shared_fallback_vals[t];
            uint candidate_idx = shared_fallback_indices[t];
            if (!fallback_found || candidate_val > fallback_val ||
                (candidate_val == fallback_val && candidate_idx > fallback_idx)) {
                fallback_found = true;
                fallback_val = candidate_val;
                fallback_idx = candidate_idx;
            }
        }

        thread float shortlist_vals[MAX_TOP_K];
        thread uint shortlist_indices[MAX_TOP_K];
        uint shortlist_count = 0u;

        for (uint t = 0u; t < THREADGROUP_SIZE; ++t) {
            uint count = shared_counts[t];
            uint offset = t * MAX_TOP_K;
            for (uint i = 0u; i < count; ++i) {
                float val = shared_vals[offset + i];
                uint idx = shared_indices[offset + i];

                uint insert_pos = 0u;
                while (insert_pos < shortlist_count && shortlist_vals[insert_pos] > val) {
                    ++insert_pos;
                }

                if (shortlist_count < effective_top_k) {
                    for (uint j = shortlist_count; j > insert_pos; --j) {
                        shortlist_vals[j] = shortlist_vals[j - 1u];
                        shortlist_indices[j] = shortlist_indices[j - 1u];
                    }
                    shortlist_vals[insert_pos] = val;
                    shortlist_indices[insert_pos] = idx;
                    ++shortlist_count;
                } else if (insert_pos < effective_top_k) {
                    for (uint j = effective_top_k - 1u; j > insert_pos; --j) {
                        shortlist_vals[j] = shortlist_vals[j - 1u];
                        shortlist_indices[j] = shortlist_indices[j - 1u];
                    }
                    shortlist_vals[insert_pos] = val;
                    shortlist_indices[insert_pos] = idx;
                }
            }
        }

        uint base = threadgroup_index * effective_top_k;
        for (uint i = 0u; i < shortlist_count; ++i) {
            partial_vals[base + i] = shortlist_vals[i];
            partial_indices[base + i] = shortlist_indices[i];
        }
        partial_counts[threadgroup_index] = shortlist_count;
        fallback_vals[threadgroup_index] = fallback_val;
        fallback_indices[threadgroup_index] = fallback_idx;
        fallback_flags[threadgroup_index] = fallback_found ? 1u : 0u;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        uint prev = atomic_fetch_add_explicit(completion_counter, 1u, memory_order_relaxed);
        bool is_last = (prev + 1u) == threadgroup_count;
        if (!is_last) {
            return;
        }

        bool fallback_found = false;
        float fallback_val = -INFINITY;
        uint fallback_idx = 0u;
        for (uint group = 0u; group < threadgroup_count; ++group) {
            if (fallback_flags[group] == 0u) {
                continue;
            }
            float candidate_val = fallback_vals[group];
            uint candidate_idx = fallback_indices[group];
            if (!fallback_found || candidate_val > fallback_val ||
                (candidate_val == fallback_val && candidate_idx > fallback_idx)) {
                fallback_found = true;
                fallback_val = candidate_val;
                fallback_idx = candidate_idx;
            }
        }

        if (skip_sampling) {
            output[0] = fallback_found ? fallback_idx : 0u;
            atomic_store_explicit(completion_counter, 0u, memory_order_relaxed);
            return;
        }

        thread float shortlist_vals[MAX_TOP_K];
        thread uint shortlist_indices[MAX_TOP_K];
        uint shortlist_count = 0u;

        for (uint group = 0u; group < threadgroup_count; ++group) {
            uint count = min(partial_counts[group], effective_top_k);
            uint base = group * effective_top_k;
            for (uint i = 0u; i < count; ++i) {
                float val = partial_vals[base + i];
                uint idx = partial_indices[base + i];

                uint insert_pos = 0u;
                while (insert_pos < shortlist_count && shortlist_vals[insert_pos] > val) {
                    ++insert_pos;
                }

                if (shortlist_count < effective_top_k) {
                    for (uint j = shortlist_count; j > insert_pos; --j) {
                        shortlist_vals[j] = shortlist_vals[j - 1u];
                        shortlist_indices[j] = shortlist_indices[j - 1u];
                    }
                    shortlist_vals[insert_pos] = val;
                    shortlist_indices[insert_pos] = idx;
                    ++shortlist_count;
                } else if (insert_pos < effective_top_k) {
                    for (uint j = effective_top_k - 1u; j > insert_pos; --j) {
                        shortlist_vals[j] = shortlist_vals[j - 1u];
                        shortlist_indices[j] = shortlist_indices[j - 1u];
                    }
                    shortlist_vals[insert_pos] = val;
                    shortlist_indices[insert_pos] = idx;
                }
            }
        }

        if (shortlist_count == 0u) {
            output[0] = fallback_found ? fallback_idx : 0u;
            atomic_store_explicit(completion_counter, 0u, memory_order_relaxed);
            return;
        }

        float max_val = shortlist_vals[0];
        for (uint i = 1u; i < shortlist_count; ++i) {
            if (shortlist_vals[i] > max_val) {
                max_val = shortlist_vals[i];
            }
        }

        float total = 0.0f;
        bool has_positive = false;
        for (uint i = 0u; i < shortlist_count; ++i) {
            float exp_val = exp(shortlist_vals[i] - max_val);
            if (exp_val > 1e10f) {
                exp_val = 1e10f;
            } else if (exp_val < 1e-10f) {
                exp_val = 0.0f;
            }
            shortlist_vals[i] = exp_val;
            total += exp_val;
            has_positive = has_positive || exp_val > 0.0f;
        }

        if (!has_positive || total <= 0.0f || !isfinite(total)) {
            output[0] = fallback_found ? fallback_idx : shortlist_indices[0];
            atomic_store_explicit(completion_counter, 0u, memory_order_relaxed);
            return;
        }

        float normalized_top_p = params.top_p;
        if (!isfinite(normalized_top_p)) {
            normalized_top_p = 1.0f;
        } else {
            normalized_top_p = clamp(normalized_top_p, 0.0f, 1.0f);
        }

        uint cutoff = shortlist_count - 1u;
        if (normalized_top_p <= 0.0f) {
            cutoff = 0u;
        } else if (normalized_top_p < 1.0f) {
            float cumulative = 0.0f;
            float threshold = normalized_top_p * total;
            for (uint i = 0u; i < shortlist_count; ++i) {
                cumulative += shortlist_vals[i];
                cutoff = i;
                if (cumulative >= threshold || !isfinite(cumulative)) {
                    break;
                }
            }
        }

        float shortlist_total = 0.0f;
        for (uint i = 0u; i <= cutoff; ++i) {
            shortlist_total += shortlist_vals[i];
        }

        if (shortlist_total <= 0.0f || !isfinite(shortlist_total)) {
            output[0] = shortlist_indices[0];
            atomic_store_explicit(completion_counter, 0u, memory_order_relaxed);
            return;
        }

        for (uint i = 0u; i <= cutoff; ++i) {
            shortlist_vals[i] /= shortlist_total;
        }

        float random_value = static_cast<float>(params.random_u32) / 4294967295.0f;
        float acc = 0.0f;
        for (uint i = 0u; i <= cutoff; ++i) {
            acc += shortlist_vals[i];
            if (random_value <= acc || !isfinite(acc)) {
                output[0] = shortlist_indices[i];
                atomic_store_explicit(completion_counter, 0u, memory_order_relaxed);
                return;
            }
        }

        output[0] = shortlist_indices[cutoff];
        atomic_store_explicit(completion_counter, 0u, memory_order_relaxed);
    }
}
