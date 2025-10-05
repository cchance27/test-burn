#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

constant uint MAX_TOP_K = 256;
constant uint THREADGROUP_SIZE = 64;
constant uint TOKENS_PER_THREAD = 16;
constant uint THREADGROUP_TOKENS = THREADGROUP_SIZE * TOKENS_PER_THREAD;

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

void swap_local(thread float* a, thread float* b, thread uint* a_idx, thread uint* b_idx) {
    float temp = *a;
    *a = *b;
    *b = temp;
    uint temp_idx = *a_idx;
    *a_idx = *b_idx;
    *b_idx = temp_idx;
}

void sort_16(thread float* vals, thread uint* indices) {
    // Bitonic sorting network for 16 elements, descending
    #define SWAP(i, j) if (vals[i] < vals[j]) { swap_local(&vals[i], &vals[j], &indices[i], &indices[j]); }

    SWAP(0, 1); SWAP(2, 3); SWAP(4, 5); SWAP(6, 7); SWAP(8, 9); SWAP(10, 11); SWAP(12, 13); SWAP(14, 15);
    SWAP(0, 2); SWAP(1, 3); SWAP(4, 6); SWAP(5, 7); SWAP(8, 10); SWAP(9, 11); SWAP(12, 14); SWAP(13, 15);
    SWAP(0, 4); SWAP(1, 5); SWAP(2, 6); SWAP(3, 7); SWAP(8, 12); SWAP(9, 13); SWAP(10, 14); SWAP(11, 15);
    SWAP(0, 8); SWAP(1, 9); SWAP(2, 10); SWAP(3, 11); SWAP(4, 12); SWAP(5, 13); SWAP(6, 14); SWAP(7, 15);

    SWAP(0, 1); SWAP(2, 3); SWAP(4, 5); SWAP(6, 7); SWAP(8, 9); SWAP(10, 11); SWAP(12, 13); SWAP(14, 15);
    SWAP(2, 4); SWAP(3, 5); SWAP(6, 8); SWAP(7, 9); SWAP(10, 12); SWAP(11, 13);
    SWAP(1, 2); SWAP(3, 4); SWAP(5, 6); SWAP(7, 8); SWAP(9, 10); SWAP(11, 12);
    SWAP(0, 1); SWAP(2, 3); SWAP(4, 5); SWAP(6, 7); SWAP(8, 9); SWAP(10, 11); SWAP(12, 13); SWAP(14, 15);

    #undef SWAP
}


kernel void sample_top_k_top_p_stage1_f32(
    device const float* logits [[buffer(0)]],
    device uint* unused_output [[buffer(1)]],
    device float* partial_vals [[buffer(2)]],
    device uint* partial_indices [[buffer(3)]],
    device uint* partial_counts [[buffer(4)]],
    device float* fallback_vals [[buffer(5)]],
    device uint* fallback_indices [[buffer(6)]],
    device uint* fallback_flags [[buffer(7)]],
    constant SamplingParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]])
{
    (void)unused_output;

    uint vocab_size = params.vocab_size;
    if (vocab_size == 0u) {
        return;
    }

    uint threadgroup_index = tg_pos.x;
    uint threadgroup_count = max(params.threadgroup_count, 1u);
    if (threadgroup_index >= threadgroup_count) {
        return;
    }

    uint start_index = threadgroup_index * THREADGROUP_TOKENS;
    if (start_index >= vocab_size) {
        if (tid == 0u) {
            partial_counts[threadgroup_index] = 0u;
            fallback_flags[threadgroup_index] = 0u;
        }
        return;
    }

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

    thread float local_vals[TOKENS_PER_THREAD];
    thread uint local_indices[TOKENS_PER_THREAD];
    uint local_count = 0u;
    bool local_fallback_found = false;
    float local_fallback_val = -INFINITY;
    uint local_fallback_idx = 0u;

    float inv_temp = skip_sampling ? 0.0f : 1.0f / temperature;

    for (uint step = 0u; step < TOKENS_PER_THREAD; ++step) {
        uint index = start_index + step * THREADGROUP_SIZE + tid;
        if (index >= vocab_size) {
            local_vals[step] = -INFINITY;
            local_indices[step] = 0;
            continue;
        }

        float logit = float(logits[index]);
        if (isfinite(logit) && (!local_fallback_found || logit > local_fallback_val ||
                                (logit == local_fallback_val && index > local_fallback_idx))) {
            local_fallback_found = true;
            local_fallback_val = logit;
            local_fallback_idx = index;
        }

        if (skip_sampling) {
            local_vals[step] = -INFINITY;
            local_indices[step] = 0;
            continue;
        }

        float scaled_val = logit * inv_temp;
        if (!isfinite(scaled_val)) {
            local_vals[step] = -INFINITY;
            local_indices[step] = 0;
            continue;
        }
        local_vals[step] = scaled_val;
        local_indices[step] = index;
    }

    sort_16(local_vals, local_indices);

    threadgroup float shared_vals[THREADGROUP_SIZE * TOKENS_PER_THREAD];
    threadgroup uint shared_indices[THREADGROUP_SIZE * TOKENS_PER_THREAD];
    threadgroup uint shared_counts[THREADGROUP_SIZE];
    threadgroup uint shared_positions[THREADGROUP_SIZE];
    threadgroup float shared_fallback_vals[THREADGROUP_SIZE];
    threadgroup uint shared_fallback_indices[THREADGROUP_SIZE];
    threadgroup uint shared_fallback_flags[THREADGROUP_SIZE];
    threadgroup float candidate_vals[THREADGROUP_SIZE];
    threadgroup uint candidate_indices[THREADGROUP_SIZE];
    threadgroup uint candidate_owners[THREADGROUP_SIZE];
    threadgroup uint selection_active;

    shared_counts[tid] = min((uint)TOKENS_PER_THREAD, effective_top_k);
    shared_positions[tid] = 0u;
    shared_fallback_vals[tid] = local_fallback_val;
    shared_fallback_indices[tid] = local_fallback_idx;
    shared_fallback_flags[tid] = local_fallback_found ? 1u : 0u;

    uint shared_base = tid * TOKENS_PER_THREAD;
    for (uint i = 0u; i < TOKENS_PER_THREAD; ++i) {
        shared_vals[shared_base + i] = local_vals[i];
        shared_indices[shared_base + i] = local_indices[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool fallback_found = false;
    float fallback_val = -INFINITY;
    uint fallback_idx = 0u;
    if (shared_fallback_flags[tid] != 0u) {
        float candidate_val = shared_fallback_vals[tid];
        uint candidate_idx = shared_fallback_indices[tid];
        if (!fallback_found || candidate_val > fallback_val ||
            (candidate_val == fallback_val && candidate_idx > fallback_idx)) {
            fallback_found = true;
            fallback_val = candidate_val;
            fallback_idx = candidate_idx;
        }
    }

    threadgroup float fallback_vals_shared[THREADGROUP_SIZE];
    threadgroup uint fallback_indices_shared[THREADGROUP_SIZE];
    threadgroup uint fallback_flags_shared[THREADGROUP_SIZE];

    fallback_vals_shared[tid] = fallback_val;
    fallback_indices_shared[tid] = fallback_idx;
    fallback_flags_shared[tid] = fallback_found ? 1u : 0u;

    for (uint offset = THREADGROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < offset) {
            float other_val = fallback_vals_shared[tid + offset];
            uint other_idx = fallback_indices_shared[tid + offset];
            uint other_flag = fallback_flags_shared[tid + offset];
            bool take_other = other_flag != 0u &&
                (fallback_flags_shared[tid] == 0u || other_val > fallback_vals_shared[tid] ||
                 (other_val == fallback_vals_shared[tid] && other_idx > fallback_indices_shared[tid]));
            if (take_other) {
                fallback_vals_shared[tid] = other_val;
                fallback_indices_shared[tid] = other_idx;
                fallback_flags_shared[tid] = 1u;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint shortlist_count = 0u;
    if (!skip_sampling) {
        for (uint selection = 0u; selection < effective_top_k; ++selection) {
            if (tid == 0u) {
                selection_active = 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint pos = shared_positions[tid];
            bool has_candidate = pos < shared_counts[tid];
            if (has_candidate) {
                uint offset = tid * TOKENS_PER_THREAD + pos;
                candidate_vals[tid] = shared_vals[offset];
                candidate_indices[tid] = shared_indices[offset];
                candidate_owners[tid] = tid;
            } else {
                candidate_vals[tid] = -INFINITY;
                candidate_indices[tid] = 0u;
                candidate_owners[tid] = tid;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint offset = THREADGROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
                if (tid < offset) {
                    float other_val = candidate_vals[tid + offset];
                    uint other_idx = candidate_indices[tid + offset];
                    uint other_owner = candidate_owners[tid + offset];
                    bool take_other = other_val > candidate_vals[tid] ||
                        (other_val == candidate_vals[tid] && other_idx > candidate_indices[tid]);
                    if (take_other) {
                        candidate_vals[tid] = other_val;
                        candidate_indices[tid] = other_idx;
                        candidate_owners[tid] = other_owner;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0u) {
                float best_val = candidate_vals[0];
                uint best_idx = candidate_indices[0];
                if (!isfinite(best_val)) {
                    selection_active = 0u;
                } else {
                    uint winner_tid = candidate_owners[0];
                    uint base = threadgroup_index * effective_top_k + shortlist_count;
                    partial_vals[base] = best_val;
                    partial_indices[base] = best_idx;
                    ++shortlist_count;

                    shared_positions[winner_tid] += 1u;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (selection_active == 0u) {
                break;
            }
        }
    }

    if (tid == 0u) {
        partial_counts[threadgroup_index] = shortlist_count;
        fallback_vals[threadgroup_index] = fallback_vals_shared[0];
        fallback_indices[threadgroup_index] = fallback_indices_shared[0];
        fallback_flags[threadgroup_index] = fallback_flags_shared[0];
    }
}

kernel void sample_top_k_top_p_stage1_f16(
    device const half* logits [[buffer(0)]],
    device uint* unused_output [[buffer(1)]],
    device float* partial_vals [[buffer(2)]],
    device uint* partial_indices [[buffer(3)]],
    device uint* partial_counts [[buffer(4)]],
    device float* fallback_vals [[buffer(5)]],
    device uint* fallback_indices [[buffer(6)]],
    device uint* fallback_flags [[buffer(7)]],
    constant SamplingParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]])
{
    (void)unused_output;

    uint vocab_size = params.vocab_size;
    if (vocab_size == 0u) {
        return;
    }

    uint threadgroup_index = tg_pos.x;
    uint threadgroup_count = max(params.threadgroup_count, 1u);
    if (threadgroup_index >= threadgroup_count) {
        return;
    }

    uint start_index = threadgroup_index * THREADGROUP_TOKENS;
    if (start_index >= vocab_size) {
        if (tid == 0u) {
            partial_counts[threadgroup_index] = 0u;
            fallback_flags[threadgroup_index] = 0u;
        }
        return;
    }

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

    thread float local_vals[TOKENS_PER_THREAD];
    thread uint local_indices[TOKENS_PER_THREAD];
    uint local_count = 0u;
    bool local_fallback_found = false;
    float local_fallback_val = -INFINITY;
    uint local_fallback_idx = 0u;

    float inv_temp = skip_sampling ? 0.0f : 1.0f / temperature;

    for (uint step = 0u; step < TOKENS_PER_THREAD; ++step) {
        uint index = start_index + step * THREADGROUP_SIZE + tid;
        if (index >= vocab_size) {
            local_vals[step] = -INFINITY;
            local_indices[step] = 0;
            continue;
        }

        float logit = float(logits[index]);
        if (isfinite(logit) && (!local_fallback_found || logit > local_fallback_val ||
                                (logit == local_fallback_val && index > local_fallback_idx))) {
            local_fallback_found = true;
            local_fallback_val = logit;
            local_fallback_idx = index;
        }

        if (skip_sampling) {
            local_vals[step] = -INFINITY;
            local_indices[step] = 0;
            continue;
        }

        float scaled_val = logit * inv_temp;
        if (!isfinite(scaled_val)) {
            local_vals[step] = -INFINITY;
            local_indices[step] = 0;
            continue;
        }
        local_vals[step] = scaled_val;
        local_indices[step] = index;
    }

    sort_16(local_vals, local_indices);

    threadgroup float shared_vals[THREADGROUP_SIZE * TOKENS_PER_THREAD];
    threadgroup uint shared_indices[THREADGROUP_SIZE * TOKENS_PER_THREAD];
    threadgroup uint shared_counts[THREADGROUP_SIZE];
    threadgroup uint shared_positions[THREADGROUP_SIZE];
    threadgroup float shared_fallback_vals[THREADGROUP_SIZE];
    threadgroup uint shared_fallback_indices[THREADGROUP_SIZE];
    threadgroup uint shared_fallback_flags[THREADGROUP_SIZE];
    threadgroup float candidate_vals[THREADGROUP_SIZE];
    threadgroup uint candidate_indices[THREADGROUP_SIZE];
    threadgroup uint candidate_owners[THREADGROUP_SIZE];
    threadgroup uint selection_active;

    shared_counts[tid] = min((uint)TOKENS_PER_THREAD, effective_top_k);
    shared_positions[tid] = 0u;
    shared_fallback_vals[tid] = local_fallback_val;
    shared_fallback_indices[tid] = local_fallback_idx;
    shared_fallback_flags[tid] = local_fallback_found ? 1u : 0u;

    uint shared_base = tid * TOKENS_PER_THREAD;
    for (uint i = 0u; i < TOKENS_PER_THREAD; ++i) {
        shared_vals[shared_base + i] = local_vals[i];
        shared_indices[shared_base + i] = local_indices[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool fallback_found = false;
    float fallback_val = -INFINITY;
    uint fallback_idx = 0u;
    if (shared_fallback_flags[tid] != 0u) {
        float candidate_val = shared_fallback_vals[tid];
        uint candidate_idx = shared_fallback_indices[tid];
        if (!fallback_found || candidate_val > fallback_val ||
            (candidate_val == fallback_val && candidate_idx > fallback_idx)) {
            fallback_found = true;
            fallback_val = candidate_val;
            fallback_idx = candidate_idx;
        }
    }

    threadgroup float fallback_vals_shared[THREADGROUP_SIZE];
    threadgroup uint fallback_indices_shared[THREADGROUP_SIZE];
    threadgroup uint fallback_flags_shared[THREADGROUP_SIZE];

    fallback_vals_shared[tid] = fallback_val;
    fallback_indices_shared[tid] = fallback_idx;
    fallback_flags_shared[tid] = fallback_found ? 1u : 0u;

    for (uint offset = THREADGROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < offset) {
            float other_val = fallback_vals_shared[tid + offset];
            uint other_idx = fallback_indices_shared[tid + offset];
            uint other_flag = fallback_flags_shared[tid + offset];
            bool take_other = other_flag != 0u &&
                (fallback_flags_shared[tid] == 0u || other_val > fallback_vals_shared[tid] ||
                 (other_val == fallback_vals_shared[tid] && other_idx > fallback_indices_shared[tid]));
            if (take_other) {
                fallback_vals_shared[tid] = other_val;
                fallback_indices_shared[tid] = other_idx;
                fallback_flags_shared[tid] = 1u;
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint shortlist_count = 0u;
    if (!skip_sampling) {
        for (uint selection = 0u; selection < effective_top_k; ++selection) {
            if (tid == 0u) {
                selection_active = 1u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            uint pos = shared_positions[tid];
            bool has_candidate = pos < shared_counts[tid];
            if (has_candidate) {
                uint offset = tid * TOKENS_PER_THREAD + pos;
                candidate_vals[tid] = shared_vals[offset];
                candidate_indices[tid] = shared_indices[offset];
                candidate_owners[tid] = tid;
            } else {
                candidate_vals[tid] = -INFINITY;
                candidate_indices[tid] = 0u;
                candidate_owners[tid] = tid;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint offset = THREADGROUP_SIZE / 2u; offset > 0u; offset >>= 1u) {
                if (tid < offset) {
                    float other_val = candidate_vals[tid + offset];
                    uint other_idx = candidate_indices[tid + offset];
                    uint other_owner = candidate_owners[tid + offset];
                    bool take_other = other_val > candidate_vals[tid] ||
                        (other_val == candidate_vals[tid] && other_idx > candidate_indices[tid]);
                    if (take_other) {
                        candidate_vals[tid] = other_val;
                        candidate_indices[tid] = other_idx;
                        candidate_owners[tid] = other_owner;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (tid == 0u) {
                float best_val = candidate_vals[0];
                uint best_idx = candidate_indices[0];
                if (!isfinite(best_val)) {
                    selection_active = 0u;
                } else {
                    uint winner_tid = candidate_owners[0];
                    uint base = threadgroup_index * effective_top_k + shortlist_count;
                    partial_vals[base] = best_val;
                    partial_indices[base] = best_idx;
                    ++shortlist_count;

                    shared_positions[winner_tid] += 1u;
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (selection_active == 0u) {
                break;
            }
        }
    }

    if (tid == 0u) {
        partial_counts[threadgroup_index] = shortlist_count;
        fallback_vals[threadgroup_index] = fallback_vals_shared[0];
        fallback_indices[threadgroup_index] = fallback_indices_shared[0];
        fallback_flags[threadgroup_index] = fallback_flags_shared[0];
    }
}

kernel void sample_top_k_top_p_finalize_f32(
    device const float* logits [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const float* partial_vals [[buffer(2)]],
    device const uint* partial_indices [[buffer(3)]],
    device const uint* partial_counts [[buffer(4)]],
    device const float* fallback_vals [[buffer(5)]],
    device const uint* fallback_indices [[buffer(6)]],
    device const uint* fallback_flags [[buffer(7)]],
    constant SamplingParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]])
{
    (void)logits;

    if (tid != 0u) {
        return;
    }

    uint vocab_size = params.vocab_size;
    if (vocab_size == 0u) {
        output[0] = 0u;
        return;
    }

    uint threadgroup_count = max(params.threadgroup_count, 1u);
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
            return;
        }
    }

    output[0] = shortlist_indices[cutoff];
}

kernel void sample_top_k_top_p_finalize_f16(
    device const half* logits [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device const float* partial_vals [[buffer(2)]],
    device const uint* partial_indices [[buffer(3)]],
    device const uint* partial_counts [[buffer(4)]],
    device const float* fallback_vals [[buffer(5)]],
    device const uint* fallback_indices [[buffer(6)]],
    device const uint* fallback_flags [[buffer(7)]],
    constant SamplingParams& params [[buffer(9)]],
    uint tid [[thread_index_in_threadgroup]])
{
    (void)logits;

    if (tid != 0u) {
        return;
    }

    uint vocab_size = params.vocab_size;
    if (vocab_size == 0u) {
        output[0] = 0u;
        return;
    }

    uint threadgroup_count = max(params.threadgroup_count, 1u);
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
            return;
        }
    }

    output[0] = shortlist_indices[cutoff];
}
