#ifndef SOFTMAX_SDPA_BATCHED_METAL_H
#define SOFTMAX_SDPA_BATCHED_METAL_H

#include <metal_stdlib>
using namespace metal;

inline uint sdpa_row_local(uint row_idx, uint rows_per_batch) {
    // rows_per_batch is M (query sequence length). In SDPA we batch heads by
    // flattening head*row into row_idx; causal masking must use row within M.
    return (rows_per_batch == 0) ? 0 : (row_idx % rows_per_batch);
}

/// Helper to find the maximum value in a row partition.
inline float find_row_max_sdpa_batched(
    const device InputStorageT* matrix,
    uint row_idx,
    uint tid,
    uint seq_k,
    uint causal,
    uint query_offset,
    uint rows_per_batch
) {
    // Per-row causal mask index (matches Context behavior), but row_idx is
    // flattened over heads: row_local = row_idx % M.
    uint row_local = sdpa_row_local(row_idx, rows_per_batch);
    uint mask_pos = row_local + query_offset;

    float local_max = -INFINITY;
    for (uint i = tid; i < seq_k; i += 256) {
        uint input_idx = row_idx * seq_k + i;
        float val = metallic_load_input(matrix, input_idx);

        if (causal != 0 && i > mask_pos) {
             val = -INFINITY;
        }
        if (val > local_max) {
             local_max = val;
        }
    }
    return local_max;
}

/// Helper to compute the sum of exponentials (val - max).
inline float compute_exp_sum_sdpa_batched(
    const device InputStorageT* matrix,
    float row_max,
    uint row_idx,
    uint tid,
    uint seq_k,
    uint causal,
    uint query_offset,
    uint rows_per_batch
) {
    if (!isfinite(row_max)) {
        // Fully-masked or invalid row: force zero probability mass.
        return 0.0f;
    }

    uint row_local = sdpa_row_local(row_idx, rows_per_batch);
    uint mask_pos = row_local + query_offset;

    float local_sum = 0.0f;
    for (uint i = tid; i < seq_k; i += 256) {
        uint input_idx = row_idx * seq_k + i;
        float val = metallic_load_input(matrix, input_idx);

        if (causal == 0 || i <= mask_pos) {
            float diff = val - row_max;
            local_sum += metal::fast::exp(diff);
        }
    }
    return local_sum;
}

/// Helper to normalize and write results.
inline void normalize_and_write_sdpa_batched(
    device OutputStorageT* output,
    const device InputStorageT* matrix,
    float row_max,
    float row_sum,
    uint row_idx,
    uint tid,
    uint seq_k,
    uint causal,
    uint query_offset,
    uint rows_per_batch
) {
    uint row_local = sdpa_row_local(row_idx, rows_per_batch);
    uint mask_pos = row_local + query_offset;
    bool invalid_norm = (!isfinite(row_max)) || (!isfinite(row_sum)) || (row_sum <= 0.0f);

    for (uint i = tid; i < seq_k; i += 256) {
        uint input_idx = row_idx * seq_k + i;
        float val = metallic_load_input(matrix, input_idx);

        if (causal != 0 && i > mask_pos) {
            metallic_store_output(output, input_idx, metallic_to_accum(0.0f));
        } else if (invalid_norm) {
            metallic_store_output(output, input_idx, metallic_to_accum(0.0f));
        } else {
            float diff = val - row_max;
            float numer = isfinite(diff) ? metal::fast::exp(diff) : 0.0f;
            float prob = numer / row_sum;
            if (!isfinite(prob)) {
                prob = 0.0f;
            }
            metallic_store_output(output, input_idx, metallic_to_accum(prob));
        }
    }
}

#endif // SOFTMAX_SDPA_BATCHED_METAL_H
