#ifndef METALLIC_V2_SIMD_METAL
#define METALLIC_V2_SIMD_METAL

#include <metal_stdlib>
using namespace metal;

/// Optimized block-wide max reduction using SIMD intrinsics.
///
/// Requires threadgroup memory `shared_max[THREADS]`.
/// Performs 2-stage reduction:
/// 1. SIMD-group reduction (registers only, no barrier).
/// 2. Threadgroup reduction (via shared memory).
/// Returns the reduction result to ALL threads (implicit broadcast via shared[0] read after barrier).
template<typename T, uint THREADS>
inline T block_reduce_max(T val, threadgroup T* shared_mem, uint tid, uint active_threads) {
    // 1. SIMD-group reduction
    T max_v = simd_max(val);

    // Optimization: If the entire block fits in one SIMD group, we are done.
    if (active_threads <= 32) {
        return max_v;
    }

    // 2. Write representative to shared memory
    // Only the first thread in each SIMD group writes
    if (tid % 32 == 0) {
        shared_mem[tid / 32] = max_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Thread 0 aggregates SIMD group results
    if (tid == 0) {
        T final_max = shared_mem[0];
        // Loop over number of active simd groups
        uint active_groups = (active_threads + 31) / 32;
        uint limit = min(active_groups, THREADS / 32);
        
        for (uint i = 1; i < limit; ++i) {
            final_max = max(final_max, shared_mem[i]);
        }
        shared_mem[0] = final_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared_mem[0];
}

/// Optimized block-wide sum reduction using SIMD intrinsics.
///
/// Requires threadgroup memory `shared_sum[THREADS]`.
template<typename T, uint THREADS>
inline T block_reduce_sum(T val, threadgroup T* shared_mem, uint tid, uint active_threads) {
    // 1. SIMD-group reduction
    T sum_v = simd_sum(val);

    // Optimization: If the entire block fits in one SIMD group, we are done.
    if (active_threads <= 32) {
        return sum_v;
    }

    // 2. Write representative to shared memory
    if (tid % 32 == 0) {
        shared_mem[tid / 32] = sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Thread 0 aggregates SIMD group results
    if (tid == 0) {
        T final_sum = shared_mem[0];
        uint active_groups = (active_threads + 31) / 32;
        uint limit = min(active_groups, THREADS / 32);

        for (uint i = 1; i < limit; ++i) {
            final_sum += shared_mem[i];
        }
        shared_mem[0] = final_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    return shared_mem[0];
}

#endif // METALLIC_V2_SIMD_METAL
