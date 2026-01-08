#include <metal_stdlib>
using namespace metal;

/// Helper to apply RoPE to a single query element in threadgroup memory.
///
/// tid: Thread ID (corresponds to feature index 0..HEAD_DIM-1)
/// HEAD_DIM: Compile-time constant or parameter
/// pos: Position index for RoPE
ALWAYS_INLINE void apply_rope_q(
    threadgroup half* q_shared,
    const device half* q_ptr,
    const device half* cos_buf,
    const device half* sin_buf,
    uint tid,
    uint HEAD_DIM,
    uint pos
) {
    uint half_dim = HEAD_DIM / 2;
    
    // Load ONE element from global
    half val = q_ptr[tid];
    
    if (tid < half_dim) {
        // Active thread for pair (tid, tid + half_dim)
        float x_i = (float)q_ptr[tid];
        float x_j = (float)q_ptr[tid + half_dim];
        
        float cos_v = (float)cos_buf[pos * half_dim + tid];
        float sin_v = (float)sin_buf[pos * half_dim + tid];
        
        // out_i = x_i * cos - x_j * sin
        float out_i = x_i * cos_v - x_j * sin_v;
        
        // out_j = x_j * cos + x_i * sin
        float out_j = x_j * cos_v + x_i * sin_v;
        
        q_shared[tid] = (half)out_i;
        q_shared[tid + half_dim] = (half)out_j;
    }
    // Threads >= half_dim do nothing
}
