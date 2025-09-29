#include <metal_stdlib>
using namespace metal;

/// Compute kernel that rearranges KV rows for grouped-query attention.
///
/// It maps input K/V of shape [M, kv_dim] where M = batch*seq into output shape
/// [batch * n_heads, seq, head_dim] by grouping Q heads to KV heads:
///   out[ out_batch, s, hd ] = input[ (b*seq + s) * kv_dim + kv_h * kv_head_dim + hd ]
/// where:
///   out_batch = b * n_heads + h
///   kv_h = h / group_size
kernel void kv_rearrange_kernel(device const float* input [[buffer(0)]],
                                device float* output [[buffer(1)]],
                                constant uint &kv_dim [[buffer(2)]],
                                constant uint &kv_head_dim [[buffer(3)]],
                                constant uint &n_heads [[buffer(4)]],
                                constant uint &n_kv_heads [[buffer(5)]],
                                constant uint &head_dim [[buffer(6)]],
                                constant uint &seq [[buffer(7)]],
                                constant uint &total_elements [[buffer(8)]],
                                uint gid [[thread_position_in_grid]]) {
    if (gid >= total_elements) return;

    // Output layout: [batch_heads, seq, head_dim]
    uint hd = gid % head_dim;
    uint tmp = gid / head_dim;
    uint s = tmp % seq;
    uint out_batch = tmp / seq;

    uint b = out_batch / n_heads;
    uint h = out_batch % n_heads;
    uint group_size = n_heads / n_kv_heads;
    uint kv_h = h / group_size;

    // Source index into input: (b*seq + s) * kv_dim + kv_h * kv_head_dim + hd
    uint src_row = b * seq + s;
    uint src_idx = src_row * kv_dim + kv_h * kv_head_dim + hd;

    output[gid] = input[src_idx];
}