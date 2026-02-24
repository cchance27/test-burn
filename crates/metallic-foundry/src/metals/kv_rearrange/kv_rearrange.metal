#include <metal_stdlib>
using namespace metal;

// KvRearrangeParams struct is injected by Foundry via struct_defs()

/// KV Rearrange kernel for runtime storage types.
///
/// Rearranges QKV outputs from [batch*seq, kv_dim] to [batch*n_heads, seq, head_dim].
/// Handles GQA (Grouped Query Attention) via group_size = n_heads / n_kv_heads.
kernel void kv_rearrange_kernel(
    const device InputStorageT* input [[buffer(0)]],
    device OutputStorageT* output [[buffer(1)]],
    constant KvRearrangeParamsResolved* params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint kv_dim = params->kv_dim;
    uint row_stride = params->row_stride;
    uint kv_head_dim = params->kv_head_dim;
    uint n_heads = params->n_heads;
    uint n_kv_heads = params->n_kv_heads;
    uint head_dim = params->head_dim;
    uint seq = params->seq;
    uint total_elements = params->total_elements;
    
    if (gid >= total_elements) return;
    
    // Decode output index: output[batch*n_heads, seq, head_dim]
    uint hd = gid % head_dim;           // feature within head
    uint tmp = gid / head_dim;
    uint s = tmp % seq;                 // sequence position
    uint out_batch = tmp / seq;
    uint b = out_batch / n_heads;       // batch index
    uint h = out_batch % n_heads;       // head index
    
    // GQA: map query head to KV head
    uint group_size = n_heads / n_kv_heads;
    uint kv_h = h / group_size;
    
    // Compute source index
    uint base_offset = kv_h * kv_head_dim + hd;
    if (base_offset >= kv_dim) return;
    
    uint src_row = b * seq + s;
    uint src_idx = src_row * row_stride + base_offset;
    
    metallic_store_output(output, gid, metallic_to_accum(metallic_load_input(input, src_idx)));
}
