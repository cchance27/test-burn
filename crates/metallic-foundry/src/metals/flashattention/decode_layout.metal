template<typename TQ, typename TK, typename TV, typename TO>
struct FlashHeadLayoutPtrs {
    const device TQ* q_ptr;
    const device TK* k_ptr;
    const device TV* v_ptr;
    device TO* output_ptr;
};

template<typename TQ, typename TK, typename TV, typename TO>
ALWAYS_INLINE FlashHeadLayoutPtrs<TQ, TK, TV, TO> run_flash_head_layout_stage(
    const device TQ* q,
    const device TK* k,
    const device TV* v,
    device TO* output,
    constant uint& q_stride_b,
    constant uint& q_stride_h,
    constant uint& k_stride_b,
    constant uint& k_stride_h,
    constant uint& v_stride_b,
    constant uint& v_stride_h,
    constant uint& out_stride_b,
    constant uint& out_stride_h,
    uint3 gid
) {
    uint head_idx = gid.y;
    uint batch_idx = gid.z;

    ulong q_offset = batch_idx * q_stride_b + head_idx * q_stride_h;

    // Decode supports expanded KV cache and compact GQA cache from the same path.
    uint q_heads = (q_stride_h > 0u) ? max(1u, q_stride_b / q_stride_h) : 1u;
    uint k_heads = (k_stride_h > 0u) ? max(1u, k_stride_b / k_stride_h) : q_heads;
    uint group_size = max(1u, q_heads / max(1u, k_heads));
    uint kv_head_idx = head_idx / group_size;

    ulong k_offset = batch_idx * k_stride_b + kv_head_idx * k_stride_h;
    ulong v_offset = batch_idx * v_stride_b + kv_head_idx * v_stride_h;
    ulong out_offset = batch_idx * out_stride_b + head_idx * out_stride_h;

    FlashHeadLayoutPtrs<TQ, TK, TV, TO> out;
    out.q_ptr = q + q_offset;
    out.k_ptr = k + k_offset;
    out.v_ptr = v + v_offset;
    out.output_ptr = output + out_offset;
    return out;
}
