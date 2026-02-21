#include <metal_stdlib>
using namespace metal;

template<typename Policy, uint vec_width, bool has_norm>
ALWAYS_INLINE float2 run_ffn_dual_project_stage(
    const device uchar* w_gate,
    const device uchar* s_gate,
    const device uchar* w_up,
    const device uchar* s_up,
    const device half* input,
    constant uint& k_dim,
    constant uint& n_dim,
    constant uint& weights_per_block,
    const device half* gamma,
    float inv_rms,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    const uint blocks_per_k = (k_dim + weights_per_block - 1) / weights_per_block;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    uint k_base = 0;

    ulong scale_stride = (ulong)row_idx * blocks_per_k * Policy::SCALE_BYTES;
    const device uchar* row_s_gate = s_gate + scale_stride;
    const device uchar* row_s_up = s_up + scale_stride;

    while (k_base + K_CHUNK_SIZE <= k_dim) {
        uint k = k_base + lane_id * vec_width;
        float4 xv_raw = *(const device float4*)(input + batch_idx * k_dim + k);
        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
            const device half* g_ptr = gamma + k;
            f_lo.x *= (float)g_ptr[0]; f_lo.y *= (float)g_ptr[1];
            f_lo.z *= (float)g_ptr[2]; f_lo.w *= (float)g_ptr[3];
            f_hi.x *= (float)g_ptr[4]; f_hi.y *= (float)g_ptr[5];
            f_hi.z *= (float)g_ptr[6]; f_hi.w *= (float)g_ptr[7];
            xv_lo = half4(f_lo);
            xv_hi = half4(f_hi);
        }

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        uint block_off = k / weights_per_block;
        
        float s_gate_val = (float)Policy::load_scale(row_s_gate, block_off);
        float s_up_val = (float)Policy::load_scale(row_s_up, block_off);
        float a_gate_val = (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_gate, block_off) : 0.0f);
        float a_up_val = (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_up, block_off) : 0.0f);

        {
            float w[vec_width];
            Policy::template load_weights<vec_width>(w_gate, row_idx * k_dim + k, w);
            acc_gate += s_gate_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (Policy::HAS_AFFINE ? (a_gate_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        {
            float w[vec_width];
            Policy::template load_weights<vec_width>(w_up, row_idx * k_dim + k, w);
            acc_up += s_up_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (Policy::HAS_AFFINE ? (a_up_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        k_base += K_CHUNK_SIZE;
    }

    if (k_base < k_dim) {
        uint k = k_base + lane_id * vec_width;
        float4 xv_raw = float4(0.0f);

        if (k + vec_width <= k_dim) {
            xv_raw = *(const device float4*)(input + batch_idx * k_dim + k);
        } else if (k < k_dim) {
            for (uint i = 0; i < vec_width && k + i < k_dim; ++i) {
                ((thread half*)&xv_raw)[i] = input[batch_idx * k_dim + k + i];
            }
        }

        half4 xv_lo = as_type<half4>(xv_raw.xy);
        half4 xv_hi = as_type<half4>(xv_raw.zw);

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
            const device half* g_ptr = gamma + k;
            f_lo.x *= (float)g_ptr[0]; f_lo.y *= (float)g_ptr[1];
            f_lo.z *= (float)g_ptr[2]; f_lo.w *= (float)g_ptr[3];
            f_hi.x *= (float)g_ptr[4]; f_hi.y *= (float)g_ptr[5];
            f_hi.z *= (float)g_ptr[6]; f_hi.w *= (float)g_ptr[7];
            xv_lo = half4(f_lo);
            xv_hi = half4(f_hi);
        }

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        uint block_off = k / weights_per_block;
        float s_gate_val = (float)Policy::load_scale(row_s_gate, block_off);
        float s_up_val = (float)Policy::load_scale(row_s_up, block_off);
        float a_gate_val = (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_gate, block_off) : 0.0f);
        float a_up_val = (Policy::HAS_AFFINE ? (float)Policy::load_affine(row_s_up, block_off) : 0.0f);

        {
            float w[vec_width];
            Policy::template load_weights<vec_width>(w_gate, row_idx * k_dim + k, w);
            acc_gate += s_gate_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (Policy::HAS_AFFINE ? (a_gate_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        {
            float w[vec_width];
            Policy::template load_weights<vec_width>(w_up, row_idx * k_dim + k, w);
            acc_up += s_up_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (Policy::HAS_AFFINE ? (a_up_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }
    }

    return float2(acc_gate, acc_up);
}
