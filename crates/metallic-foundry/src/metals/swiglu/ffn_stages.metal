#include <metal_stdlib>
using namespace metal;

typedef FastScalarT FfnFastScalarT;
typedef FastVec4T FfnFastVec4T;

ALWAYS_INLINE FfnFastVec4T ffn_load_input_half4(const device InputStorageT* input, const ulong idx) {
#if METALLIC_FASTPATH_INPUT_HALF
    return ((const device FfnFastVec4T*)((const device FfnFastScalarT*)input + idx))[0];
#else
    return FfnFastVec4T(metallic_load_input_vec4f(input, idx));
#endif
}

ALWAYS_INLINE FfnFastVec4T ffn_load_gamma_half4(const device GammaStorageT* gamma, const ulong idx) {
#if METALLIC_FASTPATH_GAMMA_HALF
    return ((const device FfnFastVec4T*)((const device FfnFastScalarT*)gamma + idx))[0];
#else
    return FfnFastVec4T(metallic_load_gamma_vec4f(gamma, idx));
#endif
}

template<typename PolicyGate, typename PolicyUp, uint vec_width, bool has_norm>
ALWAYS_INLINE float2 run_ffn_dual_project_stage(
    const device uchar* w_gate,
    const device uchar* s_gate,
    const device uchar* w_up,
    const device uchar* s_up,
    const device InputStorageT* input,
    constant uint& k_dim,
    constant uint& n_dim,
    constant uint& weights_per_block_gate,
    constant uint& weights_per_block_up,
    const device GammaStorageT* gamma,
    float inv_rms,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    const uint blocks_per_k_gate = (k_dim + weights_per_block_gate - 1) / weights_per_block_gate;
    const uint blocks_per_k_up = (k_dim + weights_per_block_up - 1) / weights_per_block_up;
    float acc_gate = 0.0f;
    float acc_up = 0.0f;
    uint k_base = 0;

    const ulong scale_stride_gate = (ulong)row_idx * blocks_per_k_gate * PolicyGate::SCALE_BYTES; // INDEX64_OK
    const ulong scale_stride_up = (ulong)row_idx * blocks_per_k_up * PolicyUp::SCALE_BYTES; // INDEX64_OK
    const device uchar* row_s_gate = s_gate + scale_stride_gate;
    const device uchar* row_s_up = s_up + scale_stride_up;
    const bool same_wpb = (weights_per_block_gate == weights_per_block_up);
#if METALLIC_FASTPATH_INPUT_HALF
    const device FfnFastScalarT* input_half = (const device FfnFastScalarT*)input;
#endif
#if METALLIC_FASTPATH_GAMMA_HALF
    const device FfnFastScalarT* gamma_half = (const device FfnFastScalarT*)gamma;
#endif

    while (k_base + K_CHUNK_SIZE <= k_dim) {
        uint k = k_base + lane_id * vec_width;
        FfnFastVec4T xv_lo;
        FfnFastVec4T xv_hi;
#if METALLIC_FASTPATH_INPUT_HALF
        const uint x_base = batch_idx * k_dim + k;
        const float4 xv_raw = *(const device float4*)(input_half + x_base);
        xv_lo = as_type<FfnFastVec4T>(xv_raw.xy);
        xv_hi = (vec_width == 8) ? as_type<FfnFastVec4T>(xv_raw.zw) : FfnFastVec4T((FfnFastScalarT)0.0f);
#else
        const ulong x_base = (ulong)batch_idx * (ulong)k_dim + (ulong)k; // INDEX64_OK
        xv_lo = ffn_load_input_half4(input, x_base + 0ul);
        xv_hi = (vec_width == 8) ? ffn_load_input_half4(input, x_base + 4ul) : FfnFastVec4T((FfnFastScalarT)0.0f);
#endif

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
#if METALLIC_FASTPATH_GAMMA_HALF
            const device FfnFastScalarT* g_ptr = gamma_half + k;
            f_lo.x *= (float)g_ptr[0];
            f_lo.y *= (float)g_ptr[1];
            f_lo.z *= (float)g_ptr[2];
            f_lo.w *= (float)g_ptr[3];
            if (vec_width == 8) {
                f_hi.x *= (float)g_ptr[4];
                f_hi.y *= (float)g_ptr[5];
                f_hi.z *= (float)g_ptr[6];
                f_hi.w *= (float)g_ptr[7];
            }
#else
            FfnFastVec4T g_lo_h = ffn_load_gamma_half4(gamma, (ulong)k + 0ul); // INDEX64_OK
            f_lo *= float4(g_lo_h);
            if (vec_width == 8) {
                FfnFastVec4T g_hi_h = ffn_load_gamma_half4(gamma, (ulong)k + 4ul); // INDEX64_OK
                f_hi *= float4(g_hi_h);
            }
#endif
            xv_lo = FfnFastVec4T(f_lo);
            xv_hi = FfnFastVec4T(f_hi);
        }

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        const uint block_off_gate = k / weights_per_block_gate;
        const uint block_off_up = same_wpb ? block_off_gate : (k / weights_per_block_up);
        
        float s_gate_val = (float)PolicyGate::load_scale(row_s_gate, block_off_gate);
        float s_up_val = (float)PolicyUp::load_scale(row_s_up, block_off_up);
        float a_gate_val = (PolicyGate::HAS_AFFINE ? (float)PolicyGate::load_affine(row_s_gate, block_off_gate) : 0.0f);
        float a_up_val = (PolicyUp::HAS_AFFINE ? (float)PolicyUp::load_affine(row_s_up, block_off_up) : 0.0f);

        {
            float w[vec_width];
            PolicyGate::template load_weights<vec_width>(w_gate, row_idx * k_dim + k, w);
            acc_gate += s_gate_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (PolicyGate::HAS_AFFINE ? (a_gate_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        {
            float w[vec_width];
            PolicyUp::template load_weights<vec_width>(w_up, row_idx * k_dim + k, w);
            acc_up += s_up_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (PolicyUp::HAS_AFFINE ? (a_up_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        k_base += K_CHUNK_SIZE;
    }

    if (k_base < k_dim) {
        uint k = k_base + lane_id * vec_width;
        float4 xv_raw = float4(0.0f);

        if (k + vec_width <= k_dim) {
            const uint x_base = batch_idx * k_dim + k;
#if METALLIC_FASTPATH_INPUT_HALF
            xv_raw = *(const device float4*)(input_half + x_base);
#else
            const FfnFastVec4T xv_lo_h = ffn_load_input_half4(input, (ulong)x_base + 0ul); // INDEX64_OK
            const FfnFastVec4T xv_hi_h = (vec_width == 8) ? ffn_load_input_half4(input, (ulong)x_base + 4ul) : FfnFastVec4T((FfnFastScalarT)0.0f); // INDEX64_OK
            xv_raw = float4(as_type<float2>(xv_lo_h), as_type<float2>(xv_hi_h));
#endif
        } else if (k < k_dim) {
#if METALLIC_FASTPATH_INPUT_HALF
            const uint x_base = batch_idx * k_dim + k;
            for (uint i = 0; i < vec_width && k + i < k_dim; ++i) {
                ((thread FfnFastScalarT*)&xv_raw)[i] = input_half[x_base + i];
            }
#else
            for (uint i = 0; i < vec_width && k + i < k_dim; ++i) {
                ((thread FfnFastScalarT*)&xv_raw)[i] = (FfnFastScalarT)metallic_load_input(
                    input,
                    (ulong)batch_idx * (ulong)k_dim + (ulong)k + (ulong)i // INDEX64_OK
                );
            }
#endif
        }

        FfnFastVec4T xv_lo = as_type<FfnFastVec4T>(xv_raw.xy);
        FfnFastVec4T xv_hi = as_type<FfnFastVec4T>(xv_raw.zw);

        if (has_norm && gamma) {
            float4 f_lo = float4(xv_lo);
            float4 f_hi = float4(xv_hi);
            f_lo *= inv_rms;
            f_hi *= inv_rms;
#if METALLIC_FASTPATH_GAMMA_HALF
            const device FfnFastScalarT* g_ptr = gamma_half + k;
            f_lo.x *= (float)g_ptr[0];
            f_lo.y *= (float)g_ptr[1];
            f_lo.z *= (float)g_ptr[2];
            f_lo.w *= (float)g_ptr[3];
            if (vec_width == 8) {
                f_hi.x *= (float)g_ptr[4];
                f_hi.y *= (float)g_ptr[5];
                f_hi.z *= (float)g_ptr[6];
                f_hi.w *= (float)g_ptr[7];
            }
#else
            FfnFastVec4T g_lo_h = ffn_load_gamma_half4(gamma, (ulong)k + 0ul); // INDEX64_OK
            f_lo *= float4(g_lo_h);
            if (vec_width == 8) {
                FfnFastVec4T g_hi_h = ffn_load_gamma_half4(gamma, (ulong)k + 4ul); // INDEX64_OK
                f_hi *= float4(g_hi_h);
            }
#endif
            xv_lo = FfnFastVec4T(f_lo);
            xv_hi = FfnFastVec4T(f_hi);
        }

        float4 xv_f32_lo = float4(xv_lo);
        float4 xv_f32_hi = float4(xv_hi);

        const uint block_off_gate = k / weights_per_block_gate;
        const uint block_off_up = same_wpb ? block_off_gate : (k / weights_per_block_up);
        float s_gate_val = (float)PolicyGate::load_scale(row_s_gate, block_off_gate);
        float s_up_val = (float)PolicyUp::load_scale(row_s_up, block_off_up);
        float a_gate_val = (PolicyGate::HAS_AFFINE ? (float)PolicyGate::load_affine(row_s_gate, block_off_gate) : 0.0f);
        float a_up_val = (PolicyUp::HAS_AFFINE ? (float)PolicyUp::load_affine(row_s_up, block_off_up) : 0.0f);

        {
            float w[vec_width];
            PolicyGate::template load_weights<vec_width>(w_gate, row_idx * k_dim + k, w);
            acc_gate += s_gate_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (PolicyGate::HAS_AFFINE ? (a_gate_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }

        {
            float w[vec_width];
            PolicyUp::template load_weights<vec_width>(w_up, row_idx * k_dim + k, w);
            acc_up += s_up_val * (dot(xv_f32_lo, float4(w[0],w[1],w[2],w[3])) + dot(xv_f32_hi, float4(w[4],w[5],w[6],w[7])))
                + (PolicyUp::HAS_AFFINE ? (a_up_val * (dot(xv_f32_lo, float4(1.0f)) + dot(xv_f32_hi, float4(1.0f)))) : 0.0f);
        }
    }

    return float2(acc_gate, acc_up);
}

template<typename Policy, uint vec_width, bool has_norm>
ALWAYS_INLINE float2 run_ffn_dual_project_stage_uniform(
    const device uchar* w_gate,
    const device uchar* s_gate,
    const device uchar* w_up,
    const device uchar* s_up,
    const device InputStorageT* input,
    constant uint& k_dim,
    constant uint& n_dim,
    constant uint& weights_per_block,
    const device GammaStorageT* gamma,
    float inv_rms,
    uint lane_id,
    uint row_idx,
    uint batch_idx
) {
    return run_ffn_dual_project_stage<Policy, Policy, vec_width, has_norm>(
        w_gate,
        s_gate,
        w_up,
        s_up,
        input,
        k_dim,
        n_dim,
        weights_per_block,
        weights_per_block,
        gamma,
        inv_rms,
        lane_id,
        row_idx,
        batch_idx
    );
}
