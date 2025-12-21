// Kernel launchers and mixed-mode entrypoints


#define PARAMS_ARGS params->K, params->N, params->blocks_per_k, params->weights_per_block, params->batch, params->stride_x, params->stride_y, params->stride_a, params->stride_w, params->stride_scale

ALWAYS_INLINE void gemv_run_loader_mode(
    GemvLoaderMode loader_mode,
    const device uchar *matrix_data,
    const device uchar *scale_bytes,
    const device half *vector_x,
    device half *result_y,
    const constant GemvParams *params,
    const device half *bias,
    const device half *residual,
    float alpha,
    float beta,
    const constant uint &diag_col,
    uint3 gid,
    uint3 lid) {

    const bool is_q8 = (loader_mode == GemvLoaderQ8Canonical)
        || (loader_mode == GemvLoaderQ8CanonicalBias)
        || (loader_mode == GemvLoaderQ8CanonicalDebug);
    
    if (is_q8) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
        const bool debug = (loader_mode == GemvLoaderQ8CanonicalDebug);
        if (debug) {
             run_gemv_q8_canonical<false, true>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        } else if (wants_bias) {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        } else {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        }
        return;
    }

    // FP16 Dense & Canonical Dispatch (SIMD Paths)
    device half *res_arr[1] = {result_y};
    const uint N_arr[1] = {params->N};

    switch (loader_mode) {
        case GemvLoaderDense: {
             // Unified SIMD Path (Dense)
             const device half *bias_arr[1] = {bias}; // Unused but passed
             const uint bias_flags[1] = {0u};
             run_simd_f16_gemv_cols8<1, false>(
                 (const device half *)matrix_data, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid,
                 PARAMS_ARGS
             );
             return;
        }
        case GemvLoaderDenseBias: {
             const device half *bias_arr[1] = {bias};
             const uint bias_flags[1] = {1u};
             run_simd_f16_gemv_cols8<1, true>(
                 (const device half *)matrix_data, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid,
                 PARAMS_ARGS
             );
             return;
        }
        case GemvLoaderDenseCanonical: {
            const device half *data_arr[1] = { (const device half *)matrix_data };
            const device half *bias_arr[1] = {bias};
            const uint bias_flags[1] = {0u};
            SimdGemvPolicyF16Canonical::Params p = { (const device half **)data_arr, params->weights_per_block };
            GemvParams gp = { PARAMS_ARGS };
            run_simd_gemv_template<SimdGemvPolicyF16Canonical, 1, 4, false>(
                p, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid, gp
            );
            return;
        }
        case GemvLoaderDenseCanonicalBias: {
            const device half *data_arr[1] = { (const device half *)matrix_data };
            const device half *bias_arr[1] = {bias};
            const uint bias_flags[1] = {1u};
            SimdGemvPolicyF16Canonical::Params p = { (const device half **)data_arr, params->weights_per_block };
            GemvParams gp = { PARAMS_ARGS };
            run_simd_gemv_template<SimdGemvPolicyF16Canonical, 1, 4, true>(
                p, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid, gp
            );
            return;
        }
        case GemvLoaderDenseStrided: {
             const device half *bias_arr[1] = {bias};
             const uint bias_flags[1] = {0u};
             run_simd_f16_gemv_strided<1, false>(
                 (const device half *)matrix_data, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid,
                 PARAMS_ARGS
             );
             return;
        }
        case GemvLoaderDenseStridedBias: {
             const device half *bias_arr[1] = {bias};
             const uint bias_flags[1] = {1u};
             run_simd_f16_gemv_strided<1, true>(
                 (const device half *)matrix_data, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid,
                 PARAMS_ARGS
             );
             return;
        }
        default: {
             // Fallback to Dense (No Bias)
             const device half *bias_arr[1] = {bias};
             const uint bias_flags[1] = {0u};
             run_simd_f16_gemv<1, false>(
                 (const device half *)matrix_data, vector_x, res_arr, N_arr, params->K, bias_arr, bias_flags, alpha, beta, residual, gid, lid,
                 PARAMS_ARGS
             );
             return;
        }
    }
}

// Unified GEMV entry for dense FP16 and Q8 using loader_mode
[[kernel]] void gemv_f16(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    gemv_run_loader_mode(
        static_cast<GemvLoaderMode>(loader_mode),
        matrix_data,
        scale_bytes,
        vector_x,
        result_y,
        params,
        bias,
        residual,
        alpha,
        beta,
        diag_col,
        gid,
        lid);
}

// Dense GEMV entry tuned for COLS_PER_TG=2 (threadgroup width 64)
[[kernel]] void gemv_f16_cols2(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    (void)scale_bytes;
    (void)diag_col;
    switch (static_cast<GemvLoaderMode>(loader_mode)) {
        case GemvLoaderDense: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 0 };
            run_simd_f16_gemv_cols2<1, false>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_gemv_cols2<1, true>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseCanonical: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 0 };
            run_simd_f16_canonical_gemv_cols2<1, false>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseCanonicalBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_canonical_gemv_cols2<1, true>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        default: {
            return;
        }
    }
}

// Dense GEMV entry tuned for COLS_PER_TG=8 (threadgroup width 256)
[[kernel]] void gemv_f16_cols8(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    (void)scale_bytes;
    (void)diag_col;
    switch (static_cast<GemvLoaderMode>(loader_mode)) {
        case GemvLoaderDense: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 0 };
            run_simd_f16_gemv_cols8<1, false>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_gemv_cols8<1, true>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseCanonical: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 0 };
            run_simd_f16_canonical_gemv_cols8<1, false>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseCanonicalBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_canonical_gemv_cols8<1, true>(
                (const device half *)matrix_data,
                vector_x,
                r_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        default: {
            return;
        }
    }
}

// RMSNorm-fused GEMV entry (Dense only)
[[kernel]] void gemv_f16_rmsnorm(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    const device half *gamma [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    (void)scale_bytes;
    (void)diag_col;
    threadgroup float inv_rms_s;
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    const float inv_rms = gemv_compute_inv_rms(vector_x, params->K, lane_id, warp_id, &inv_rms_s);

    switch (static_cast<GemvLoaderMode>(loader_mode)) {
        case GemvLoaderDense: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 0 };
            run_simd_f16_gemv_rmsnorm_cols8<1, false>(
                (const device half *)matrix_data,
                vector_x,
                gamma,
                inv_rms,
                r_arr,
                n_arr,
                params->K,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_gemv_rmsnorm_cols8<1, true>(
                (const device half *)matrix_data,
                vector_x,
                gamma,
                inv_rms,
                r_arr,
                n_arr,
                params->K,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseCanonical: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 0 };
            run_simd_f16_canonical_gemv_rmsnorm<1, false>(
                (const device half *)matrix_data,
                vector_x,
                gamma,
                inv_rms,
                r_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        case GemvLoaderDenseCanonicalBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_canonical_gemv_rmsnorm<1, true>(
                (const device half *)matrix_data,
                vector_x,
                gamma,
                inv_rms,
                r_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                b_arr,
                hb_arr,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        default: {
            return;
        }
    }
}

// Optimized entry point for Q8 without Shared Memory allocation (High Occupancy)
[[kernel]] void gemv_q8_entry(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
    
    // NO x_tile allocation.
    
    // Direct dispatch to Q8 Canonical logic
    if (loader_mode == GemvLoaderQ8Canonical || loader_mode == GemvLoaderQ8CanonicalBias) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
        if (wants_bias) {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        } else {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        }
        return;
    }
    
    // Fallback for switch-based dispatch if needed (only Q8 supported essentially)
    switch (loader_mode) {
        case GemvLoaderQ8Canonical: {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
            return;
        }
        case GemvLoaderQ8CanonicalBias: {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
            return;
        }
        case GemvLoaderQ8CanonicalDebug: {
            run_gemv_q8_canonical<false, true>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
            return;
        }
    }
}

// Q8 GEMV entry tuned for COLS_PER_TG=2 (threadgroup width 64)
[[kernel]] void gemv_q8_entry_cols2(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    if (loader_mode == GemvLoaderQ8Canonical || loader_mode == GemvLoaderQ8CanonicalBias) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
        const device uchar *data_arr[1] = { matrix_data };
        const device uchar *scale_arr[1] = { scale_bytes };
        device half *res_arr[1] = { result_y };
        const uint n_arr[1] = { params->N };
        const device half *bias_arr[1] = { bias };
        const uint bias_flags[1] = { wants_bias ? 1u : 0u };

        run_simd_q8_gemv_cols2<1, true>(
            data_arr,
            scale_arr,
            vector_x,
            res_arr,
            n_arr,
            params->K,
            params->weights_per_block,
            bias_arr,
            bias_flags,
            alpha,
            beta,
            residual,
            gid,
            lid,
            PARAMS_ARGS
        );
        return;
    }

    switch (loader_mode) {
        case GemvLoaderQ8Canonical:
        case GemvLoaderQ8CanonicalBias:
        case GemvLoaderQ8CanonicalDebug: {
            const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
            const device uchar *data_arr[1] = { matrix_data };
            const device uchar *scale_arr[1] = { scale_bytes };
            device half *res_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *bias_arr[1] = { bias };
            const uint bias_flags[1] = { wants_bias ? 1u : 0u };

            run_simd_q8_gemv_cols2<1, true>(
                data_arr,
                scale_arr,
                vector_x,
                res_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                bias_arr,
                bias_flags,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
    }
}

// Q8 GEMV entry tuned for COLS_PER_TG=8 (threadgroup width 256)
[[kernel]] void gemv_q8_entry_cols8(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    if (loader_mode == GemvLoaderQ8Canonical || loader_mode == GemvLoaderQ8CanonicalBias) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
        const device uchar *data_arr[1] = { matrix_data };
        const device uchar *scale_arr[1] = { scale_bytes };
        device half *res_arr[1] = { result_y };
        const uint n_arr[1] = { params->N };
        const device half *bias_arr[1] = { bias };
        const uint bias_flags[1] = { wants_bias ? 1u : 0u };

        run_simd_q8_gemv_cols8<1, true>(
            data_arr,
            scale_arr,
            vector_x,
            res_arr,
            n_arr,
            params->K,
            params->weights_per_block,
            bias_arr,
            bias_flags,
            alpha,
            beta,
            residual,
            gid,
            lid,
            PARAMS_ARGS
        );
        return;
    }

    switch (loader_mode) {
        case GemvLoaderQ8Canonical:
        case GemvLoaderQ8CanonicalBias:
        case GemvLoaderQ8CanonicalDebug: {
            const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
            const device uchar *data_arr[1] = { matrix_data };
            const device uchar *scale_arr[1] = { scale_bytes };
            device half *res_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *bias_arr[1] = { bias };
            const uint bias_flags[1] = { wants_bias ? 1u : 0u };

            run_simd_q8_gemv_cols8<1, true>(
                data_arr,
                scale_arr,
                vector_x,
                res_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                bias_arr,
                bias_flags,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
    }
}

// RMSNorm-fused GEMV entry for Q8 (no x_tile allocation)
[[kernel]] void gemv_q8_rmsnorm_entry(
    const device uchar *matrix_data [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    device half *result_y [[buffer(2)]],
    const constant GemvParams *params [[buffer(3)]],
    const device half *bias [[buffer(4)]],
    constant uint &loader_mode [[buffer(5)]],
    const device uchar *scale_bytes [[buffer(6)]],
    const constant uint &diag_col [[buffer(8)]],
    const device half *residual [[buffer(7)]],
    constant float &alpha [[buffer(9)]],
    constant float &beta [[buffer(10)]],
    const device half *gamma [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

    threadgroup float inv_rms_s;
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    const float inv_rms = gemv_compute_inv_rms(vector_x, params->K, lane_id, warp_id, &inv_rms_s);

    if (loader_mode == GemvLoaderQ8Canonical || loader_mode == GemvLoaderQ8CanonicalBias) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
        const device uchar *data_arr[1] = { matrix_data };
        const device uchar *scale_arr[1] = { scale_bytes };
        device half *res_arr[1] = { result_y };
        const uint n_arr[1] = { params->N };
        const device half *bias_arr[1] = { bias };
        const uint bias_flags[1] = { wants_bias ? 1u : 0u };

        run_simd_q8_gemv_rmsnorm<1, true>(
            data_arr,
            scale_arr,
            vector_x,
            gamma,
            inv_rms,
            res_arr,
            n_arr,
            params->K,
            params->weights_per_block,
            bias_arr,
            bias_flags,
            alpha,
            beta,
            residual,
            gid,
            lid,
            PARAMS_ARGS
        );
        return;
    }

    switch (loader_mode) {
        case GemvLoaderQ8Canonical:
        case GemvLoaderQ8CanonicalBias:
        case GemvLoaderQ8CanonicalDebug: {
            const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias);
            const device uchar *data_arr[1] = { matrix_data };
            const device uchar *scale_arr[1] = { scale_bytes };
            device half *res_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *bias_arr[1] = { bias };
            const uint bias_flags[1] = { wants_bias ? 1u : 0u };

            run_simd_q8_gemv_rmsnorm<1, true>(
                data_arr,
                scale_arr,
                vector_x,
                gamma,
                inv_rms,
                res_arr,
                n_arr,
                params->K,
                params->weights_per_block,
                bias_arr,
                bias_flags,
                alpha,
                beta,
                residual,
                gid,
                lid,
                PARAMS_ARGS
            );
            return;
        }
        default: {
            return;
        }
    }
}
