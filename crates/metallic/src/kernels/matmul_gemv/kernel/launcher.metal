// Kernel launchers and mixed-mode entrypoints

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
    uint3 lid,
    threadgroup float *x_tile) {

    if (params->weights_per_block != 0u && loader_mode != GemvLoaderQ8CanonicalDebug) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias) || (loader_mode == GemvLoaderDenseBias);
        if (wants_bias) {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        } else {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid);
        }
        return;
    }

    switch (loader_mode) {
        case GemvLoaderDense: {
             const device half *matrix_a = (const device half *)matrix_data;
             run_gemv_dense<false>(matrix_a, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, x_tile);
             return;
        }
        case GemvLoaderDenseBias: {
             const device half *matrix_a = (const device half *)matrix_data;
             run_gemv_dense<true>(matrix_a, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, x_tile);
             return;
        }
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
        default: {
             // Fallback to legacy or error?
             // Legacy required X_tile.
             // But we are in gemv_run_loader_mode which has x_tile arg.
             // We can keep legacy fallback for default?
             const device half *matrix_a = (const device half *)matrix_data;
             run_gemv_dense<false>(matrix_a, vector_x, result_y, params, bias, residual, alpha, beta, gid, lid, x_tile);
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
    threadgroup float x_tile[TILE_K];

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
        lid,
        x_tile);
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
                lid
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
                lid
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
                lid
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
                lid
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
            run_simd_f16_gemv_rmsnorm<1, false>(
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
                lid
            );
            return;
        }
        case GemvLoaderDenseBias: {
            device half *r_arr[1] = { result_y };
            const uint n_arr[1] = { params->N };
            const device half *b_arr[1] = { bias };
            const uint hb_arr[1] = { 1 };
            run_simd_f16_gemv_rmsnorm<1, true>(
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
                lid
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
    if (params->weights_per_block != 0u && loader_mode != GemvLoaderQ8CanonicalDebug) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias) || (loader_mode == GemvLoaderDenseBias);
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

    if (params->weights_per_block != 0u && loader_mode != GemvLoaderQ8CanonicalDebug) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias) || (loader_mode == GemvLoaderDenseBias);
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
            lid
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
                lid
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

    if (params->weights_per_block != 0u && loader_mode != GemvLoaderQ8CanonicalDebug) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias) || (loader_mode == GemvLoaderDenseBias);
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
            lid
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
                lid
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

    if (params->weights_per_block != 0u && loader_mode != GemvLoaderQ8CanonicalDebug) {
        const bool wants_bias = (loader_mode == GemvLoaderQ8CanonicalBias) || (loader_mode == GemvLoaderDenseBias);
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
            lid
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
                lid
            );
            return;
        }
        default: {
            return;
        }
    }
}
