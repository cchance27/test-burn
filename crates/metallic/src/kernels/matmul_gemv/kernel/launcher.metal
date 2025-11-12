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
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
        } else {
            run_gemv_q8_canonical<false, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
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
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
            return;
        }
        case GemvLoaderQ8CanonicalBias: {
            run_gemv_q8_canonical<true, false>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
            return;
        }
        case GemvLoaderQ8CanonicalDebug: {
            run_gemv_q8_canonical<false, true>(
                matrix_data, scale_bytes, vector_x, result_y, params, bias, residual, alpha, beta, diag_col, gid, lid, x_tile);
            return;
        }
        default: {
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
