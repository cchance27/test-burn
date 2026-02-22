use metallic_env::GEMV_F16_COLS8;

#[inline]
pub(super) fn use_f16_cols8() -> bool {
    // Default ON: this path mirrors the legacy Context RowMajor FP16 GEMV pointer arithmetic and is consistently faster
    // for decode-heavy shapes (e.g. K=896, K=4864). Allow an escape hatch to disable for debugging/regressions.
    GEMV_F16_COLS8.get().ok().flatten().unwrap_or(true)
}
