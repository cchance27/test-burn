// Tunable thresholds with environment overrides for benchmarking.
// Defaults chosen conservatively; adjust via env to A/B test without code changes.

pub fn smalln_max_n() -> usize {
    // METALLIC_MATMUL_SMALLN_MAX_N, default 8
    std::env::var("METALLIC_MATMUL_SMALLN_MAX_N")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(8)
}

pub fn simd_m_min() -> usize {
    std::env::var("METALLIC_MATMUL_SIMD_M_MIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(64)
}

pub fn simd_n_min() -> usize {
    std::env::var("METALLIC_MATMUL_SIMD_N_MIN")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(16)
}
