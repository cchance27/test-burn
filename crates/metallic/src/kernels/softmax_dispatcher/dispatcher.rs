use super::types::{SoftmaxPolicy, SoftmaxShape, SoftmaxVariant};

// TODO: Move to caps.rs
#[derive(Debug)]
pub struct SoftmaxCaps {
    pub max_threads_per_threadgroup: usize,
    // pub has_simdgroup_reductions: bool, // TODO: confirm this
}

impl Default for SoftmaxCaps {
    fn default() -> Self {
        Self {
            max_threads_per_threadgroup: 1024, // Conservative default // TODO: confirm these values
        }
    }
}

// TODO: Move to prefs.rs and integrate with metallic_env
#[derive(Debug, Default)]
pub struct SoftmaxPrefs {
    pub forced_variant: Option<SoftmaxVariant>,
    pub forced_tg_size: Option<usize>,
}

/// Selects the nearest power of two for the threadgroup size, bounded by a maximum.
fn nearest_pow2_bounded(val: usize, max_bound: usize) -> usize {
    if val == 0 {
        return 0;
    }
    // Find the next power of 2, or return max_bound if it exceeds it.
    val.next_power_of_two().min(max_bound)
}

fn default_tg_size() -> usize {
    256 // Matches legacy softmax kernel
}

fn classify_variant(seq_k: usize) -> SoftmaxVariant {
    match seq_k {
        0..=255 => SoftmaxVariant::Auto, // kernel path wins for very short rows
        256..=767 => SoftmaxVariant::Vec,
        768..=895 => SoftmaxVariant::Block,
        896..=1023 => SoftmaxVariant::Vec,
        1024..=1279 => SoftmaxVariant::Block,
        1280..=2047 => SoftmaxVariant::Vec,
        2048..=8191 => SoftmaxVariant::Auto,
        _ => SoftmaxVariant::Vec, // very long rows fall back to custom vec kernel
    }
}

fn select_threadgroup_size(variant: SoftmaxVariant, seq_k: usize, caps: &SoftmaxCaps, prefs: &SoftmaxPrefs) -> usize {
    if let Some(forced) = prefs.forced_tg_size {
        return forced;
    }

    match variant {
        SoftmaxVariant::Vec | SoftmaxVariant::Block => nearest_pow2_bounded(seq_k, caps.max_threads_per_threadgroup),
        SoftmaxVariant::Auto => default_tg_size(),
    }
}

/// Selects the overall policy for softmax execution.
pub fn select_policy(shape: SoftmaxShape, caps: &SoftmaxCaps, prefs: &SoftmaxPrefs) -> SoftmaxPolicy {
    // Establish baseline strategy from benchmark-driven heuristics.
    let mut variant = classify_variant(shape.seq_k);

    // Respect user overrides.
    if let Some(forced_variant) = prefs.forced_variant {
        variant = forced_variant;
    }

    let threadgroup_size = select_threadgroup_size(variant, shape.seq_k, caps, prefs);
    SoftmaxPolicy { variant, threadgroup_size }
}
