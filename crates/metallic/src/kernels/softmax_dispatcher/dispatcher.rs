use super::types::{SoftmaxBackend, SoftmaxPolicy, SoftmaxShape, SoftmaxVariant};

// Tunable threshold for selecting vec vs block softmax.
// Updated based on benchmark analysis (see benches/analyze_crossover_points.py results).
const SOFTMAX_VEC_BLOCK_THRESHOLD: usize = 1280;

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
    pub forced_backend: Option<SoftmaxBackend>,
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

/// Selects the softmax variant and threadgroup size based on shape, capabilities, and preferences.
pub fn select_variant_and_tg(seq_k: usize, caps: &SoftmaxCaps, prefs: &SoftmaxPrefs) -> (SoftmaxVariant, usize) {
    if let Some(forced_variant) = prefs.forced_variant {
        let tg_size = prefs.forced_tg_size.unwrap_or_else(default_tg_size);
        return (forced_variant, tg_size);
    }

    let tg_size = nearest_pow2_bounded(seq_k, caps.max_threads_per_threadgroup);

    // Heuristic from GGML-METALLIC.md and 5.1-todo.md
    // Choose Vec for shorter sequences, Block for longer ones.
    // TODO: Consider device capabilities and dtype in future refinements.
    if seq_k <= SOFTMAX_VEC_BLOCK_THRESHOLD {
        (SoftmaxVariant::Vec, tg_size)
    } else {
        (SoftmaxVariant::Block, tg_size)
    }
}

/// Selects the overall policy for softmax execution.
pub fn select_policy(shape: SoftmaxShape, caps: &SoftmaxCaps, prefs: &SoftmaxPrefs) -> SoftmaxPolicy {
    let (variant, threadgroup_size) = select_variant_and_tg(shape.seq_k, caps, prefs);

    // Backend selection logic
    let backend = match prefs.forced_backend {
        Some(backend) => backend,
        None => {
            // For now we select MPS for auto case, but in the future we might want logic to select
            // based on sequence length, dtype support, or other factors.
            // For small sequence lengths MPS might be more efficient, while for larger sequences
            // our custom kernels might be better.
            SoftmaxBackend::Auto // Will allow dynamic selection based on dtype and other factors
        }
    };

    SoftmaxPolicy {
        backend,
        variant,
        threadgroup_size,
    }
}
