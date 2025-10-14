use super::types::{SoftmaxBackend, SoftmaxPolicy, SoftmaxShape, SoftmaxVariant};

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
pub fn select_variant_and_tg(
    seq_k: usize,
    caps: &SoftmaxCaps,
    prefs: &SoftmaxPrefs,
) -> (SoftmaxVariant, usize) {
    if let Some(forced_variant) = prefs.forced_variant {
        let tg_size = prefs.forced_tg_size.unwrap_or_else(default_tg_size);
        return (forced_variant, tg_size);
    }

    let tg_size = nearest_pow2_bounded(seq_k, caps.max_threads_per_threadgroup);

    // Heuristic from GGML-METALLIC.md and 5.1-todo.md
    // For now, we only have one custom kernel, so we don't differentiate between Vec and Block.
    // This scaffolding allows us to do so in the future. 
    // TODO: This should likely be based on the capabilities of the device
    if seq_k <= 1024 {
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
            // Default to Custom for now, as it's the one we are optimizing.
            // We can add more sophisticated logic later (e.g., based on shape or variant).
            SoftmaxBackend::Custom
        }
    };

    SoftmaxPolicy {
        backend,
        variant,
        threadgroup_size,
    }
}