use super::types::{SoftmaxBackend, SoftmaxVariant};

/// Preferences for softmax dispatcher loaded from environment variables.
#[derive(Debug, Default)]
pub struct Prefs {
    pub forced_backend: Option<SoftmaxBackend>,
    pub forced_variant: Option<SoftmaxVariant>,
    pub forced_tg_size: Option<usize>,
}

fn parse_backend(s: &str) -> Option<SoftmaxBackend> {
    match s.to_ascii_lowercase().as_str() {
        "kernel" | "compute" | "pipeline" => Some(SoftmaxBackend::Custom),
        "mps" | "metal" => Some(SoftmaxBackend::MPS),
        "auto" | "" => Some(SoftmaxBackend::Auto),
        _ => None,
    }
}

fn parse_variant(s: &str) -> Option<SoftmaxVariant> {
    match s.to_ascii_lowercase().as_str() {
        "vec" => Some(SoftmaxVariant::Vec),
        "block" => Some(SoftmaxVariant::Block),
        "auto" | "" => Some(SoftmaxVariant::Auto),
        _ => None,
    }
}

pub fn load_prefs_from_env() -> Prefs {
    let backend = std::env::var("METALLIC_SOFTMAX_BACKEND")
        .ok()
        .and_then(|s| parse_backend(&s));

    let variant = std::env::var("METALLIC_SOFTMAX_VARIANT")
        .ok()
        .and_then(|s| parse_variant(&s));

    let tg_size = std::env::var("METALLIC_SOFTMAX_TG_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());

    Prefs {
        forced_backend: backend,
        forced_variant: variant,
        forced_tg_size: tg_size,
    }
}