use super::types::SoftmaxVariant;

/// Preferences for softmax dispatcher loaded from environment variables.
#[derive(Debug, Default)]
pub struct Prefs {
    pub forced_variant: Option<SoftmaxVariant>,
    pub forced_tg_size: Option<usize>,
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
    let variant = std::env::var("METALLIC_SOFTMAX_VARIANT").ok().and_then(|s| parse_variant(&s));

    let tg_size = std::env::var("METALLIC_SOFTMAX_TG_SIZE").ok().and_then(|s| s.parse::<usize>().ok());

    Prefs {
        forced_variant: variant,
        forced_tg_size: tg_size,
    }
}
