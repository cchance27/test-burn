use crate::kernels::matmul_dispatcher::types::{MatmulBackend, Prefs};
use std::env;

fn parse_backend(s: &str) -> MatmulBackend {
    match s.to_ascii_lowercase().as_str() {
        "mlx" => MatmulBackend::Mlx,
        "mps" => MatmulBackend::Mps,
        "custom" => MatmulBackend::Custom,
        "auto" => MatmulBackend::Auto,
        _ => MatmulBackend::Auto,
    }
}

pub fn load_prefs_from_env() -> Prefs {
    let backend = env::var("METALLIC_MATMUL_BACKEND")
        .ok()
        .map(|s| parse_backend(&s))
        .unwrap_or(MatmulBackend::Auto);
    let force_smalln = env::var("METALLIC_MATMUL_FORCE_SMALLN")
        .ok()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    Prefs { backend, force_smalln }
}
