use metallic_env::FORCE_MATMUL_BACKEND;

use crate::kernels::matmul_dispatcher::types::{MatmulBackend, Prefs};

fn parse_backend(s: &str) -> MatmulBackend {
    match s.to_ascii_lowercase().as_str() {
        "mlx" => MatmulBackend::Mlx,
        "mps" => MatmulBackend::Mps,
        "gemv" => MatmulBackend::Gemv,
        "auto" => MatmulBackend::Auto,
        _ => MatmulBackend::Auto,
    }
}

pub fn load_prefs_from_env() -> Prefs {
    let backend = FORCE_MATMUL_BACKEND
        .get()
        .ok()
        .flatten()
        .map(|s| parse_backend(&s))
        .unwrap_or(MatmulBackend::Auto);
    // For now, keep smallN force via raw env until a typed var is added
    let force_smalln = std::env::var("METALLIC_MATMUL_FORCE_SMALLN")
        .ok()
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    Prefs { backend, force_smalln }
}
