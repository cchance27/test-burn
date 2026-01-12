use metallic_env::FORCE_MATMUL_BACKEND_VAR;

// Capacity tiering function to reduce the number of buffer allocations during incremental decoding
pub fn tier_up_capacity(seq: usize) -> usize {
    match seq {
        0..=16 => 16,
        17..=32 => 32,
        33..=64 => 64,
        65..=128 => 128,
        129..=256 => 256,
        257..=512 => 512,
        513..=1024 => 1024,
        1025..=2048 => 2048,
        2049..=4096 => 4096,
        _ => seq.div_ceil(1024) * 1024, // Round up to the next 1KB boundary for very large sequences
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryUsage {
    pub pool_used: usize,
    pub pool_capacity: usize,
    pub kv_used: usize,
    pub kv_capacity: usize,
    pub kv_cache_bytes: usize,
}

use rustc_hash::FxHashMap;

#[derive(Clone, Debug)]
pub struct GpuProfilerLabel {
    pub op_name: String,
    pub backend: String,
    pub data: Option<FxHashMap<String, String>>,
}

impl GpuProfilerLabel {
    pub fn new(op_name: String, backend: String) -> Self {
        Self {
            op_name,
            backend,
            data: None,
        }
    }

    pub fn with_data(op_name: String, backend: String, data: FxHashMap<String, String>) -> Self {
        Self {
            op_name,
            backend,
            data: Some(data),
        }
    }

    pub fn fallback(op_name: &str) -> Self {
        Self {
            op_name: op_name.to_string(),
            backend: GPU_PROFILER_BACKEND.to_string(),
            data: None,
        }
    }
}

pub const GPU_PROFILER_BACKEND: &str = "Metal";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulBackend {
    Gemv,
    Mlx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatMulBackendOverride {
    Default,
    Force(MatMulBackend),
    Auto,
}

pub fn detect_forced_matmul_backend() -> MatMulBackendOverride {
    match FORCE_MATMUL_BACKEND_VAR.get() {
        Ok(Some(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                MatMulBackendOverride::Default
            } else {
                match trimmed.to_ascii_lowercase().as_str() {
                    "mlx" => MatMulBackendOverride::Force(MatMulBackend::Mlx),
                    "mps" => MatMulBackendOverride::Force(MatMulBackend::Mlx),
                    "gemv" => MatMulBackendOverride::Force(MatMulBackend::Gemv),
                    "auto" => MatMulBackendOverride::Auto,
                    _ => MatMulBackendOverride::Default,
                }
            }
        }
        _ => MatMulBackendOverride::Default,
    }
}
