//! Dynamic context and buffer management for Foundry.
//!
//! Supports LLM, DiT, audio, video, MoE workloads via abstract context and eviction policies.

use sysinfo::System;

use crate::{spec::Architecture, types::Device};

/// Configuration for context length and buffer management.
#[derive(Debug)]
pub struct ContextConfig {
    /// Maximum context/sequence length allowed for this session.
    pub max_context_len: usize,
    /// Current capacity allocated in buffers (must be <= max_context_len).
    pub allocated_capacity: usize,
    /// Memory budget configuration.
    pub memory_budget: MemoryBudget,
    /// Strategy for growing buffers.
    pub growth_strategy: GrowthStrategy,
    /// Policy for handling context overflow.
    pub eviction_policy: Box<dyn EvictionPolicy>,
}

/// Memory budget configuration.
#[derive(Debug, Clone, Copy)]
pub enum MemoryBudget {
    /// Auto-detect from GPU (uses recommendedMaxWorkingSetSize with percentage).
    Auto { percentage: f32 },
    /// Explicit budget in bytes.
    Explicit(usize),
}

/// Strategy for growing buffers over time.
#[derive(Debug, Clone, Copy)]
pub enum GrowthStrategy {
    /// Reserve the full max_context_len immediately at startup.
    FullReserve,
    /// Start with an initial size and grow geometrically as needed.
    GrowOnDemand { initial: usize, growth_factor: f32 },
}

impl Default for GrowthStrategy {
    fn default() -> Self {
        GrowthStrategy::GrowOnDemand {
            initial: 2048,
            growth_factor: 2.0,
        }
    }
}

/// Specialized error for context eviction failures.
#[derive(thiserror::Error, Debug)]
pub enum EvictionError {
    #[error("Eviction not supported by this policy")]
    NotSupported,
    #[error("Failed to evict enough space: requested {requested}, but could only free {freed}")]
    InsufficientSpace { requested: usize, freed: usize },
}

/// Policy for handling context overflow.
pub trait EvictionPolicy: std::fmt::Debug + Send + Sync {
    /// Name for debug logging.
    fn name(&self) -> &'static str;

    /// Whether this policy supports evicting existing context.
    fn can_evict(&self) -> bool;

    /// Evict tokens to make room for `requested` new tokens.
    ///
    /// Returns the new valid logical range [start_pos, end_pos).
    fn evict(&self, current_len: usize, requested: usize, max: usize) -> Result<(usize, usize), EvictionError>;
}

/// Default policy that simply errors on overflow.
#[derive(Debug, Default, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct NoEviction;

impl EvictionPolicy for NoEviction {
    fn name(&self) -> &'static str {
        "NoEviction"
    }
    fn can_evict(&self) -> bool {
        false
    }
    fn evict(&self, _current_len: usize, _requested: usize, _max: usize) -> Result<(usize, usize), EvictionError> {
        Err(EvictionError::NotSupported)
    }
}

/// Memory estimate for a model's KV cache.
#[derive(Debug, Clone, Copy)]
pub struct MemoryEstimate {
    pub kv_cache_bytes: usize,
    pub per_layer_bytes: usize,
}

impl ContextConfig {
    /// Creates a new ContextConfig from architectural parameters and environment overrides.
    ///
    /// Rules:
    /// - Default maximum is the model's `arch.max_seq_len` when present.
    /// - Callers may provide an explicit *cap* (DSL/runtime), and/or `METALLIC_MAX_CONTEXT_LEN`.
    ///   These act as upper bounds and are clamped to the model max (never exceed the model).
    /// - If the model does not specify a max (0), we fall back to the cap/env/default.
    pub fn from_architecture(arch: &Architecture, override_len: Option<usize>) -> Self {
        let env_cap = std::env::var("METALLIC_MAX_CONTEXT_LEN")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());

        let requested_cap = override_len.or(env_cap).filter(|v| *v > 0);
        let model_max: Option<usize> = (arch.max_seq_len() > 0).then_some(arch.max_seq_len());

        let max_len = match (model_max, requested_cap) {
            (Some(model), Some(cap)) => model.min(cap),
            (Some(model), None) => model,
            (None, Some(cap)) => cap,
            (None, None) => 2048,
        }
        .max(1);

        let full_reserve = std::env::var("METALLIC_FULL_CONTEXT_RESERVE").is_ok();
        let growth_strategy = if full_reserve {
            GrowthStrategy::FullReserve
        } else {
            GrowthStrategy::default()
        };

        let initial_capacity = match growth_strategy {
            GrowthStrategy::FullReserve => max_len,
            GrowthStrategy::GrowOnDemand { initial, .. } => initial.min(max_len),
        };

        let budget_mb = std::env::var("METALLIC_KV_MEMORY_BUDGET_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());

        let memory_budget = if let Some(mb) = budget_mb {
            MemoryBudget::Explicit(mb * 1024 * 1024)
        } else {
            MemoryBudget::Auto { percentage: 0.5 }
        };

        Self {
            max_context_len: max_len,
            allocated_capacity: Self::align_capacity(initial_capacity),
            memory_budget,
            growth_strategy,
            eviction_policy: Box::new(NoEviction),
        }
    }

    /// Apply the configured memory budget to clamp `max_context_len` and `allocated_capacity`.
    ///
    /// This is intentionally separated from `from_architecture()` so the caller can provide the
    /// actual `MTLDevice` used by Foundry.
    pub fn apply_memory_budget(&mut self, device: &Device, arch: &Architecture) {
        let budget_bytes = match self.memory_budget {
            MemoryBudget::Explicit(bytes) => bytes,
            MemoryBudget::Auto { percentage } => {
                let pct = percentage.clamp(0.0, 1.0);
                (Self::gpu_recommended_memory(device) as f64 * pct as f64) as usize
            }
        };

        // KV cache bytes per token:
        // K: d_model f16 + V: d_model f16 => 2 * d_model * 2 bytes = 4*d_model bytes per layer.
        // Total: n_layers * 4*d_model.
        let per_token_bytes = arch
            .d_model()
            .checked_mul(4)
            .and_then(|v| v.checked_mul(arch.n_layers()))
            .unwrap_or(usize::MAX);

        if per_token_bytes == 0 || budget_bytes == 0 {
            return;
        }

        let max_tokens_by_budget = budget_bytes / per_token_bytes;
        if max_tokens_by_budget == 0 {
            // Too small to hold even 1 token of KV within budget; fail-fast at 1 and let
            // later allocations surface the real error with context.
            self.max_context_len = 1;
            self.allocated_capacity = Self::align_capacity(1);
            return;
        }

        let budget_cap = Self::align_capacity(max_tokens_by_budget).max(1);
        if budget_cap < self.max_context_len {
            tracing::info!(
                "Clamping max_context_len {} -> {} due to KV budget (budget={}MB, per_token={} bytes)",
                self.max_context_len,
                budget_cap,
                budget_bytes / 1024 / 1024,
                per_token_bytes
            );
            self.max_context_len = budget_cap;
        }

        if self.allocated_capacity > self.max_context_len {
            self.allocated_capacity = Self::align_capacity(self.max_context_len);
        }
    }

    /// Round a capacity value up to the nearest alignment boundary (128 elements).
    pub fn align_capacity(capacity: usize) -> usize {
        const ALIGN: usize = 128;
        if capacity == 0 {
            return ALIGN;
        }
        capacity.next_multiple_of(ALIGN)
    }

    /// Estimate the memory required for the KV cache.
    pub fn estimate_kv_memory(arch: &Architecture, context_len: usize) -> MemoryEstimate {
        let head_dim = arch.d_model() / arch.n_heads();
        // 2 (K+V) * n_layers * n_heads * context_len * head_dim * 2 (F16 bytes)
        let per_layer_bytes = 2 * arch.n_heads() * context_len * head_dim * 2;
        let total = per_layer_bytes * arch.n_layers();
        MemoryEstimate {
            kv_cache_bytes: total,
            per_layer_bytes,
        }
    }

    /// Get the recommended maximum working set size for the Metal device.
    pub fn gpu_recommended_memory(device: &Device) -> usize {
        device.recommended_max_working_set_size() as usize
    }

    /// Log the current system memory state using sysinfo.
    pub fn log_system_memory() {
        // Keep this lightweight: we only need memory totals, not process enumeration.
        let mut sys = System::new();
        sys.refresh_memory();
        tracing::debug!(
            "System Memory: Used {}MB / Total {}MB",
            sys.used_memory() / 1024 / 1024,
            sys.total_memory() / 1024 / 1024
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::Architecture;

    fn mock_arch(max_seq_len: usize) -> Architecture {
        use crate::spec::ArchValue;
        let mut params = rustc_hash::FxHashMap::default();
        params.insert("d_model".to_string(), ArchValue::USize(512));
        params.insert("n_heads".to_string(), ArchValue::USize(8));
        params.insert("n_kv_heads".to_string(), ArchValue::USize(2));
        params.insert("n_layers".to_string(), ArchValue::USize(4));
        params.insert("ff_dim".to_string(), ArchValue::USize(2048));
        params.insert("vocab_size".to_string(), ArchValue::USize(1000));
        params.insert("max_seq_len".to_string(), ArchValue::USize(max_seq_len));
        params.insert("rope_base".to_string(), ArchValue::F32(10000.0));
        params.insert("rms_eps".to_string(), ArchValue::F32(1e-6));

        Architecture {
            params,
            tensor_names: Default::default(),
            metadata_keys: Default::default(),
            prepare: Default::default(),
            weight_bindings: Vec::new(),
            forward: Vec::new(),
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_context_config_priority() {
        let arch = mock_arch(4096);

        // 1. Default to model max
        let config = ContextConfig::from_architecture(&arch, None);
        assert_eq!(config.max_context_len, 4096);

        // 2. DSL/runtime override acts as a cap
        let config2 = ContextConfig::from_architecture(&arch, Some(2048));
        assert_eq!(config2.max_context_len, 2048);

        // 3. Env var acts as a cap
        set_env_safe("METALLIC_MAX_CONTEXT_LEN", "1024");
        let config3 = ContextConfig::from_architecture(&arch, None);
        assert_eq!(config3.max_context_len, 1024);
        unset_env_safe("METALLIC_MAX_CONTEXT_LEN");
    }

    fn set_env_safe(k: &str, v: &str) {
        unsafe { std::env::set_var(k, v) }
    }
    fn unset_env_safe(k: &str) {
        unsafe { std::env::remove_var(k) }
    }

    #[test]
    fn test_alignment() {
        assert_eq!(ContextConfig::align_capacity(0), 128);
        assert_eq!(ContextConfig::align_capacity(1), 128);
        assert_eq!(ContextConfig::align_capacity(127), 128);
        assert_eq!(ContextConfig::align_capacity(128), 128);
        assert_eq!(ContextConfig::align_capacity(129), 256);
        assert_eq!(ContextConfig::align_capacity(200), 256);
    }

    #[test]
    fn test_memory_estimation() {
        let arch = mock_arch(2048);
        let est = ContextConfig::estimate_kv_memory(&arch, 2048);
        // layers=4, heads=8, seq=2048, head_dim=64 (512/8), f16=2 bytes, K+V=2
        // 4 * 2 * 8 * 2048 * 64 * 2 = 16,777,216 bytes (16MB)
        assert_eq!(est.kv_cache_bytes, 16777216);
    }
}
