//! Dynamic context and buffer management for Foundry.
//!
//! Supports LLM, DiT, audio, video, MoE workloads via abstract context and eviction policies.

use metallic_env::{FoundryEnvVar, KV_MEMORY_BUDGET_MB, MAX_CONTEXT_LEN, is_set};
use sysinfo::System;

use crate::{
    model::KvGeometry, spec::{Architecture, StorageClass, TensorBindings}, tensor::dtypes::DtypeExt, types::Device
};

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
        let env_cap = MAX_CONTEXT_LEN.get().ok().flatten();

        let requested_cap = override_len.or(env_cap).filter(|v| *v > 0);
        let model_max: Option<usize> = (arch.max_seq_len() > 0).then_some(arch.max_seq_len());

        let max_len = match (model_max, requested_cap) {
            (Some(model), Some(cap)) => model.min(cap),
            (Some(model), None) => model,
            (None, Some(cap)) => cap,
            (None, None) => 2048,
        }
        .max(1);

        let full_reserve = is_set(FoundryEnvVar::FullContextReserve);
        let growth_strategy = if full_reserve {
            GrowthStrategy::FullReserve
        } else {
            GrowthStrategy::default()
        };

        let initial_capacity = match growth_strategy {
            GrowthStrategy::FullReserve => max_len,
            GrowthStrategy::GrowOnDemand { initial, .. } => initial.min(max_len),
        };

        let budget_mb = KV_MEMORY_BUDGET_MB.get().ok().flatten();

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

        // Prefer exact KV byte sizing from prepare.tensors (handles F32, I8+scales, etc.).
        // Fall back to the legacy dense-F16 estimator when prepare metadata is insufficient.
        let kv = KvGeometry::from_architecture(arch);
        let (per_token_bytes, estimate_source) = if let Some(est) = Self::estimate_kv_memory_from_prepare(arch, 1) {
            (est.kv_cache_bytes, "prepare")
        } else {
            (kv.per_token_bytes_f16(arch.n_layers()), "fallback_f16_dense")
        };
        tracing::debug!(
            layout = ?kv.layout,
            n_heads = kv.n_heads,
            n_kv_heads = kv.n_kv_heads,
            group_size = kv.group_size,
            head_dim = kv.head_dim,
            cache_heads = kv.cache_heads(),
            budget_mb = budget_bytes / 1024 / 1024,
            per_token_bytes,
            estimate_source,
            "Applying KV memory budget"
        );

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
        if let Some(estimate) = Self::estimate_kv_memory_from_prepare(arch, context_len) {
            tracing::trace!(
                source = "prepare",
                n_layers = arch.n_layers(),
                context_len,
                per_layer_bytes = estimate.per_layer_bytes,
                total_bytes = estimate.kv_cache_bytes,
                "Estimated KV memory"
            );
            return estimate;
        }

        let kv = KvGeometry::from_architecture(arch);
        // 2 (K+V) * cache_heads(layout) * context_len * head_dim * 2 (F16 bytes)
        let per_layer_bytes = 2 * kv.cache_heads() * context_len * kv.head_dim * 2;
        let total = per_layer_bytes * arch.n_layers();
        tracing::trace!(
            layout = ?kv.layout,
            n_layers = arch.n_layers(),
            context_len,
            cache_heads = kv.cache_heads(),
            head_dim = kv.head_dim,
            per_layer_bytes,
            total_bytes = total,
            "Estimated KV memory"
        );
        MemoryEstimate {
            kv_cache_bytes: total,
            per_layer_bytes,
        }
    }

    fn estimate_kv_memory_from_prepare(arch: &Architecture, context_len: usize) -> Option<MemoryEstimate> {
        if arch.prepare.tensors.is_empty() {
            return None;
        }

        fn eval_expr(expr: &crate::spec::IntExpr, bindings: &TensorBindings) -> Option<usize> {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| expr.eval(bindings))).ok()
        }

        fn resolve_repeat_count(bindings: &TensorBindings, raw: &str) -> Option<usize> {
            if let Ok(v) = raw.parse::<usize>() {
                return Some(v);
            }
            if let Some(v) = bindings.get_int_global(raw) {
                return Some(v);
            }
            bindings.get_var(raw).and_then(|v| v.parse::<usize>().ok())
        }

        let mut bindings = TensorBindings::new();
        for (name, value) in &arch.params {
            if let Some(v) = value.as_usize() {
                bindings.set_int_global(name, v);
                bindings.set_global(name, v.to_string());
            }
        }
        for (name, value) in &arch.prepare.dynamics {
            bindings.set_int_global(name, *value);
            bindings.set_global(name, value.to_string());
        }
        let context_len = context_len.max(1);
        bindings.set_int_global("max_seq_len", context_len);
        bindings.set_global("max_seq_len", context_len.to_string());

        for (name, expr) in &arch.prepare.globals {
            let value = eval_expr(expr, &bindings)?;
            bindings.set_int_global(name, value);
            bindings.set_global(name, value.to_string());
        }
        for derived in &arch.prepare.derived_globals {
            let value = eval_expr(&derived.expr, &bindings)?;
            bindings.set_int_global(&derived.name, value);
            bindings.set_global(&derived.name, value.to_string());
        }

        let mut total_bytes = 0usize;
        for tensor in &arch.prepare.tensors {
            if tensor.storage != StorageClass::KvCache {
                continue;
            }

            let repeat_count = tensor
                .repeat
                .as_ref()
                .map(|repeat| resolve_repeat_count(&bindings, &repeat.count))
                .unwrap_or(Some(1))?;

            if repeat_count == 0 {
                continue;
            }

            if let Some(repeat) = &tensor.repeat {
                bindings.push_scope();
                for i in 0..repeat_count {
                    bindings.set_var(&repeat.var, i.to_string());
                    let dims = tensor
                        .dims
                        .iter()
                        .map(|expr| eval_expr(expr, &bindings))
                        .collect::<Option<Vec<_>>>()?;
                    if dims.is_empty() || dims.contains(&0) {
                        return None;
                    }
                    let bytes = tensor.dtype.layout_size(&dims);
                    total_bytes = total_bytes.checked_add(bytes)?;
                }
                bindings.pop_scope();
            } else {
                let dims = tensor
                    .dims
                    .iter()
                    .map(|expr| eval_expr(expr, &bindings))
                    .collect::<Option<Vec<_>>>()?;
                if dims.is_empty() || dims.contains(&0) {
                    return None;
                }
                let bytes = tensor.dtype.layout_size(&dims);
                total_bytes = total_bytes.checked_add(bytes)?;
            }
        }

        if total_bytes == 0 {
            return None;
        }

        let n_layers = arch.n_layers().max(1);
        Some(MemoryEstimate {
            kv_cache_bytes: total_bytes,
            per_layer_bytes: total_bytes / n_layers,
        })
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

#[path = "context.test.rs"]
mod tests;
