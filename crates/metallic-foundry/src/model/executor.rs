//! CompiledModel executor for running inference.
//!
//! Interprets the ModelSpec execution plan (DSL) by calling Step::execute().

use std::{
    any::{Any, TypeId}, collections::{VecDeque, hash_map::DefaultHasher}, hash::{Hash, Hasher}, sync::OnceLock
};

use half::f16;
use metallic_loader::ModelMetadata;
use parking_lot::{Mutex, RwLock};
use rustc_hash::{FxHashMap, FxHashSet};

use super::{builder::WeightBundle, kv_geometry::KvGeometry};
mod buffers;
mod config;
mod forward;
mod kv_cache;
mod prepare;
mod rope;
mod session;
mod weights;

use crate::{
    Foundry, MetalError, compound::Layout, model::ContextConfig, policy::resolve_policy, spec::{
        ArchValue, IntExpr, ModelSpec, StorageClass, TensorBindings, WeightBindingSpec, WeightLayoutSpec, compiled::{CompiledStep, FastBindings, SymbolTable}
    }, types::{MetalResourceOptions, TensorArg}
};

#[inline(never)]
fn validate_quantized_bindings(symbols: &SymbolTable, fast: &FastBindings) -> Result<(), MetalError> {
    for (name, &idx) in symbols.iter() {
        let Some(arg) = fast.get(idx) else { continue };
        if !arg.dtype.is_quantized() {
            continue;
        }

        // Quantization scale tensors are stored as raw bytes (`Dtype::Q8_0` or packed quant dtypes) and interpreted by
        // the active quantization policy (e.g. Q8 scales, Q4_0/Q4_1 block params). We skip validating those as weights.
        if name.ends_with("_scales") {
            continue;
        }

        // Any quantized tensor is treated as a quantized weight.
        // Resolve policy and validate layout.
        let policy = resolve_policy(arg.dtype);
        let buf_size = arg.buffer.as_ref().map(|b| b.length()).unwrap_or(0);
        policy.validate_weight_layout(&arg.dims, buf_size)?;
    }

    Ok(())
}

/// A compiled, ready-to-run model.
///
/// Created via `ModelBuilder::build()`, this struct holds everything
/// needed to run inference. The forward pass is executed by iterating
/// through the DSL steps defined in the ModelSpec.
pub struct CompiledModel {
    spec: ModelSpec,
    weights: WeightBundle,
    cache_namespace_fingerprint: std::sync::Arc<str>,
    /// Optimized execution steps (compiled from DSL)
    compiled_steps: Vec<Box<dyn CompiledStep>>,
    /// Symbol table mapping tensor names to integer indices for fast lookup
    symbol_table: SymbolTable,
    /// Interned global keys for fast `int_globals` updates without allocations.
    ///
    /// This is intentionally lazy-interning so new workflows/DSL can introduce new global keys
    /// without requiring code changes in the executor.
    interned_globals: RwLock<FxHashMap<String, &'static str>>,
    /// Reusable execution session (weights/intermediates + small persistent buffers).
    session: Mutex<Option<ModelSession>>,
    /// Prefix KV snapshots for cross-conversation reuse (system prompt and short prompt prefixes).
    kv_prefix_cache: Mutex<KvPrefixCache>,
    /// Generic per-model cache for reusable runtime instances (tokenizer, adapters, etc.).
    ///
    /// Keyed by `(TypeId, logical_key)` so callers can safely cache multiple resource types
    /// under the same logical key without collisions.
    // PERF: Keep this cache attached to the compiled model so expensive runtime helpers are
    // built once per loaded model and reused across workflows/conversations.
    instance_cache: Mutex<FxHashMap<(TypeId, String), std::sync::Arc<dyn Any + Send + Sync>>>,
}

pub(crate) struct ModelSession {
    pub(crate) bindings: TensorBindings,
    pub(crate) fast_bindings: FastBindings,
    pub(crate) current_pos: usize,
    pub(crate) context_config: ContextConfig,
}

#[derive(Debug)]
struct KvPrefixTensorSnapshot {
    name: String,
    buffer: crate::types::MetalBuffer,
    dtype: crate::tensor::Dtype,
    heads: usize,
    payload_elems: usize,
}

#[derive(Debug)]
struct KvPrefixSnapshot {
    key: Option<std::sync::Arc<str>>,
    tokens: std::sync::Arc<[u32]>,
    prefix_len: usize,
    kv_tensors: Vec<KvPrefixTensorSnapshot>,
    bytes: usize,
}

#[derive(Debug, Default)]
struct KvPrefixCache {
    entries: VecDeque<KvPrefixSnapshot>,
}

impl CompiledModel {
    /// Allocate a U32 buffer for tokens.
    pub fn symbol_id(&self, name: &str) -> Option<usize> {
        self.symbol_table.get(name)
    }

    // Keep old method for backward compatibility, delegating to new one
    #[deprecated(note = "Use prepare_bindings instead")]
    pub fn prepare_weight_bindings(&self, foundry: &mut Foundry) -> Result<TensorBindings, MetalError> {
        let mut config = crate::model::ContextConfig::from_architecture(self.architecture(), None);
        config.apply_memory_budget(&foundry.device, self.architecture());
        self.prepare_bindings_with_config(foundry, &config).map(|(b, _)| b)
    }
}

#[path = "executor.test.rs"]
mod tests;
