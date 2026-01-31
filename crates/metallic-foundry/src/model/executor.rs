//! CompiledModel executor for running inference.
//!
//! Interprets the ModelSpec execution plan (DSL) by calling Step::execute().

use std::{cell::RefCell, sync::OnceLock};

use half::f16;
use rustc_hash::{FxHashMap, FxHashSet};

use super::builder::WeightBundle;
use crate::{
    Foundry, error::MetalError, metals::sampling::SampleTopK, policy::{WeightLayout, resolve_policy}, spec::{
        MetadataValue, ModelSpec, StorageClass, TensorBindings, WeightBindingSpec, WeightLayoutSpec, compiled::{CompiledStep, FastBindings, SymbolTable}
    }, types::{MetalResourceOptions, TensorArg}
};

#[inline(never)]
fn validate_quantized_bindings(symbols: &SymbolTable, fast: &FastBindings) -> Result<(), MetalError> {
    use crate::tensor::Dtype;

    for (name, &idx) in symbols.iter() {
        let Some(arg) = fast.get(idx) else { continue };
        if arg.dtype != Dtype::U8 {
            continue;
        }

        // Quantization scale tensors are stored as raw bytes (`Dtype::U8`) and interpreted by the
        // active quantization policy (e.g. Q8 block scales). We skip validating those as weights.
        if name.ends_with("_scales") {
            continue;
        }

        // Any U8 tensor is treated as a quantized weight.
        // Resolve policy and validate layout.
        let policy = crate::policy::resolve_policy(Dtype::U8.into());
        let buf_size = arg.buffer.as_ref().map(|b| b.length()).unwrap_or(0);
        policy.validate_weight_layout(&arg.dims, buf_size)?;
    }

    Ok(())
}

#[cfg(test)]
mod quant_binding_validation_tests {
    use smallvec::smallvec;

    use super::*;
    use crate::tensor::Dtype;

    fn u8_arg_2d(n: usize, k: usize) -> crate::types::TensorArg {
        crate::types::TensorArg {
            buffer: None,
            offset: 0,
            dtype: Dtype::U8,
            dims: smallvec![n, k],
            strides: smallvec![k, 1],
        }
    }

    fn u8_arg_1d(len: usize) -> crate::types::TensorArg {
        crate::types::TensorArg {
            buffer: None,
            offset: 0,
            dtype: Dtype::U8,
            dims: smallvec![len],
            strides: smallvec![1],
        }
    }

    #[test]
    fn quantized_weight_validation_delegates_to_policy() {
        let mut symbols = SymbolTable::new();
        let w_idx = symbols.get_or_create("w".to_string());

        let mut fast = FastBindings::new(symbols.len());
        // Q8 requires K divisible by 32. 31 is invalid.
        fast.set(w_idx, u8_arg_2d(64, 31));

        let err = validate_quantized_bindings(&symbols, &fast).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("must be divisible by 32"));
    }

    #[test]
    fn quantized_weight_scales_pass_for_consistent_bindings() {
        let mut symbols = SymbolTable::new();
        let w_idx = symbols.get_or_create("w".to_string());
        let s_idx = symbols.get_or_create("w_scales".to_string());

        let mut fast = FastBindings::new(symbols.len());
        fast.set(w_idx, u8_arg_2d(64, 32));
        fast.set(s_idx, u8_arg_1d(128));

        validate_quantized_bindings(&symbols, &fast).unwrap();
    }
}

/// A compiled, ready-to-run model.
///
/// Created via `ModelBuilder::build()`, this struct holds everything
/// needed to run inference. The forward pass is executed by iterating
/// through the DSL steps defined in the ModelSpec.
pub struct CompiledModel {
    spec: ModelSpec,
    weights: WeightBundle,
    /// Optimized execution steps (compiled from DSL)
    compiled_steps: Vec<Box<dyn CompiledStep>>,
    /// Symbol table mapping tensor names to integer indices for fast lookup
    symbol_table: SymbolTable,
    /// Interned global keys for fast `int_globals` updates without allocations.
    ///
    /// This is intentionally lazy-interning so new workflows/DSL can introduce new global keys
    /// without requiring code changes in the executor.
    interned_globals: RefCell<FxHashMap<String, &'static str>>,
    /// Reusable execution session (weights/intermediates + small persistent buffers).
    session: RefCell<Option<ModelSession>>,
}

pub(crate) struct ModelSession {
    pub(crate) bindings: TensorBindings,
    pub(crate) fast_bindings: FastBindings,
    pub(crate) current_pos: usize,
    pub(crate) context_config: crate::model::ContextConfig,
}

impl CompiledModel {
    const METALLIC_IGNORE_EOS_STOP_ENV: &'static str = "METALLIC_IGNORE_EOS_STOP";

    #[inline]
    fn ignore_eos_stop_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            let Ok(value) = std::env::var(Self::METALLIC_IGNORE_EOS_STOP_ENV) else {
                return false;
            };
            let trimmed = value.trim();
            if trimmed.is_empty() {
                return false;
            }
            let lowered = trimmed.to_ascii_lowercase();
            !matches!(lowered.as_str(), "0" | "false" | "no" | "off")
        })
    }

    fn read_prefill_usize(var: &str) -> Option<usize> {
        std::env::var(var).ok().and_then(|v| v.trim().parse::<usize>().ok())
    }

    fn prefill_config() -> (usize, usize) {
        // Defaults chosen to be "big enough to matter" but not explode memory (logits = max*V).
        const DEFAULT_MAX_PREFILL_CHUNK: usize = 32;
        const DEFAULT_PREFILL_CHUNK_SIZE: usize = 32;
        const MAX_ALLOWED: usize = 512;

        let mut max_prefill_chunk = Self::read_prefill_usize("METALLIC_MAX_PREFILL_CHUNK").unwrap_or(DEFAULT_MAX_PREFILL_CHUNK);
        let mut prefill_chunk_size = Self::read_prefill_usize("METALLIC_PREFILL_CHUNK_SIZE").unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE);

        max_prefill_chunk = max_prefill_chunk.clamp(1, MAX_ALLOWED);
        prefill_chunk_size = prefill_chunk_size.clamp(1, MAX_ALLOWED);

        // Allocation must cover the largest runtime M.
        if prefill_chunk_size > max_prefill_chunk {
            max_prefill_chunk = prefill_chunk_size;
        }

        (max_prefill_chunk, prefill_chunk_size)
    }

    #[inline]
    fn interned_key(&self, key: &str) -> &'static str {
        if let Some(v) = self.interned_globals.borrow().get(key) {
            return *v;
        }

        let leaked: &'static str = Box::leak(key.to_string().into_boxed_str());
        self.interned_globals.borrow_mut().insert(key.to_string(), leaked);
        leaked
    }

    #[inline]
    pub(crate) fn set_int_global(&self, bindings: &mut TensorBindings, key: &str, value: usize) {
        let interned = self.interned_key(key);
        bindings.set_int_global(interned, value);
        // Also update string global for interpolation support (e.g. KvPrepFused params)
        bindings.set_global(interned, value.to_string());
    }

    #[inline]
    pub(crate) fn set_global_usize(&self, bindings: &mut TensorBindings, key: &str, value: usize) {
        self.set_int_global(bindings, key, value);
        bindings.set_global(key, value.to_string());
    }

    #[inline]
    fn set_global_f32(&self, bindings: &mut TensorBindings, key: &str, value: f32) {
        bindings.set_global(key, value.to_string());
    }

    pub(crate) fn apply_derived_globals(&self, bindings: &mut TensorBindings) {
        for spec in &self.spec.architecture.prepare.derived_globals {
            let value = spec.expr.eval(bindings);
            self.set_int_global(bindings, &spec.name, value);
        }
    }

    fn required_prepare_vars(&self) -> FxHashSet<&str> {
        let arch = self.architecture();
        let mut vars: FxHashSet<&str> = FxHashSet::default();

        // prepare.globals expressions
        for expr in arch.prepare.globals.values() {
            for v in expr.vars() {
                vars.insert(v.as_ref());
            }
        }
        // derived globals expressions
        for g in &arch.prepare.derived_globals {
            for v in g.expr.vars() {
                vars.insert(v.as_ref());
            }
        }
        // tensor dims/strides expressions
        for t in &arch.prepare.tensors {
            for e in &t.dims {
                for v in e.vars() {
                    vars.insert(v.as_ref());
                }
            }
            if let Some(strides) = &t.strides {
                for e in strides {
                    for v in e.vars() {
                        vars.insert(v.as_ref());
                    }
                }
            }
            if let Some(rep) = &t.repeat
                && rep.count.parse::<usize>().is_err()
            {
                vars.insert(rep.count.as_str());
            }
        }
        // weight binding expressions
        for w in &arch.weight_bindings {
            if let Some(rep) = &w.repeat
                && rep.count.parse::<usize>().is_err()
            {
                vars.insert(rep.count.as_str());
            }
            if let Some(z) = &w.fallback_zero_len {
                for v in z.vars() {
                    vars.insert(v.as_ref());
                }
            }
            if let WeightLayoutSpec::Canonical { expected_k, expected_n } = &w.layout {
                for v in expected_k.vars() {
                    vars.insert(v.as_ref());
                }
                for v in expected_n.vars() {
                    vars.insert(v.as_ref());
                }
            }
        }

        vars
    }

    fn storage_mode_for(storage: StorageClass) -> MetalResourceOptions {
        match storage {
            StorageClass::Intermediate => {
                if std::env::var("METALLIC_INTERMEDIATES_SHARED").is_ok() {
                    MetalResourceOptions::StorageModeShared
                } else {
                    MetalResourceOptions::StorageModePrivate
                }
            }
            StorageClass::KvCache => {
                if std::env::var("METALLIC_KV_CACHE_SHARED").is_ok() {
                    MetalResourceOptions::StorageModeShared
                } else {
                    MetalResourceOptions::StorageModePrivate
                }
            }
            StorageClass::RopeCache => {
                if std::env::var("METALLIC_ROPE_SHARED").is_ok() {
                    MetalResourceOptions::StorageModeShared
                } else {
                    MetalResourceOptions::StorageModePrivate
                }
            }
            StorageClass::Shared => MetalResourceOptions::StorageModeShared,
            StorageClass::Private => MetalResourceOptions::StorageModePrivate,
        }
    }

    fn allocate_tensor_from_spec(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        name: &str,
        dtype: crate::tensor::Dtype,
        dims: Vec<usize>,
        strides: Vec<usize>,
        storage: StorageClass,
        zero_fill: bool,
    ) -> Result<(), MetalError> {
        if dims.is_empty() {
            return Err(MetalError::InvalidShape(format!("prepare.tensors '{name}' requires dims.len()>0")));
        }
        if dims.iter().any(|&d| d == 0) {
            return Err(MetalError::InvalidShape(format!(
                "prepare.tensors '{name}' requires all dims>0, got {dims:?}"
            )));
        }
        if strides.len() != dims.len() {
            return Err(MetalError::InvalidShape(format!(
                "prepare.tensors '{name}' strides.len() must equal dims.len() ({} != {})",
                strides.len(),
                dims.len()
            )));
        }
        if bindings.contains(name) {
            return Err(MetalError::InvalidOperation(format!(
                "prepare.tensors attempted to re-bind existing tensor '{name}'"
            )));
        }

        let elements = dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| MetalError::InvalidShape(format!("prepare.tensors '{name}' element count overflow")))?;
        let byte_size = elements
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| MetalError::InvalidShape(format!("prepare.tensors '{name}' byte size overflow")))?;

        let storage_mode = Self::storage_mode_for(storage);
        let buffer = foundry
            .device
            .new_buffer(byte_size, storage_mode)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        if zero_fill {
            if storage_mode != MetalResourceOptions::StorageModeShared {
                return Err(MetalError::InvalidOperation(format!(
                    "prepare.tensors '{name}' requested zero_fill=true but storage is not Shared (set storage=shared)"
                )));
            }
            buffer.fill_bytes(0, byte_size);
        }

        let tensor_arg = TensorArg::from_buffer(buffer, dtype, dims, strides);
        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        Ok(())
    }

    fn seed_prepare_globals(&self, bindings: &mut TensorBindings) {
        let arch = self.architecture();
        let required = self.required_prepare_vars();

        // Seed all architecture parameters.
        for (name, val) in &arch.params {
            // Runtime overrides (e.g. memory-budget-clamped `max_seq_len`) must win over GGUF baseline.
            // `prepare_bindings_with_config` may seed some globals (like max_seq_len/max_prefill_chunk)
            // before calling this function.
            if bindings.get_int_global(name.as_str()).is_some() {
                continue;
            }
            match val {
                MetadataValue::USize(v) => self.set_global_usize(bindings, name, *v),
                MetadataValue::F32(v) => self.set_global_f32(bindings, name, *v),
            }
        }

        // Derived globals often depend on d_model/n_heads (e.g. head_dim).
        // Ensure these are available as int globals for IntExpr evaluation.
        if let Some(v) = arch.params.get("d_model").and_then(|v| v.as_usize()) {
            self.set_int_global(bindings, "d_model", v);
        }
        if let Some(v) = arch.params.get("n_heads").and_then(|v| v.as_usize()) {
            self.set_int_global(bindings, "n_heads", v);
        }
        if let Some(v) = arch.params.get("n_kv_heads").and_then(|v| v.as_usize()) {
            self.set_int_global(bindings, "n_kv_heads", v);
        }

        // Default runtime globals; workflows can override these at runtime.
        if required.contains("m") {
            self.set_int_global(bindings, "m", 1);
        }
        if required.contains("seq_len") {
            self.set_int_global(bindings, "seq_len", 1);
        }
        if required.contains("position_offset") {
            self.set_int_global(bindings, "position_offset", 0);
        }

        // Apply one-time prepare.globals.
        for (key, expr) in &arch.prepare.globals {
            let value = expr.eval(bindings);
            self.set_global_usize(bindings, key, value);
        }

        // Derived globals may be used by tensor dims (e.g. head_dim/kv_dim).
        self.apply_derived_globals(bindings);
    }

    fn allocate_prepare_tensors(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
    ) -> Result<(), MetalError> {
        let arch = self.architecture();

        // Allocate prepare.tensors.
        for tensor in &arch.prepare.tensors {
            if tensor.name.contains('{') && tensor.repeat.is_none() {
                return Err(MetalError::InvalidShape(format!(
                    "prepare.tensors '{name}' contains '{{}}' but repeat is not set",
                    name = tensor.name
                )));
            }

            if let Some(repeat) = &tensor.repeat {
                let count_val = if let Ok(v) = repeat.count.parse::<usize>() {
                    v
                } else {
                    bindings
                        .get_var(&repeat.count)
                        .ok_or_else(|| {
                            MetalError::InvalidOperation(format!("prepare.tensors repeat count variable '{}' not found", repeat.count))
                        })?
                        .parse::<usize>()
                        .map_err(|e| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' is not a valid integer: {}",
                                repeat.count, e
                            ))
                        })?
                };

                bindings.push_scope();
                for i in 0..count_val {
                    bindings.set_var(&repeat.var, i.to_string());
                    let resolved_name = bindings.interpolate(tensor.name.clone());

                    // Compute dims/strides under this scope.
                    let dims: Vec<usize> = tensor.dims.iter().map(|e| e.eval(bindings)).collect();
                    let strides: Vec<usize> = if let Some(strides) = &tensor.strides {
                        strides.iter().map(|e| e.eval(bindings)).collect()
                    } else {
                        crate::tensor::compute_strides(&dims)
                    };

                    self.allocate_tensor_from_spec(
                        foundry,
                        bindings,
                        fast_bindings,
                        &resolved_name,
                        tensor.dtype,
                        dims,
                        strides,
                        tensor.storage,
                        tensor.zero_fill,
                    )?;
                }
                bindings.pop_scope();
            } else {
                let resolved_name = bindings.interpolate(tensor.name.clone());
                let dims: Vec<usize> = tensor.dims.iter().map(|e| e.eval(bindings)).collect();
                let strides: Vec<usize> = if let Some(strides) = &tensor.strides {
                    strides.iter().map(|e| e.eval(bindings)).collect()
                } else {
                    crate::tensor::compute_strides(&dims)
                };
                self.allocate_tensor_from_spec(
                    foundry,
                    bindings,
                    fast_bindings,
                    &resolved_name,
                    tensor.dtype,
                    dims,
                    strides,
                    tensor.storage,
                    tensor.zero_fill,
                )?;
            }
        }

        Ok(())
    }

    fn bind_weights_from_spec(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        available: &FxHashMap<String, ()>,
    ) -> Result<(), MetalError> {
        let arch = &self.spec.architecture;
        if arch.weight_bindings.is_empty() {
            return Ok(());
        }

        let mut zero_cache: FxHashMap<usize, TensorArg> = FxHashMap::default();

        for spec in &arch.weight_bindings {
            if let Some(repeat) = &spec.repeat {
                let count_val = if let Ok(v) = repeat.count.parse::<usize>() {
                    v
                } else {
                    bindings
                        .get_var(&repeat.count)
                        .ok_or_else(|| {
                            MetalError::InvalidOperation(format!("weight_bindings repeat count variable '{}' not found", repeat.count))
                        })?
                        .parse::<usize>()
                        .map_err(|e| {
                            MetalError::InvalidOperation(format!(
                                "weight_bindings repeat count variable '{}' is not a valid integer: {e}",
                                repeat.count
                            ))
                        })?
                };

                bindings.push_scope();
                for i in 0..count_val {
                    bindings.set_var(&repeat.var, i.to_string());
                    let layer_idx = if spec.key.starts_with("layer.") { Some(i) } else { None };
                    self.bind_one_weight_binding(foundry, bindings, fast_bindings, available, &mut zero_cache, spec, layer_idx)?;
                }
                bindings.pop_scope();
            } else {
                if spec.key.starts_with("layer.") {
                    return Err(MetalError::InvalidShape(format!(
                        "weight_bindings key '{}' requires repeat to bind per-layer tensors",
                        spec.key
                    )));
                }
                self.bind_one_weight_binding(foundry, bindings, fast_bindings, available, &mut zero_cache, spec, None)?;
            }
        }

        Ok(())
    }

    fn bind_one_weight_binding(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        available: &FxHashMap<String, ()>,
        zero_cache: &mut FxHashMap<usize, TensorArg>,
        spec: &WeightBindingSpec,
        layer_idx: Option<usize>,
    ) -> Result<(), MetalError> {
        let tensor_names = &self.spec.architecture.tensor_names;

        let logical_name = bindings.interpolate(spec.logical_name.clone());
        if bindings.contains(&logical_name) {
            return Err(MetalError::InvalidOperation(format!(
                "weight_bindings attempted to re-bind existing tensor '{logical_name}'"
            )));
        }

        if let Some(gguf_name) = tensor_names.resolve(&spec.key, layer_idx, available) {
            match &spec.layout {
                WeightLayoutSpec::RowMajor => {
                    self.bind_gguf_tensor(bindings, fast_bindings, foundry, &gguf_name, &logical_name)?;
                }
                WeightLayoutSpec::Canonical { expected_k, expected_n } => {
                    let k = expected_k.eval(bindings);
                    let n = expected_n.eval(bindings);
                    self.bind_gguf_tensor_canonical(bindings, fast_bindings, foundry, &gguf_name, &logical_name, k, n)?;
                }
            }
            return Ok(());
        }

        if let Some(zero_len_expr) = &spec.fallback_zero_len {
            let size = zero_len_expr.eval(bindings);
            if size == 0 {
                return Err(MetalError::InvalidShape(format!(
                    "weight_bindings '{logical_name}' fallback_zero_len evaluated to 0"
                )));
            }
            let zero = if let Some(tensor) = zero_cache.get(&size) {
                tensor.clone()
            } else {
                let tensor = self.zero_tensor_arg(foundry, size)?;
                zero_cache.insert(size, tensor.clone());
                tensor
            };
            self.insert_binding(bindings, fast_bindings, logical_name, zero);
            return Ok(());
        }

        Err(MetalError::InputNotFound(format!(
            "GGUF tensor not found for weight_bindings key='{}' layer_idx={layer_idx:?} logical_name='{}'",
            spec.key, logical_name
        )))
    }

    /// Optimized execution steps (compiled from DSL)
    pub fn compiled_steps(&self) -> &[Box<dyn CompiledStep>] {
        &self.compiled_steps
    }

    /// Symbol table mapping tensor names to integer indices for fast lookup
    pub fn symbol_table(&self) -> &SymbolTable {
        &self.symbol_table
    }

    /// Create a new CompiledModel from spec and weights.
    pub(crate) fn new(spec: ModelSpec, weights: WeightBundle) -> Result<Self, MetalError> {
        if let Some(gguf_arch) = weights.architecture() {
            tracing::debug!("Loading model: spec='{}' gguf_arch='{}'", spec.name, gguf_arch);
        }

        tracing::info!("Compiling model with {} forward DSL steps...", spec.architecture.forward.len());

        // Compiler setup
        let mut symbols = SymbolTable::new();
        let mut resolver = TensorBindings::new();
        let arch = &spec.architecture;

        // Set config globals for DSL variable interpolation (needed for Repeat unrolling, etc.)
        for (name, val) in &arch.params {
            match val {
                MetadataValue::USize(v) => resolver.set_global(name.clone(), v.to_string()),
                MetadataValue::F32(v) => resolver.set_global(name.clone(), v.to_string()),
            }
        }

        // Ensure essential int globals are available for immediate IntExpr evaluation during compilation.
        // We also seed "dynamic" globals with default values from the spec so that expressions
        // depending on them (like kv_seq_len = position_offset + seq_len) can be evaluated.
        for (name, val) in &arch.prepare.dynamics {
            resolver.set_int_global(name.as_str(), *val);
        }

        for (name, val) in &arch.params {
            if let Some(v) = val.as_usize() {
                resolver.set_int_global(name.as_str(), v);
            }
        }

        // Seed common derived globals if provided in the DSL's prepare.globals.
        // If not provided, we rely on the DSL declaring them in derived_globals.
        for (key, expr) in &arch.prepare.globals {
            let value = expr.eval(&resolver);
            resolver.set_int_global(key.as_str(), value);
            resolver.set_global(key.clone(), value.to_string());
        }

        tracing::info!(
            "Architecture params: d_model={}, n_heads={}, n_kv_heads={}, n_layers={}",
            resolver.get_int_global("d_model").unwrap_or(0),
            resolver.get_int_global("n_heads").unwrap_or(0),
            resolver.get_int_global("n_kv_heads").unwrap_or(0),
            resolver.get_int_global("n_layers").unwrap_or(0)
        );

        // Apply DSL-defined derived globals so they are available for step compilation.
        for spec in &arch.prepare.derived_globals {
            let value = spec.expr.eval(&resolver);
            resolver.set_int_global(spec.name.as_str(), value);
            resolver.set_global(spec.name.clone(), value.to_string());
        }

        // Compile steps
        let mut compiled_steps = Vec::new();
        for step in &spec.architecture.forward {
            compiled_steps.extend(step.compile(&mut resolver, &mut symbols));
        }

        tracing::info!(
            "CompiledModel ready: {} compiled steps, {} symbols",
            compiled_steps.len(),
            symbols.len()
        );

        let interned_globals = {
            // Pre-intern DSL-declared globals and derived globals to avoid one-time allocations
            // at runtime, but allow any missing keys to be lazily interned.
            let mut map: FxHashMap<String, &'static str> = FxHashMap::default();

            for k in spec.architecture.prepare.globals.keys() {
                let leaked: &'static str = Box::leak(k.clone().into_boxed_str());
                map.insert(k.clone(), leaked);
            }
            for g in &spec.architecture.prepare.derived_globals {
                let leaked: &'static str = Box::leak(g.name.clone().into_boxed_str());
                map.insert(g.name.clone(), leaked);
            }

            // Pre-intern baseline architecture keys commonly referenced by DSL.
            for k in [
                "n_layers",
                "d_model",
                "n_heads",
                "n_kv_heads",
                "ff_dim",
                "vocab_size",
                "max_seq_len",
                "rope_base",
                "rms_eps",
                "head_dim",
                "kv_dim",
            ] {
                let leaked: &'static str = Box::leak(k.to_string().into_boxed_str());
                map.insert(k.to_string(), leaked);
            }

            RefCell::new(map)
        };

        Ok(Self {
            spec,
            weights,
            compiled_steps,
            symbol_table: symbols,
            interned_globals,
            session: RefCell::new(None),
        })
    }

    /// Get the model name from the spec.
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    /// Get the architecture configuration.
    pub fn architecture(&self) -> &crate::spec::Architecture {
        &self.spec.architecture
    }

    /// Get access to the underlying weights bundle.
    pub fn weights(&self) -> &WeightBundle {
        &self.weights
    }

    /// Get access to the underlying GGUF model for tensor materialization.
    pub fn gguf_model(&self) -> &crate::gguf::model_loader::GGUFModel {
        self.weights.gguf_model()
    }

    /// Create a tokenizer from the model's GGUF metadata, with optional override from ModelSpec.
    pub fn tokenizer(&self) -> Result<crate::BPETokenizer, MetalError> {
        let mut tokenizer = crate::BPETokenizer::from_gguf_metadata(self.weights.gguf_model().get_metadata())?;

        // Prioritize template from ModelSpec (DSL override)
        if let Some(template_override) = &self.spec.chat_template {
            tokenizer.set_chat_template(template_override.clone());
        }

        Ok(tokenizer)
    }

    pub fn initialize_session(&self, foundry: &mut Foundry) -> Result<(), MetalError> {
        let mut session = self.session.borrow_mut();
        if session.is_some() {
            return Ok(());
        }

        let arch = self.architecture();
        let mut context_config = crate::model::ContextConfig::from_architecture(arch, None);
        context_config.apply_memory_budget(&foundry.device, arch);
        let (mut bindings, mut fast_bindings) = self.prepare_bindings_with_config(foundry, &context_config)?;

        let max_seq_len = context_config.max_context_len;
        let allocated_capacity = context_config.allocated_capacity;
        self.set_global_usize(&mut bindings, "max_seq_len", allocated_capacity);

        // Allocate a single prompt/input buffer sized for the full potential context to avoid reallocating IDs.
        let input_ids_full = self.allocate_u32_buffer(foundry, "input_ids_full", max_seq_len)?;

        // Seed `input_ids` and `input_ids_full`.
        {
            let mut tensor_input = TensorArg::from_buffer(input_ids_full.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
            tensor_input.offset = 0;
            self.set_binding(&mut bindings, &mut fast_bindings, "input_ids", tensor_input.clone());

            let tensor_full = TensorArg::from_buffer(input_ids_full, crate::tensor::Dtype::U32, vec![max_seq_len], vec![1]);
            self.set_binding(&mut bindings, &mut fast_bindings, "input_ids_full", tensor_full);
        }

        // Pre-allocate decode sample-output buffers to avoid tiny allocations in the hot path.
        let default_decode_batch_size = bindings
            .get("output_weight")
            .ok()
            .map(|w| if w.dtype == crate::tensor::Dtype::F16 { 16 } else { 64 })
            .unwrap_or(64);

        let decode_batch_size = {
            const MAX: usize = 256;
            let parsed = std::env::var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok());
            parsed.unwrap_or(default_decode_batch_size).clamp(1, MAX)
        };

        // Seed any architecture-declared sample output buffers.
        for i in 0..decode_batch_size {
            let buf = self.allocate_u32_buffer(foundry, &format!("sample_out_{i}"), 1)?;
            let tensor = TensorArg::from_buffer(buf, crate::tensor::Dtype::U32, vec![1], vec![1]);
            self.set_binding(&mut bindings, &mut fast_bindings, &format!("sample_out_{i}"), tensor);
        }

        *session = Some(ModelSession {
            bindings,
            fast_bindings,
            current_pos: 0,
            context_config,
        });
        Ok(())
    }

    pub(crate) fn with_session_mut<T>(
        &self,
        foundry: &mut Foundry,
        f: impl FnOnce(&mut Foundry, &mut ModelSession) -> Result<T, MetalError>,
    ) -> Result<T, MetalError> {
        self.initialize_session(foundry)?;
        let mut session_guard = self.session.borrow_mut();
        let session = session_guard
            .as_mut()
            .ok_or_else(|| MetalError::OperationFailed("Foundry session missing after initialization".into()))?;
        f(foundry, session)
    }

    /// Prepare tensor bindings by:
    /// 1. Setting config globals (n_layers, d_model, etc.)
    /// 2. Materializing weight tensors from GGUF using logical name resolution
    /// 3. Allocating intermediate buffers for activations
    ///
    /// Prepare runtime bindings (weights + intermediates + KV caches) using a caller-provided context config.
    ///
    /// Prefer `prepare_bindings()` unless you need to explicitly cap/shape context behavior.
    pub fn prepare_bindings_with_config(
        &self,
        foundry: &mut Foundry,
        context_config: &crate::model::ContextConfig,
    ) -> Result<(TensorBindings, FastBindings), MetalError> {
        let mut bindings = TensorBindings::new();
        let mut fast_bindings = FastBindings::new(self.symbol_table.len());

        let _gguf = self.weights.gguf_model();
        let arch = &self.spec.architecture;
        let tensor_names = &arch.tensor_names;
        let allocated_capacity = context_config.allocated_capacity;

        if tracing::enabled!(tracing::Level::DEBUG) {
            let estimate = crate::model::ContextConfig::estimate_kv_memory(arch, allocated_capacity);
            crate::model::ContextConfig::log_system_memory();
            tracing::debug!(
                "Context config: max={}, allocated={}, strategy={:?}, estimated KV memory={:.2}MB",
                context_config.max_context_len,
                allocated_capacity,
                context_config.growth_strategy,
                estimate.kv_cache_bytes as f64 / 1e6
            );
        }

        // 1. Initialize runtime globals needed for DSL evaluation.
        // Physical max context capacity for allocations/strides.
        self.set_global_usize(&mut bindings, "max_seq_len", allocated_capacity);
        // Max prefill chunk (allocation cap) is runtime-tuned; the DSL may reference it for buffer dims.
        let (max_prefill_chunk, _prefill_chunk_size) = Self::prefill_config();
        self.set_global_usize(&mut bindings, "max_prefill_chunk", max_prefill_chunk);
        // Baseline + prepare.globals + prepare.derived_globals.
        self.seed_prepare_globals(&mut bindings);

        // Build a set of available GGUF tensor names for resolution
        let available: FxHashMap<String, ()> = self.weights.tensor_names().map(|name| (name.clone(), ())).collect();

        // 2. Bind weights either via DSL-declared weight_bindings (preferred) or legacy hardcoded maps.
        if arch.weight_bindings.is_empty() {
            // DEBT: legacy path; remove once all shipped DSL specs declare weight_bindings.
            let head_dim = arch.d_model() / arch.n_heads();
            let kv_dim = head_dim * arch.n_kv_heads();

            // Resolve and bind global weight tensors (legacy).
            let global_keys = ["embedding", "final_norm", "rope_cos", "rope_sin"];
            for key in global_keys {
                if let Some(gguf_name) = tensor_names.resolve(key, None, &available) {
                    self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, key)?;
                }
            }

            // Bind output_weight (LM Head) as NK row-major ([Vocab, DModel]) to enable GEMM prefill.
            if let Some(gguf_name) = tensor_names.resolve("output_weight", None, &available) {
                self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, "output_weight")?;
            }

            // Resolve and bind per-layer weight tensors (legacy).
            let regular_layer_keys = ["layer.attn_norm", "layer.ffn_norm"];
            let canonical_layer_keys = ["layer.attn_q", "layer.attn_k", "layer.attn_v"];
            let nk_layer_keys = ["layer.attn_output", "layer.ffn_gate", "layer.ffn_up", "layer.ffn_down"];
            for i in 0..arch.n_layers() {
                for key in regular_layer_keys {
                    if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                        let logical_name = format!("{key}_{i}");
                        self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, &logical_name)?;
                    }
                }
                for key in canonical_layer_keys {
                    if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                        let logical_name = format!("{key}_{i}");
                        let (expected_k, expected_n) = match key {
                            "layer.attn_q" => (arch.d_model(), arch.d_model()),
                            "layer.attn_k" => (arch.d_model(), kv_dim),
                            "layer.attn_v" => (arch.d_model(), kv_dim),
                            _ => {
                                return Err(MetalError::InvalidShape(format!(
                                    "Unhandled canonical weight key '{}'; update executor canonical binding map",
                                    key
                                )));
                            }
                        };
                        self.bind_gguf_tensor_canonical(
                            &mut bindings,
                            &mut fast_bindings,
                            foundry,
                            &gguf_name,
                            &logical_name,
                            expected_k,
                            expected_n,
                        )?;
                    }
                }
                for key in nk_layer_keys {
                    if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                        let logical_name = format!("{key}_{i}");
                        self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, &logical_name)?;
                    }
                }
            }

            // Resolve and bind per-layer bias tensors (fallback to zero if missing) (legacy).
            let kv_dim = arch.d_model() / arch.n_heads() * arch.n_kv_heads();
            let mut zero_cache: FxHashMap<usize, TensorArg> = FxHashMap::default();
            let bias_specs = [
                ("layer.attn_q_bias", arch.d_model()),
                ("layer.attn_k_bias", kv_dim),
                ("layer.attn_v_bias", kv_dim),
                ("layer.ffn_gate_bias", arch.ff_dim()),
                ("layer.ffn_up_bias", arch.ff_dim()),
                ("layer.ffn_down_bias", arch.d_model()),
            ];

            for i in 0..arch.n_layers() {
                for (key, size) in bias_specs {
                    let logical_name = format!("{key}_{i}");
                    if let Some(gguf_name) = tensor_names.resolve(key, Some(i), &available) {
                        self.bind_gguf_tensor(&mut bindings, &mut fast_bindings, foundry, &gguf_name, &logical_name)?;
                    } else {
                        let zero = if let Some(tensor) = zero_cache.get(&size) {
                            tensor.clone()
                        } else {
                            let tensor = self.zero_tensor_arg(foundry, size)?;
                            zero_cache.insert(size, tensor.clone());
                            tensor
                        };
                        self.insert_binding(&mut bindings, &mut fast_bindings, logical_name, zero);
                    }
                }
            }
        } else {
            self.bind_weights_from_spec(foundry, &mut bindings, &mut fast_bindings, &available)?;
        }

        // 3. Allocate intermediates/KV caches as declared by the DSL prepare plan.
        self.allocate_prepare_tensors(foundry, &mut bindings, &mut fast_bindings)?;

        // 4. Compute and bind RoPE cos/sin caches (grow-on-demand, sized to allocated capacity).
        // Names come from the DSL; values are computed/uploaded by the executor.
        if let Some(rope) = arch.prepare.rope.as_ref() {
            self.compute_and_bind_rope_caches_named(
                &mut bindings,
                &mut fast_bindings,
                foundry,
                arch,
                &rope.cos,
                &rope.sin,
                allocated_capacity,
            )?;
        }

        // Fail-fast validation: ensure quantized weights/scales bindings are structurally consistent.
        // This prevents silent correctness regressions where a quantized tensor is bound but its
        // required companion tensors (e.g. Q8 scales) are missing or malformed.
        validate_quantized_bindings(&self.symbol_table, &fast_bindings)?;

        tracing::info!("Prepared {} bindings (weights + prepare.tensors + RoPE)", bindings.len());
        Ok((bindings, fast_bindings))
    }

    /// Prepare runtime bindings using the model defaults, applying memory-budget clamping.
    pub fn prepare_bindings(&self, foundry: &mut Foundry) -> Result<(TensorBindings, FastBindings), MetalError> {
        let mut config = crate::model::ContextConfig::from_architecture(self.architecture(), None);
        config.apply_memory_budget(&foundry.device, self.architecture());
        self.prepare_bindings_with_config(foundry, &config)
    }

    // NOTE: keep validation helpers outside of hot paths; this runs once at load time.

    /// Ensure RoPE cos/sin caches cover `rope_len` positions.
    ///
    /// RoPE caches are hot-read by the GPU every attention step, so we store them in
    /// `StorageModePrivate` by default (with a debug override).
    fn compute_and_bind_rope_caches_named(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        arch: &crate::spec::Architecture,
        rope_cos_name: &str,
        rope_sin_name: &str,
        rope_len: usize,
    ) -> Result<(), MetalError> {
        let head_dim = arch.d_model() / arch.n_heads();
        let dim_half = head_dim / 2;
        let rope_base = arch.rope_base();

        self.ensure_rope_capacity_named(
            bindings,
            fast_bindings,
            foundry,
            rope_base,
            head_dim,
            dim_half,
            rope_cos_name,
            rope_sin_name,
            0,
            rope_len,
        )?;
        tracing::debug!("Prepared RoPE caches: [{}, {}] (rope_base={})", rope_len, dim_half, rope_base);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ensure_rope_capacity_named(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        rope_base: f32,
        head_dim: usize,
        dim_half: usize,
        rope_cos_name: &str,
        rope_sin_name: &str,
        old_len: usize,
        new_len: usize,
    ) -> Result<(), MetalError> {
        use half::f16;

        use crate::types::MetalResourceOptions;

        if new_len == 0 || dim_half == 0 {
            return Err(MetalError::InvalidShape("RoPE cache requires new_len>0 and dim_half>0".into()));
        }
        if old_len > new_len {
            return Err(MetalError::InvalidShape(format!(
                "RoPE cache cannot shrink: {old_len} -> {new_len}"
            )));
        }

        let storage_mode = if std::env::var("METALLIC_ROPE_SHARED").is_ok() {
            MetalResourceOptions::StorageModeShared
        } else {
            MetalResourceOptions::StorageModePrivate
        };

        let total_elements = new_len
            .checked_mul(dim_half)
            .ok_or_else(|| MetalError::InvalidShape("RoPE table size overflow".into()))?;
        let total_bytes = total_elements
            .checked_mul(2)
            .ok_or_else(|| MetalError::InvalidShape("RoPE table byte size overflow".into()))?;

        let alloc_private = |name: &str| -> Result<crate::types::MetalBuffer, MetalError> {
            foundry
                .device
                .new_buffer(total_bytes, storage_mode)
                .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {name} buffer")))
        };

        // Capture old tensors (if present) before rebinding.
        let old_cos = bindings.get(rope_cos_name).ok();
        let old_sin = bindings.get(rope_sin_name).ok();

        let can_reuse_existing = old_len == 0
            && old_cos
                .as_ref()
                .is_some_and(|t| t.dtype == crate::tensor::Dtype::F16 && t.dims.as_slice() == [new_len, dim_half].as_slice())
            && old_sin
                .as_ref()
                .is_some_and(|t| t.dtype == crate::tensor::Dtype::F16 && t.dims.as_slice() == [new_len, dim_half].as_slice());

        // Allocate new buffers (or reuse existing) and bind them.
        let (new_cos_buf, new_sin_buf, dst_cos_offset, dst_sin_offset) = if can_reuse_existing {
            let cos = old_cos
                .as_ref()
                .and_then(|t| t.buffer.as_ref())
                .ok_or_else(|| MetalError::InvalidOperation("rope_cos buffer missing".into()))?;
            let sin = old_sin
                .as_ref()
                .and_then(|t| t.buffer.as_ref())
                .ok_or_else(|| MetalError::InvalidOperation("rope_sin buffer missing".into()))?;
            (
                cos.clone(),
                sin.clone(),
                old_cos.as_ref().map(|t| t.offset).unwrap_or(0),
                old_sin.as_ref().map(|t| t.offset).unwrap_or(0),
            )
        } else {
            bindings.remove(rope_cos_name);
            bindings.remove(rope_sin_name);

            let new_cos_buf = alloc_private(rope_cos_name)?;
            let new_sin_buf = alloc_private(rope_sin_name)?;

            let cos_tensor = TensorArg::from_buffer(
                new_cos_buf.clone(),
                crate::tensor::Dtype::F16,
                vec![new_len, dim_half],
                vec![dim_half, 1],
            );
            let sin_tensor = TensorArg::from_buffer(
                new_sin_buf.clone(),
                crate::tensor::Dtype::F16,
                vec![new_len, dim_half],
                vec![dim_half, 1],
            );
            self.insert_binding(bindings, fast_bindings, rope_cos_name.to_string(), cos_tensor);
            self.insert_binding(bindings, fast_bindings, rope_sin_name.to_string(), sin_tensor);
            (new_cos_buf, new_sin_buf, 0, 0)
        };

        // Batch blits into a single command buffer when not already capturing.
        let nested_capture = foundry.is_capturing();
        if !nested_capture {
            foundry.start_capture()?;
        }

        // Preserve existing history by copying the contiguous prefix when growing.
        if !can_reuse_existing && old_len > 0 {
            let copy_bytes = old_len
                .checked_mul(dim_half)
                .and_then(|v| v.checked_mul(2))
                .ok_or_else(|| MetalError::InvalidShape("RoPE copy size overflow".into()))?;

            let old_cos = old_cos.ok_or_else(|| MetalError::InvalidOperation("Missing rope_cos during growth".into()))?;
            let old_sin = old_sin.ok_or_else(|| MetalError::InvalidOperation("Missing rope_sin during growth".into()))?;
            let old_cos_buf = old_cos
                .buffer
                .as_ref()
                .ok_or_else(|| MetalError::InvalidOperation("rope_cos buffer missing during growth".into()))?;
            let old_sin_buf = old_sin
                .buffer
                .as_ref()
                .ok_or_else(|| MetalError::InvalidOperation("rope_sin buffer missing during growth".into()))?;

            foundry.blit_copy(old_cos_buf, old_cos.offset, &new_cos_buf, dst_cos_offset, copy_bytes)?;
            foundry.blit_copy(old_sin_buf, old_sin.offset, &new_sin_buf, dst_sin_offset, copy_bytes)?;
        }

        // Upload the missing suffix [old_len, new_len).
        if new_len > old_len {
            const POS_CHUNK: usize = 1024;
            let mut pos = old_len;
            while pos < new_len {
                let end = (pos + POS_CHUNK).min(new_len);
                let chunk_len = end - pos;
                let chunk_elems = chunk_len * dim_half;

                let mut cos_data: Vec<f16> = vec![f16::ZERO; chunk_elems];
                let mut sin_data: Vec<f16> = vec![f16::ZERO; chunk_elems];

                for p in pos..end {
                    let local_p = p - pos;
                    for i in 0..dim_half {
                        let idx = local_p * dim_half + i;
                        let exponent = (2 * i) as f32 / head_dim as f32;
                        let inv_freq = 1.0f32 / rope_base.powf(exponent);
                        let angle = p as f32 * inv_freq;
                        cos_data[idx] = f16::from_f32(angle.cos());
                        sin_data[idx] = f16::from_f32(angle.sin());
                    }
                }

                let chunk_bytes = chunk_elems * 2;
                let staging_cos = foundry
                    .device
                    .new_buffer(chunk_bytes, MetalResourceOptions::StorageModeShared)
                    .ok_or_else(|| MetalError::OperationFailed("Failed to allocate RoPE staging buffer".into()))?;
                let staging_sin = foundry
                    .device
                    .new_buffer(chunk_bytes, MetalResourceOptions::StorageModeShared)
                    .ok_or_else(|| MetalError::OperationFailed("Failed to allocate RoPE staging buffer".into()))?;

                staging_cos.copy_from_slice(&cos_data);
                staging_sin.copy_from_slice(&sin_data);

                let dst_offset_bytes = dst_cos_offset + pos * dim_half * 2;
                foundry.blit_copy(&staging_cos, 0, &new_cos_buf, dst_offset_bytes, chunk_bytes)?;
                let dst_offset_bytes = dst_sin_offset + pos * dim_half * 2;
                foundry.blit_copy(&staging_sin, 0, &new_sin_buf, dst_offset_bytes, chunk_bytes)?;

                pos = end;
            }
        }

        if !nested_capture {
            let cmd = foundry.end_capture()?;
            cmd.wait_until_completed();
        }

        Ok(())
    }

    /// Bind a GGUF tensor to bindings under a logical name.
    fn bind_gguf_tensor(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        gguf_name: &str,
        logical_name: &str,
    ) -> Result<(), MetalError> {
        self.bind_gguf_tensor_with_layout(bindings, fast_bindings, foundry, gguf_name, logical_name, WeightLayout::RowMajor)
    }

    /// Bind a GGUF tensor to bindings in canonical k-block-major layout.
    /// Used for 2D weight matrices with GemvCanonical kernel.
    #[allow(clippy::too_many_arguments)]
    fn bind_gguf_tensor_canonical(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        gguf_name: &str,
        logical_name: &str,
        expected_k: usize,
        expected_n: usize,
    ) -> Result<(), MetalError> {
        self.bind_gguf_tensor_with_layout(
            bindings,
            fast_bindings,
            foundry,
            gguf_name,
            logical_name,
            WeightLayout::Canonical { expected_k, expected_n },
        )
    }

    /// Helper to bind a GGUF tensor using the new QuantizationPolicy system.
    fn bind_gguf_tensor_with_layout(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        gguf_name: &str,
        logical_name: &str,
        layout: WeightLayout,
    ) -> Result<(), MetalError> {
        let gguf = self.weights.gguf_model();
        let tensor_info = gguf
            .get_tensor(gguf_name)
            .ok_or_else(|| MetalError::InputNotFound(format!("Tensor '{}' not found in GGUF", gguf_name)))?;

        let policy = resolve_policy(tensor_info.data_type());

        // Load weights using the policy (handles Q8 splitting, F32 downcast, canonical, etc.)
        let loaded_tensors = policy
            .load_weights(foundry, gguf, gguf_name, logical_name, layout)
            .map_err(|e| MetalError::OperationFailed(format!("Policy load failed for '{}': {}", gguf_name, e)))?;

        for (name, tensor_arg) in loaded_tensors {
            self.insert_binding(bindings, fast_bindings, name.clone(), tensor_arg.clone());

            if name == logical_name {
                // Also alias the GGUF name to the same tensor arg
                self.insert_binding(bindings, fast_bindings, gguf_name.to_string(), tensor_arg);
                tracing::trace!("Bound '{}' -> '{}' using policy {}", logical_name, gguf_name, policy.short_name());
            } else {
                tracing::trace!("Bound derived '{}' using policy {}", name, policy.short_name());
            }
        }

        Ok(())
    }

    fn zero_tensor_arg(&self, foundry: &mut Foundry, size: usize) -> Result<TensorArg, MetalError> {
        if size == 0 {
            return Err(MetalError::InvalidShape("zero_tensor_arg size must be > 0".into()));
        }

        let byte_size = size * std::mem::size_of::<f16>();
        let buffer = foundry
            .device
            .new_buffer(byte_size, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for zero buffer", byte_size)))?;

        buffer.fill_bytes(0, byte_size);

        Ok(TensorArg::from_buffer(buffer, crate::tensor::Dtype::F16, vec![size], vec![1]))
    }

    #[allow(dead_code)]
    /// Allocate an intermediate buffer for activations.
    fn allocate_intermediate(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        name: &str,
        size: usize,
    ) -> Result<(), MetalError> {
        if size == 0 {
            return Err(MetalError::InvalidShape(format!("allocate_intermediate '{name}' requires size>0")));
        }

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let storage_mode = if std::env::var("METALLIC_INTERMEDIATES_SHARED").is_ok() {
            MetalResourceOptions::StorageModeShared
        } else {
            MetalResourceOptions::StorageModePrivate
        };

        // Allocate F16 buffer (2 bytes per element)
        let byte_size = size * 2;
        let buffer = foundry
            .device
            .new_buffer(byte_size, storage_mode)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(buffer, crate::tensor::Dtype::F16, vec![size], vec![1]);

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated intermediate '{}' ({} elements)", name, size);

        Ok(())
    }

    #[allow(dead_code)]
    /// Allocate a 2D intermediate buffer for activations (row-major).
    fn allocate_intermediate_2d(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<(), MetalError> {
        if rows == 0 || cols == 0 {
            return Err(MetalError::InvalidShape(format!(
                "allocate_intermediate_2d '{name}' requires rows>0 and cols>0"
            )));
        }

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let storage_mode = if std::env::var("METALLIC_INTERMEDIATES_SHARED").is_ok() {
            MetalResourceOptions::StorageModeShared
        } else {
            MetalResourceOptions::StorageModePrivate
        };

        let total_elements = rows * cols;
        let byte_size = total_elements * 2; // F16
        let buffer = foundry
            .device
            .new_buffer(byte_size, storage_mode)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            buffer,
            crate::tensor::Dtype::F16,
            vec![rows, cols],
            crate::tensor::compute_strides(&[rows, cols]),
        );

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated 2D intermediate '{}' [{}, {}]", name, rows, cols);

        Ok(())
    }

    #[allow(dead_code)]
    /// Allocate a 3D intermediate buffer for SDPA-style tensors.
    #[allow(clippy::too_many_arguments)]
    fn allocate_intermediate_3d(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        name: &str,
        batch: usize,
        seq_len: usize,
        dim: usize,
    ) -> Result<(), MetalError> {
        if batch == 0 || seq_len == 0 || dim == 0 {
            return Err(MetalError::InvalidShape(format!(
                "allocate_intermediate_3d '{name}' requires batch>0, seq_len>0, dim>0"
            )));
        }

        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let storage_mode = if std::env::var("METALLIC_INTERMEDIATES_SHARED").is_ok() {
            MetalResourceOptions::StorageModeShared
        } else {
            MetalResourceOptions::StorageModePrivate
        };

        let total_elements = batch * seq_len * dim;
        let byte_size = total_elements * 2; // F16 = 2 bytes
        let buffer = foundry
            .device
            .new_buffer(byte_size, storage_mode)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            buffer,
            crate::tensor::Dtype::F16,
            vec![batch, seq_len, dim],
            crate::tensor::compute_strides(&[batch, seq_len, dim]),
        );

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated 3D intermediate '{}' [{}, {}, {}]", name, batch, seq_len, dim);

        Ok(())
    }

    /// Helper to insert a tensor into both string and fast bindings
    fn insert_binding(&self, bindings: &mut TensorBindings, fast_bindings: &mut FastBindings, name: String, tensor: TensorArg) {
        if let Some(id) = self.symbol_table.get(&name) {
            fast_bindings.set(id, tensor.clone());
        }
        bindings.insert(name, tensor);
    }

    /// Helper to update an existing binding without allocating a new String key.
    /// Falls back to insert if the binding isn't present (should be rare on hot paths).
    pub(crate) fn set_binding(&self, bindings: &mut TensorBindings, fast_bindings: &mut FastBindings, name: &str, tensor: TensorArg) {
        if let Some(id) = self.symbol_table.get(name) {
            fast_bindings.set(id, tensor.clone());
        }
        bindings.set_binding(name, tensor);
    }

    #[allow(dead_code)]
    /// Allocate a KV cache buffer for attention caching.
    /// Shape: [n_heads, max_seq_len, head_dim]
    #[allow(clippy::too_many_arguments)]
    fn allocate_kv_cache(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        name: &str,
        n_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
    ) -> Result<(), MetalError> {
        // Skip if already bound
        if bindings.contains(name) {
            return Ok(());
        }

        let total_elements = n_heads * max_seq_len * head_dim;
        let byte_size = total_elements * 2; // F16 = 2 bytes

        // KV caches are hot-read on GPU every decode step; prefer private storage.
        // Shared is only useful for debugging (CPU visibility) and is slower on GPU.
        let storage_mode = if std::env::var("METALLIC_KV_CACHE_SHARED").is_ok() {
            MetalResourceOptions::StorageModeShared
        } else {
            MetalResourceOptions::StorageModePrivate
        };

        let buffer = foundry
            .device
            .new_buffer(byte_size, storage_mode)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        let tensor_arg = TensorArg::from_buffer(
            buffer,
            crate::tensor::Dtype::F16,
            vec![n_heads, max_seq_len, head_dim],
            crate::tensor::compute_strides(&[n_heads, max_seq_len, head_dim]),
        );

        self.insert_binding(bindings, fast_bindings, name.to_string(), tensor_arg);
        tracing::trace!("Allocated KV cache '{}' [{}, {}, {}]", name, n_heads, max_seq_len, head_dim);

        Ok(())
    }

    /// Ensure that KV caches and related buffers have enough capacity for the requested length.
    /// Grows buffers on-demand if needed, respecting alignment and max context limits.
    pub(crate) fn ensure_kv_capacity(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        context_config: &mut crate::model::context::ContextConfig,
        current_pos: usize,
        required_len: usize,
    ) -> Result<(), MetalError> {
        if required_len <= context_config.allocated_capacity {
            return Ok(());
        }

        if required_len > context_config.max_context_len {
            return Err(MetalError::InvalidOperation(format!(
                "Requested context length {} exceeds maximum allowed {}",
                required_len, context_config.max_context_len
            )));
        }

        // Calculate new capacity using growth strategy
        let mut new_capacity = match context_config.growth_strategy {
            crate::model::context::GrowthStrategy::FullReserve => context_config.max_context_len,
            crate::model::context::GrowthStrategy::GrowOnDemand { growth_factor, .. } => {
                let grown = (context_config.allocated_capacity as f32 * growth_factor) as usize;
                grown.max(required_len).min(context_config.max_context_len)
            }
        };

        // Enforce alignment to 128 elements to keep kernels happy
        new_capacity = crate::model::context::ContextConfig::align_capacity(new_capacity);

        tracing::info!(
            "Growing KV cache capacity: {} -> {} (requested {})",
            context_config.allocated_capacity,
            new_capacity,
            required_len
        );

        let old_capacity = context_config.allocated_capacity;
        self.reallocate_kv_buffers(foundry, bindings, fast_bindings, current_pos, old_capacity, new_capacity)?;
        context_config.allocated_capacity = new_capacity;

        Ok(())
    }

    /// Reallocate all context-dependent buffers (KV caches, slice buffers, etc.).
    fn reallocate_kv_buffers(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        current_pos: usize,
        old_capacity: usize,
        new_capacity: usize,
    ) -> Result<(), MetalError> {
        let arch = self.architecture();
        let head_dim = arch.d_model() / arch.n_heads();

        // Batch all buffer copies into a single command buffer when possible.
        let nested_capture = foundry.is_capturing();
        if !nested_capture {
            foundry.start_capture()?;
        }

        // 1. Update physical max context capacity for kernels and expression evaluation.
        self.set_global_usize(bindings, "max_seq_len", new_capacity);
        // Derived globals may depend on max_seq_len (e.g. kv_seq_len).
        self.apply_derived_globals(bindings);

        // 2. Reallocate any grow_with_kv tensors declared by the DSL.
        // Rope caches are handled by the dedicated RoPE grow path below.
        let preserve_kv_cache = |foundry: &mut Foundry, name: &str, old: &TensorArg, new: &TensorArg| -> Result<(), MetalError> {
            if current_pos == 0 {
                return Ok(());
            }
            if old.dtype != crate::tensor::Dtype::F16 || new.dtype != crate::tensor::Dtype::F16 {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' must be F16 for growth preservation"
                )));
            }
            if old.dims.as_slice() != [arch.n_heads(), old_capacity, head_dim].as_slice() {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' old dims mismatch: got {:?}, expected [{}, {}, {}]",
                    old.dims,
                    arch.n_heads(),
                    old_capacity,
                    head_dim
                )));
            }
            if new.dims.as_slice() != [arch.n_heads(), new_capacity, head_dim].as_slice() {
                return Err(MetalError::InvalidShape(format!(
                    "KV cache '{name}' new dims mismatch: got {:?}, expected [{}, {}, {}]",
                    new.dims,
                    arch.n_heads(),
                    new_capacity,
                    head_dim
                )));
            }

            let old_buf = old
                .buffer
                .as_ref()
                .ok_or_else(|| MetalError::InvalidOperation(format!("{name} buffer missing during growth")))?;
            let new_buf = new
                .buffer
                .as_ref()
                .ok_or_else(|| MetalError::InvalidOperation(format!("{name} buffer missing after growth")))?;

            let copy_size = current_pos * head_dim * 2;
            for h in 0..arch.n_heads() {
                let old_head_offset = h * old_capacity * head_dim * 2;
                let new_head_offset = h * new_capacity * head_dim * 2;
                foundry.blit_copy(
                    old_buf,
                    old.offset + old_head_offset,
                    new_buf,
                    new.offset + new_head_offset,
                    copy_size,
                )?;
            }
            Ok(())
        };

        for tensor in &arch.prepare.tensors {
            if !tensor.grow_with_kv {
                continue;
            }
            if tensor.storage == StorageClass::RopeCache {
                continue;
            }

            let mut alloc_one = |bindings: &mut TensorBindings, name: String| -> Result<(), MetalError> {
                let old = bindings.get(&name).ok();
                bindings.remove(&name);

                let dims: Vec<usize> = tensor.dims.iter().map(|e| e.eval(bindings)).collect();
                let strides: Vec<usize> = if let Some(strides) = &tensor.strides {
                    strides.iter().map(|e| e.eval(bindings)).collect()
                } else {
                    crate::tensor::compute_strides(&dims)
                };

                self.allocate_tensor_from_spec(
                    foundry,
                    bindings,
                    fast_bindings,
                    &name,
                    tensor.dtype,
                    dims,
                    strides,
                    tensor.storage,
                    tensor.zero_fill,
                )?;

                if tensor.storage == StorageClass::KvCache {
                    let old = old.ok_or_else(|| MetalError::InvalidOperation(format!("Missing KV cache '{name}' during growth")))?;
                    let new = bindings.get(&name)?;
                    preserve_kv_cache(foundry, &name, &old, &new)?;
                }

                Ok(())
            };

            if let Some(repeat) = &tensor.repeat {
                let count_val = if let Ok(v) = repeat.count.parse::<usize>() {
                    v
                } else {
                    bindings
                        .get_var(&repeat.count)
                        .ok_or_else(|| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' not found during growth",
                                repeat.count
                            ))
                        })?
                        .parse::<usize>()
                        .map_err(|e| {
                            MetalError::InvalidOperation(format!(
                                "prepare.tensors repeat count variable '{}' is not a valid integer during growth: {}",
                                repeat.count, e
                            ))
                        })?
                };
                bindings.push_scope();
                for i in 0..count_val {
                    bindings.set_var(&repeat.var, i.to_string());
                    let resolved = bindings.interpolate(tensor.name.clone());
                    alloc_one(bindings, resolved)?;
                }
                bindings.pop_scope();
            } else {
                let resolved = bindings.interpolate(tensor.name.clone());
                alloc_one(bindings, resolved)?;
            }
        }

        // 3. Grow RoPE tables to match the new physical capacity.
        let dim_half = head_dim / 2;
        if let Some(rope) = arch.prepare.rope.as_ref() {
            self.ensure_rope_capacity_named(
                bindings,
                fast_bindings,
                foundry,
                arch.rope_base(),
                head_dim,
                dim_half,
                &rope.cos,
                &rope.sin,
                old_capacity,
                new_capacity,
            )?;
        }

        if !nested_capture {
            let cmd = foundry.end_capture()?;
            cmd.wait_until_completed();
        }

        Ok(())
    }

    /// Run a single forward step by executing all DSL steps.
    ///
    /// Each step in `spec.architecture.forward` is executed via `Step::execute()`.
    /// Run a single forward step by executing all compiled steps.
    pub fn forward(&self, foundry: &mut Foundry, bindings: &mut TensorBindings, fast_bindings: &FastBindings) -> Result<(), MetalError> {
        //eprintln!(
        //    "Forward: m={}, seq_len={}, pos={}, kv_seq_len={} | StrPos={}",
        //    bindings.get_int_global("m").unwrap_or(0),
        //    bindings.get_int_global("seq_len").unwrap_or(0),
        //    bindings.get_int_global("position_offset").unwrap_or(0),
        //    bindings.get_int_global("kv_seq_len").unwrap_or(0),
        //    bindings.get_var("position_offset").map(|s| s.as_str()).unwrap_or("MISSING")
        //);
        // If we are already capturing (e.g. batched prompt processing), don't start a new capture.
        let nested_capture = foundry.is_capturing();
        let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
        let debug_step_log = std::env::var("METALLIC_DEBUG_STEP_LOG").is_ok();
        let debug_step_sync = std::env::var("METALLIC_DEBUG_COMPILED_STEP_SYNC").is_ok();

        // Always start capture, even in profiling mode (dispatch_pipeline handles per-kernel sync)
        if !nested_capture {
            foundry.start_capture()?;
        }

        for (idx, step) in self.compiled_steps.iter().enumerate() {
            if debug_step_log {
                tracing::info!("Forward compiled step {:03}: {}", idx, step.name());
            }
            step.execute(foundry, fast_bindings, bindings, &self.symbol_table)?;
            if debug_step_sync {
                // Flush after each compiled step to isolate GPU hangs.
                // This is intentionally expensive and intended only for debugging.
                foundry.restart_capture_sync()?;
            }
        }

        // End capture only if we're not profiling per-kernel (profiling mode syncs per dispatch)
        if !nested_capture && !profiling_per_kernel {
            foundry.end_capture()?;
        }

        Ok(())
    }

    /// Run a single forward step by executing DSL steps (uncompiled/interpreted).
    ///
    /// Unlike `forward()` which uses pre-compiled steps, this method executes the
    /// original `Step::execute()` method on each step in `spec.architecture.forward`.
    /// This allows runtime modification of variables like `n_layers` via `bindings.set_global()`.
    pub fn forward_uncompiled(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        // Start a new command buffer for this forward pass (token)
        foundry.start_capture()?;

        for step in &self.spec.architecture.forward {
            if step.name() == "Sample" {
                continue;
            }
            step.execute(foundry, bindings)?;
        }

        // Commit and wait for the token to complete
        let cmd_buffer = foundry.end_capture()?;
        cmd_buffer.wait_until_completed();

        Ok(())
    }

    /// Generate multiple tokens autoregressively.
    ///
    /// # Arguments
    /// * `foundry` - The foundry instance for kernel execution
    /// * `prompt_tokens` - Initial token IDs for the prompt
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `stop_tokens` - Token IDs that signal end of generation
    /// * `temperature` - Sampling temperature (higher = more random)
    /// * `top_k` - Top-K sampling parameter
    /// * `top_p` - Top-P (nucleus) sampling parameter
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        foundry: &mut Foundry,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<Vec<u32>, MetalError> {
        self.generate_with_seed(
            foundry,
            prompt_tokens,
            max_new_tokens,
            stop_tokens,
            temperature,
            top_k,
            top_p,
            42u32,
        )
    }

    /// Generate multiple tokens autoregressively with an explicit base seed for sampling.
    ///
    /// The seed is advanced once per generated token (`seed + step`) to avoid pathological repetition
    /// while keeping the output deterministic for parity testing.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_seed_streaming<F>(
        &self,
        foundry: &mut Foundry,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
        mut callback: F,
    ) -> Result<Vec<u32>, MetalError>
    where
        F: FnMut(u32, std::time::Duration, std::time::Duration, Option<std::time::Duration>) -> Result<bool, MetalError>,
    {
        //eprintln!(
        //    "Starting generate_with_seed_streaming: prompt_len={}, max_new={}",
        //    prompt_tokens.len(),
        //    max_new_tokens
        //);
        if prompt_tokens.is_empty() {
            return Err(MetalError::InvalidShape("generate requires non-empty prompt_tokens".into()));
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let arch = &self.spec.architecture;

        let setup_start = std::time::Instant::now();

        // Reuse the prepared session (weights + intermediates + persistent buffers) instead of
        // materializing everything per generation.
        self.initialize_session(foundry)?;
        let mut session_guard = self.session.borrow_mut();
        let session = session_guard
            .as_mut()
            .ok_or_else(|| MetalError::OperationFailed("Foundry session missing after initialization".into()))?;
        let start_pos = session.current_pos;
        tracing::info!(
            "Foundry generate start: pos={}, prompt_tokens len={}, tokens={:?}",
            start_pos,
            prompt_tokens.len(),
            prompt_tokens
        );
        let prompt_len = prompt_tokens.len();
        // Ensure we have enough KV capacity for this prompt + existing history
        self.ensure_kv_capacity(
            foundry,
            &mut session.bindings,
            &mut session.fast_bindings,
            &mut session.context_config,
            start_pos,
            prompt_len + start_pos,
        )?;

        let bindings = &mut session.bindings;
        let fast_bindings = &mut session.fast_bindings;

        if prompt_len + start_pos > session.context_config.max_context_len {
            return Err(MetalError::InvalidShape(format!(
                "Prompt length {prompt_len} + start_pos {start_pos} exceeds max_context_len {}",
                session.context_config.max_context_len
            )));
        }

        // Write all prompt tokens to the shared buffer
        // input_ids_full is MetalBuffer.
        let input_ids_full_arg = bindings.get("input_ids_full")?;
        let input_ids_full = input_ids_full_arg.buffer.as_ref().unwrap();
        input_ids_full.copy_from_slice_offset(prompt_tokens, start_pos);

        // Defaults for decode (m=1, seq_len=1). Prefill overrides these per chunk.
        self.set_int_global(bindings, "m", 1);
        self.set_int_global(bindings, "seq_len", 1);
        self.set_int_global(bindings, "position_offset", start_pos);
        self.apply_derived_globals(bindings);

        let greedy = temperature <= 0.0 || !temperature.is_finite() || top_k == 0;
        let vocab_size = arch.vocab_size() as u32;

        // Prefill KV cache.
        //
        // Fast path is batched prefill (M>1) in large chunks. However, we've observed correctness
        // issues when the *final* chunk is very small (e.g. 32 + 2 tokens). To keep performance
        // while avoiding that tail pathology, we rebalance the chunk size so the prompt splits
        // into similarly-sized chunks (e.g. 34 @ 32 => 17 + 17) instead of a tiny tail.
        //
        // Set `METALLIC_DISABLE_BATCHED_PREFILL=1` to force fully sequential prefill for isolation.
        let profiling_per_kernel = crate::instrument::foundry_per_kernel_profiling_enabled();
        let disable_batched_prefill_env = std::env::var("METALLIC_DISABLE_BATCHED_PREFILL").is_ok();
        let disable_batched_prefill = profiling_per_kernel || disable_batched_prefill_env;
        // We use "prefill_chunk_size" as the vector width (m) for batched prefill and as a capture
        // batching knob in sequential mode.
        let max_prefill_chunk = bindings.get_int_global("max_prefill_chunk").unwrap_or(32).max(1);
        let (_, mut prefill_chunk_size) = Self::prefill_config();
        prefill_chunk_size = prefill_chunk_size.min(max_prefill_chunk).max(1);

        let input_ids_key = "input_ids";

        // Check for debug sync flag to disable batched capture for isolation
        let debug_sync = std::env::var("METALLIC_DEBUG_FORWARD_SYNC").is_ok();

        let prefill_start = std::time::Instant::now();
        let setup_duration = prefill_start.duration_since(setup_start);
        let mut last_prefill_m = 1usize;

        if !disable_batched_prefill {
            if start_pos > 0 {
                tracing::debug!(
                    "Using BATCHED prefill for multi-turn (m={}, start_pos={}). This logic is experimental.",
                    prompt_tokens.len(),
                    start_pos
                );
            } else {
                tracing::debug!("Using BATCHED prefill (m={})", prompt_tokens.len());
            }
            // === BATCHED PREFILL (m>1) ===
            // In debug-sync mode, isolate each chunk into its own synchronous command buffer.
            if !debug_sync && !profiling_per_kernel {
                foundry.start_capture()?;
            }

            let rebalance_chunk_size = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
                let requested = requested.max(1).min(max_allowed.max(1));
                if prompt_len <= 1 {
                    return 1;
                }

                let chunks = prompt_len.div_ceil(requested);
                let balanced = prompt_len.div_ceil(chunks);
                balanced.max(1).min(max_allowed)
            };

            let chunk_size = rebalance_chunk_size(prompt_len, prefill_chunk_size, max_prefill_chunk);

            for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(chunk_size).enumerate() {
                let m = chunk_tokens.len();
                last_prefill_m = m;
                let base_pos = start_pos + chunk_idx * chunk_size;

                if debug_sync && !profiling_per_kernel {
                    foundry.start_capture()?;
                }

                self.set_int_global(bindings, "m", m);
                self.set_int_global(bindings, "seq_len", m);
                self.set_int_global(bindings, "position_offset", base_pos);
                self.apply_derived_globals(bindings);

                //eprintln!(
                //    "Prefill chunk: m={}, seq_len={}, pos={}, kv_seq_len={}",
                //    m,
                //    m,
                //    base_pos,
                //    bindings.get_int_global("kv_seq_len").unwrap_or(0)
                //);

                let input_ids_full = bindings.get("input_ids_full")?.buffer.as_ref().unwrap().clone();
                let mut tensor_input = TensorArg::from_buffer(input_ids_full, crate::tensor::Dtype::U32, vec![m], vec![1]);
                tensor_input.offset = base_pos * 4;
                self.set_binding(bindings, fast_bindings, input_ids_key, tensor_input);

                self.forward(foundry, bindings, &*fast_bindings)?;

                if debug_sync && !profiling_per_kernel {
                    let cmd = foundry.end_capture()?;
                    cmd.wait_until_completed();
                }
            }

            if !debug_sync && !profiling_per_kernel {
                let cmd = foundry.end_capture()?;
                cmd.wait_until_completed();
            }
        } else {
            tracing::info!("Using SEQUENTIAL prefill (m=1)");
            // === SEQUENTIAL PREFILL (seq_len=1) ===
            // Capture the whole prefill to amortize submission overhead, but execute each token
            // with the same semantics as the decode loop.
            self.set_int_global(bindings, "m", 1);
            self.set_int_global(bindings, "seq_len", 1);

            for (chunk_idx, chunk_tokens) in prompt_tokens.chunks(prefill_chunk_size).enumerate() {
                let base_pos = start_pos + chunk_idx * prefill_chunk_size;

                if !profiling_per_kernel && (debug_sync || chunk_idx == 0) {
                    foundry.start_capture()?;
                }

                for (i, _token_id) in chunk_tokens.iter().enumerate() {
                    let pos = base_pos + i;
                    self.set_int_global(bindings, "position_offset", pos);
                    self.apply_derived_globals(bindings);

                    let input_ids_full = bindings.get("input_ids_full")?.buffer.as_ref().unwrap().clone();
                    let mut tensor_input = TensorArg::from_buffer(input_ids_full, crate::tensor::Dtype::U32, vec![1], vec![1]);
                    tensor_input.offset = pos * 4;
                    self.set_binding(bindings, fast_bindings, input_ids_key, tensor_input);

                    self.forward(foundry, bindings, &*fast_bindings)?;
                }

                if debug_sync && !profiling_per_kernel {
                    let cmd = foundry.end_capture()?;
                    cmd.wait_until_completed();
                }
            }

            if !debug_sync && !profiling_per_kernel {
                let cmd = foundry.end_capture()?;
                cmd.wait_until_completed();
            }
        }

        let prefill_duration = prefill_start.elapsed();

        // Reset to decode mode for autoregressive decode (M=1).
        self.set_int_global(bindings, "m", 1);
        self.set_int_global(bindings, "seq_len", 1);
        self.set_int_global(bindings, "position_offset", start_pos + prompt_len);
        self.apply_derived_globals(bindings);

        // Now autoregressive decode: sample from last prompt-token logits, then step forward per token.
        // Reuse the session input buffer as a valid fallback input_ids buffer (we overwrite binding to sampled-token
        // buffers during decode).
        let single_input_buffer = bindings.get("input_ids_full")?.buffer.as_ref().unwrap().clone();

        let default_decode_batch_size = bindings
            .get("output_weight")
            .ok()
            .map(|w| if w.dtype == crate::tensor::Dtype::F16 { 16 } else { 64 })
            .unwrap_or(64);

        let decode_batch_size = || -> usize {
            const MAX: usize = 256;
            let parsed = std::env::var("METALLIC_FOUNDRY_DECODE_BATCH_SIZE")
                .ok()
                .and_then(|v| v.trim().parse::<usize>().ok());
            parsed.unwrap_or(default_decode_batch_size).clamp(1, MAX)
        };

        // Ensure input_ids is bound to the single input buffer (offset 0) for the generated tokens
        let input_ids_key = "input_ids";
        // Seed `input_ids` with any valid U32 buffer; we overwrite it per-step below to point at the
        // sampled-token buffer (avoids a dedicated CopyU32 kernel each step).
        {
            let mut tensor_input = TensorArg::from_buffer(single_input_buffer.clone(), crate::tensor::Dtype::U32, vec![1], vec![1]);
            tensor_input.offset = 0;
            self.set_binding(bindings, fast_bindings, input_ids_key, tensor_input);
        }

        // BATCHING: We batch multiple tokens into a single command buffer to amortize submission overhead.
        // We keep tokens on GPU (copying Sample -> Input) and only sync when the batch is full.
        let batch_size = if profiling_per_kernel { 1 } else { decode_batch_size() };
        let sample_out_count = bindings.iter().filter(|(k, _)| k.starts_with("sample_out_")).count();
        if batch_size > sample_out_count {
            return Err(MetalError::InvalidShape(format!(
                "Decode batch_size {batch_size} exceeds session capacity {sample_out_count}. This typically means METALLIC_FOUNDRY_DECODE_BATCH_SIZE changed after model load."
            )));
        }

        let mut pending_count = 0;
        let ignore_eos_stop = Self::ignore_eos_stop_enabled();
        let emit_host_metrics = crate::instrument::foundry_metrics_enabled();
        let mut batch_encode_start: Option<std::time::Instant> = None;
        let mut iteration_duration: Option<std::time::Duration> = None;

        for step in 0..max_new_tokens {
            let batch_idx = pending_count;

            // Ensure we have enough KV capacity for the entire upcoming batch.
            // This avoids having to grow mid-capture (which would break batching and add extra syncs).
            if batch_idx == 0 {
                let current_pos = start_pos + prompt_len + step;
                let remaining = max_new_tokens - step;
                let lookahead = remaining.min(batch_size);
                let required_len = current_pos + lookahead;
                self.ensure_kv_capacity(
                    foundry,
                    bindings,
                    fast_bindings,
                    &mut session.context_config,
                    current_pos,
                    required_len,
                )?;
            }

            // Start capture at the beginning of a batch (only if not already capturing)
            // In profiling mode, dispatch_pipeline() restarts capture after each kernel sync
            if batch_idx == 0 && !foundry.is_capturing() {
                foundry.start_capture()?;
            }
            // Track timing for different modes
            let step_start = std::time::Instant::now();
            if emit_host_metrics && !profiling_per_kernel && batch_idx == 0 {
                batch_encode_start = Some(step_start);
            }

            // 1. Get logits from previous forward pass (or prefill)
            let logits = bindings.get("logits")?;

            let mut logits_arg = logits.clone();

            // If sampling from prefill result (step 0), offset to the last token if batch > 1
            if step == 0 && last_prefill_m > 1 {
                logits_arg.offset += (last_prefill_m - 1) * (vocab_size as usize) * 2; // F16 bytes
            }

            // 2. Sample (Greedy or Random) using GPU kernel
            // We use SampleTopK for both to keep execution on GPU. Greedy is just top_k=1.
            let effective_top_k = if greedy { 1 } else { top_k };
            let sample_out_name = format!("sample_out_{batch_idx}");
            let sample_out = bindings.get(&sample_out_name)?;

            // Create sample kernel with destination buffer
            let sample_kernel = SampleTopK::new(
                &logits_arg,
                &sample_out,
                vocab_size,
                effective_top_k,
                top_p,
                temperature,
                seed.wrapping_add(step as u32),
            );
            foundry.run(&sample_kernel)?;

            // 3. Feed the sampled token directly into the next forward pass by rebinding `input_ids`
            // to the sampled-token buffer. This avoids an extra CopyU32 dispatch per step.
            self.set_binding(bindings, fast_bindings, input_ids_key, sample_out.clone());

            // 4. Update globals for the NEXT forward pass
            // We are about to run forward for position `start_pos + prompt_len + step` (the token we just sampled).
            self.set_int_global(bindings, "position_offset", start_pos + prompt_len + step);
            self.apply_derived_globals(bindings);

            //if step < 5 {
            // Avoid spamming loops
            //eprintln!(
            //   "Decode step {}: pos={}, kv_seq_len={}",
            //   step,
            //   bindings.get_int_global("position_offset").unwrap_or(0),
            //   bindings.get_int_global("kv_seq_len").unwrap_or(0)
            //;
            //}

            // 5. Run Forward (predicts next token's logits)
            self.forward(foundry, bindings, &*fast_bindings)?;

            pending_count += 1;

            // 6. If batch is full or we are done, sync and process
            if pending_count >= batch_size || step == max_new_tokens - 1 {
                if !profiling_per_kernel {
                    // 1. Commit/End Capture
                    let end_capture_start = std::time::Instant::now();
                    let cmd = foundry.end_capture()?;
                    let end_capture_duration = end_capture_start.elapsed();

                    if emit_host_metrics {
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                            parent_op_name: "generation_loop".to_string(),
                            internal_kernel_name: "end_capture".to_string(),
                            duration_us: end_capture_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                        });
                    }

                    // 2. Block until GPU work in this batch completes (command buffer completion).
                    let wait_start = std::time::Instant::now();
                    cmd.wait_until_completed();
                    let wait_duration = wait_start.elapsed();

                    if emit_host_metrics {
                        // Emit command-buffer completion time so it shows under Forward Step in the TUI.
                        // Include `batch_size` so the UI can attribute wall time to tokens.
                        let mut cb_data = rustc_hash::FxHashMap::default();
                        cb_data.insert("batch_size".to_string(), pending_count.to_string());
                        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::GpuOpCompleted {
                            op_name: "Generation Loop/Forward Step/CB Wait".to_string(),
                            backend: "Foundry".to_string(),
                            duration_us: wait_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                            data: Some(cb_data),
                        });

                        // 3. Record total forward-step wall time (dispatch + completion), reported per token.
                        // Measuring from `batch_encode_start` captures the full wall-clock time for the batch.
                        if let Some(start) = batch_encode_start.take() {
                            let total_duration = start.elapsed();

                            // Capture duration for callback (tok/s reporting)
                            // Use the full wall-clock time for accurate throughput calculation
                            iteration_duration = Some(total_duration);

                            // Emit PER-TOKEN latency for the TUI (total_duration / tokens_in_batch)
                            // This ensures the Latency View shows ~10ms for 100 tok/s instead of ~160ms batch time.
                            let per_token_us = (total_duration.as_micros() / pending_count as u128).min(u128::from(u64::MAX)) as u64;
                            metallic_instrumentation::record_metric_async!(
                                metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                                    parent_op_name: "generation_loop".to_string(),
                                    internal_kernel_name: "forward_step_total".to_string(),
                                    duration_us: per_token_us,
                                }
                            );
                        }
                    }
                } else {
                    // Profiling mode: per-kernel metrics already emitted in dispatch_pipeline()
                    // But we still need to track iteration_duration for tok/s in the footer

                    // Synchronize any remaining work. In profiling mode, dispatch_pipeline already waits per kernel.
                    foundry.synchronize()?;

                    // Measure actual step duration for accurate tok/s
                    if emit_host_metrics {
                        iteration_duration = Some(step_start.elapsed());
                    }
                }

                // Process the batch results
                let process_start = std::time::Instant::now();
                let mut batch_done = false;
                for i in 0..pending_count {
                    let sample_out_name = format!("sample_out_{i}");
                    let sample_out_arg = bindings.get(&sample_out_name)?;
                    let token_buf = sample_out_arg.buffer.as_ref().unwrap();
                    let token: u32 = token_buf.read_scalar();

                    generated.push(token);

                    if !ignore_eos_stop && stop_tokens.contains(&token) {
                        batch_done = true;
                        break;
                    }
                    if !callback(
                        token,
                        prefill_duration,
                        setup_duration,
                        iteration_duration.map(|d| d / pending_count as u32),
                    )? {
                        batch_done = true;
                        break;
                    }
                }
                let process_duration = process_start.elapsed();
                if emit_host_metrics && !process_duration.is_zero() {
                    metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
                        parent_op_name: "generation_loop".to_string(),
                        internal_kernel_name: "token_callback".to_string(),
                        duration_us: process_duration.as_micros().min(u128::from(u64::MAX)) as u64,
                    });
                }
                pending_count = 0;

                if batch_done {
                    break;
                }
            }
        }

        session.current_pos += prompt_len + generated.len();
        tracing::info!(
            "Foundry loop done. Tokens: {:?} (len={}), New Pos: {}",
            generated,
            generated.len(),
            session.current_pos
        );

        Ok(generated)
    }

    /// Generate multiple tokens autoregressively with an explicit base seed for sampling.
    ///
    /// The seed is advanced once per generated token (`seed + step`) to avoid pathological repetition
    /// while keeping the output deterministic for parity testing.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_seed(
        &self,
        foundry: &mut Foundry,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u32,
    ) -> Result<Vec<u32>, MetalError> {
        self.generate_with_seed_streaming(
            foundry,
            prompt_tokens,
            max_new_tokens,
            stop_tokens,
            temperature,
            top_k,
            top_p,
            seed,
            |_, _, _, _| Ok(true),
        )
    }

    /// Report memory metrics for the model weights and host memory (activations + KV cache).
    pub fn report_memory_metrics(&self) {
        use std::collections::BTreeMap;

        use rustc_hash::FxHashMap;

        use crate::gguf::model_loader::GGUFModel;

        // 1. Report Model Weights
        let mut weight_breakdown = FxHashMap::default();
        let mut total_weights = 0u64;

        // Iterate via tensor_names to be safe and robust
        for name in self.weights.tensor_names() {
            if let Some(tensor) = self.weights.get_tensor(name) {
                let dtype = GGUFModel::map_dtype(tensor.data_type());
                let size = tensor.len() as u64 * dtype.size_bytes() as u64;
                weight_breakdown.insert(name.clone(), size);
                total_weights += size;
            }
        }

        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::ModelWeights {
            total_bytes: total_weights,
            breakdown: weight_breakdown,
        });

        // 2. Report Host Memory (Activations + KV Cache)
        if let Some(session) = &*self.session.borrow() {
            let mut tensor_pool_used = 0u64;
            let mut kv_pool_used = 0u64;

            // HostMemory breakdown expects BTreeMap<usize, (String, BTreeMap<String, u64>)>
            let mut forward_breakdown: BTreeMap<usize, (String, BTreeMap<String, u64>)> = BTreeMap::new();

            let mut io_map = BTreeMap::new();
            let mut act_map = BTreeMap::new();
            let mut kv_map = BTreeMap::new();

            let mut visited_ptrs = std::collections::HashSet::new();

            for (name, arg) in session.bindings.iter() {
                // Identify buffer uniqueness by pointer
                // TensorArg buffer is Option<MetalBuffer>
                if let Some(buf) = &arg.buffer {
                    // Use Retained::as_ptr to get the raw pointer
                    let ptr = buf.as_ptr_addr();

                    // Skip if already counted (aliased bindings)
                    if !visited_ptrs.insert(ptr) {
                        continue;
                    }

                    // Skip weights (heuristic: name in GGUF or starts with "rope")
                    if self.weights.get_tensor(name).is_some() || name.starts_with("rope") {
                        continue;
                    }

                    let size = buf.length() as u64;

                    // Classify
                    if name.contains("cache") {
                        kv_pool_used += size;
                        kv_map.insert(name.clone(), size);
                    } else if name.starts_with("input_ids") || name.starts_with("sample_out") {
                        tensor_pool_used += size;
                        io_map.insert(name.clone(), size);
                    } else {
                        tensor_pool_used += size;
                        act_map.insert(name.clone(), size);
                    }
                }
            }

            if !io_map.is_empty() {
                forward_breakdown.insert(0, ("IO".to_string(), io_map));
            }
            if !act_map.is_empty() {
                forward_breakdown.insert(1, ("Activations".to_string(), act_map));
            }
            if !kv_map.is_empty() {
                forward_breakdown.insert(2, ("KV Cache".to_string(), kv_map));
            }

            metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::HostMemory {
                total_bytes: tensor_pool_used + kv_pool_used,
                tensor_pool_reserved_bytes: tensor_pool_used,
                tensor_pool_used_bytes: tensor_pool_used,
                kv_pool_reserved_bytes: kv_pool_used,
                kv_pool_used_bytes: kv_pool_used,
                forward_pass_breakdown: forward_breakdown,
            });
        }
    }

    /// Convenience helper: apply the model's chat template (when available), tokenize, and generate.
    ///
    pub fn generate_from_prompt(
        &self,
        foundry: &mut Foundry,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<Vec<u32>, MetalError> {
        let tokenizer = self.tokenizer()?;
        let prompt_tokens = tokenizer.encode_single_turn_chat_prompt(prompt)?;
        let eos = tokenizer.special_tokens().eos_token_id.unwrap_or(151645);
        self.generate(foundry, &prompt_tokens, max_new_tokens, &[eos], temperature, top_k, top_p)
    }

    /// Convenience helper: generate and decode newly generated text (not including the prompt).
    pub fn generate_text_from_prompt(
        &self,
        foundry: &mut Foundry,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: u32,
        top_p: f32,
    ) -> Result<String, MetalError> {
        let tokenizer = self.tokenizer()?;
        let tokens = self.generate_from_prompt(foundry, prompt, max_new_tokens, temperature, top_k, top_p)?;
        tokenizer.decode(&tokens)
    }

    /// Allocate a U32 buffer for tokens.
    fn allocate_u32_buffer(&self, foundry: &mut Foundry, name: &str, count: usize) -> Result<crate::types::MetalBuffer, MetalError> {
        let byte_size = count * 4; // u32 = 4 bytes
        let buffer = foundry
            .device
            .new_buffer(byte_size, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        Ok(buffer)
    }

    // Keep old method for backward compatibility, delegating to new one
    /// Get the compiled symbol ID for a tensor name.
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

#[cfg(test)]
mod tests {
    #[test]
    fn test_rebalanced_prefill_chunk_size_avoids_tiny_tail() {
        let rebalance = |prompt_len: usize, requested: usize, max_allowed: usize| -> usize {
            let requested = requested.max(1).min(max_allowed.max(1));
            if prompt_len <= 1 {
                return 1;
            }
            let chunks = prompt_len.div_ceil(requested);
            let balanced = prompt_len.div_ceil(chunks);
            balanced.max(1).min(max_allowed)
        };

        // Repro shape: 34 with requested 32 becomes 17+17 (no 2-token tail).
        assert_eq!(rebalance(34, 32, 32), 17);
        // Already balanced.
        assert_eq!(rebalance(1000, 32, 32), 32);
        // Single chunk.
        assert_eq!(rebalance(31, 32, 32), 31);
        // Degenerate.
        assert_eq!(rebalance(0, 32, 32), 1);
        assert_eq!(rebalance(1, 32, 32), 1);
    }

    use super::*;
    use crate::{Foundry, spec::ModelSpec};

    #[test]
    #[serial_test::serial]
    fn test_model_session_kv_growth() -> Result<(), crate::error::MetalError> {
        let spec = ModelSpec::from_json(
            r#"
            {
              "name": "test-growth",
              "architecture": {
                "d_model": 128,
                "n_heads": 2,
                "n_kv_heads": 1,
                "n_layers": 1,
                "ff_dim": 256,
                "vocab_size": 100,
                "max_seq_len": 4096,
                "rope_base": 10000.0,
                "rms_eps": 0.000001,
                "prepare": {
                  "rope": { "cos": "rope_cos", "sin": "rope_sin" },
                  "tensors": [
                    {
                      "name": "k_cache_{i}",
                      "repeat": { "count": "n_layers", "var": "i" },
                      "storage": "kv_cache",
                      "dims": ["n_heads", "max_seq_len", "d_model / n_heads"],
                      "grow_with_kv": true
                    },
                    {
                      "name": "v_cache_{i}",
                      "repeat": { "count": "n_layers", "var": "i" },
                      "storage": "kv_cache",
                      "dims": ["n_heads", "max_seq_len", "d_model / n_heads"],
                      "grow_with_kv": true
                    }
                  ]
                },
                "forward": []
              }
            }
            "#,
        )
        .map_err(|e| crate::error::MetalError::InvalidOperation(e.to_string()))?;

        // We don't need real weights for allocation testing
        let weights = super::WeightBundle::new_empty();
        let model = CompiledModel::new(spec, weights)?;

        let mut foundry = Foundry::new()?;
        model.initialize_session(&mut foundry)?;

        {
            let mut session_guard = model.session.borrow_mut();
            let session = session_guard.as_mut().unwrap();

            // Initial capacity should be 2048 (default) aligned to 128
            assert_eq!(session.context_config.allocated_capacity, 2048);

            // Grow to 3000. Geometric growth (2x) from 2048 is 4096.
            model.ensure_kv_capacity(
                &mut foundry,
                &mut session.bindings,
                &mut session.fast_bindings,
                &mut session.context_config,
                session.current_pos,
                3000,
            )?;
            assert_eq!(session.context_config.allocated_capacity, 4096);

            // Should be present in bindings
            assert!(session.bindings.contains("k_cache_0"));
            let k_cache = session.bindings.get("k_cache_0")?;
            // Shape: [n_heads, allocated_capacity, head_dim] -> [2, 4096, 64]
            assert_eq!(k_cache.dims.as_slice(), &[2, 4096, 64]);
        }

        Ok(())
    }
}
