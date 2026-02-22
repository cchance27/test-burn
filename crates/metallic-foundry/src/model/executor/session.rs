use super::*;

impl CompiledModel {
    #[inline]
    pub fn clear_session(&self) {
        let mut session = self.session.lock();
        *session = None;
        tracing::debug!(
            target: "metallic_foundry::model::executor",
            model = self.name(),
            cache_fingerprint = self.cache_namespace_fingerprint(),
            cache_entries = self.kv_prefix_cache.lock().entries.len(),
            "Session cleared; prefix KV cache retained"
        );
    }

    #[inline]
    pub fn rewind_session(&self) {
        let mut session = self.session.lock();
        if let Some(session) = session.as_mut() {
            session.current_pos = 0;
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                cache_entries = self.kv_prefix_cache.lock().entries.len(),
                "Session rewound to position 0; bindings retained"
            );
        } else {
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                "Session rewind requested but no active session was present"
            );
        }
    }

    /// Report memory metrics for weights and active session buffers.
    pub fn report_memory_metrics(&self) {
        use std::collections::{BTreeMap, HashSet};

        let mut weight_breakdown = FxHashMap::default();
        let mut total_weights = 0u64;

        for name in self.weights.tensor_names() {
            if let Some(tensor_info) = self.weights.get_tensor_info(&name) {
                let dtype = tensor_info.data_type;
                let elements: usize = tensor_info.dimensions.iter().product();
                let size = elements as u64 * dtype.size_bytes() as u64;
                weight_breakdown.insert(name, size);
                total_weights += size;
            }
        }

        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::ModelWeights {
            total_bytes: total_weights,
            breakdown: weight_breakdown,
        });

        if let Some(session) = &*self.session.lock() {
            let mut tensor_pool_used = 0u64;
            let mut kv_pool_used = 0u64;
            let mut forward_breakdown: BTreeMap<usize, (String, BTreeMap<String, u64>)> = BTreeMap::new();
            let mut io_map = BTreeMap::new();
            let mut act_map = BTreeMap::new();
            let mut kv_map = BTreeMap::new();
            let mut visited_ptrs = HashSet::new();

            for (name, arg) in session.bindings.iter() {
                if let Some(buf) = &arg.buffer {
                    let ptr = crate::types::Buffer::as_ptr_addr(buf);
                    if !visited_ptrs.insert(ptr) {
                        continue;
                    }
                    if self.weights.get_tensor_info(name).is_some() || name.starts_with("rope") {
                        continue;
                    }

                    let size = buf.length() as u64;
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

    #[inline]
    fn interned_key(&self, key: &str) -> &'static str {
        {
            let map = self.interned_globals.read();
            if let Some(v) = map.get(key) {
                return v;
            }
        }

        let leaked: &'static str = Box::leak(key.to_string().into_boxed_str());
        self.interned_globals.write().insert(key.to_string(), leaked);
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
    pub(super) fn set_global_f32(&self, bindings: &mut TensorBindings, key: &str, value: f32) {
        bindings.set_global(key, value.to_string());
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn set_runtime_window_globals_with_keys(
        &self,
        bindings: &mut TensorBindings,
        m_key: &str,
        seq_len_key: &str,
        position_offset_key: &str,
        m: usize,
        seq_len: usize,
        position_offset: usize,
        apply_derived_globals: bool,
    ) {
        self.set_int_global(bindings, m_key, m);
        self.set_int_global(bindings, seq_len_key, seq_len);
        self.set_int_global(bindings, position_offset_key, position_offset);
        if apply_derived_globals {
            self.apply_derived_globals(bindings);
        }
    }

    #[inline]
    pub(crate) fn set_runtime_window_globals(
        &self,
        bindings: &mut TensorBindings,
        m: usize,
        seq_len: usize,
        position_offset: usize,
        apply_derived_globals: bool,
    ) {
        self.set_runtime_window_globals_with_keys(
            bindings,
            "m",
            "seq_len",
            "position_offset",
            m,
            seq_len,
            position_offset,
            apply_derived_globals,
        );
    }

    #[inline]
    pub(crate) fn bind_u32_input_window(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        input_binding: &str,
        source_buffer: crate::types::MetalBuffer,
        token_count: usize,
        token_offset: usize,
    ) {
        let mut tensor_input = TensorArg::from_buffer(source_buffer, crate::tensor::Dtype::U32, vec![token_count], vec![1]);
        tensor_input.offset = token_offset * std::mem::size_of::<u32>();
        self.set_binding(bindings, fast_bindings, input_binding, tensor_input);
    }

    #[inline]
    pub(crate) fn binding_buffer_clone(&self, bindings: &TensorBindings, name: &str) -> Result<crate::types::MetalBuffer, MetalError> {
        let arg = bindings.get(name)?;
        arg.buffer
            .as_ref()
            .cloned()
            .ok_or_else(|| MetalError::InvalidOperation(format!("Binding '{name}' does not reference a buffer")))
    }

    pub(crate) fn apply_derived_globals(&self, bindings: &mut TensorBindings) {
        for spec in &self.spec.architecture.prepare.derived_globals {
            let value = spec.expr.eval(bindings);
            self.set_int_global(bindings, &spec.name, value);
        }
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
        if let Some(model_arch) = weights.model().architecture() {
            tracing::debug!("Loading model: spec='{}' model_arch='{}'", spec.name, model_arch);
        }

        tracing::info!("Compiling model with {} forward DSL steps...", spec.architecture.forward.len());

        // Compiler setup
        let mut symbols = SymbolTable::new();
        let mut resolver = TensorBindings::new();
        let arch = &spec.architecture;

        // Set config globals for DSL variable interpolation (needed for Repeat unrolling, etc.)
        for (name, val) in &arch.params {
            match val {
                ArchValue::USize(v) => resolver.set_global(name.clone(), v.to_string()),
                ArchValue::F32(v) => resolver.set_global(name.clone(), v.to_string()),
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

            RwLock::new(map)
        };

        let cache_namespace_fingerprint = Self::compute_cache_namespace_fingerprint(&spec, &weights);

        Ok(Self {
            spec,
            weights,
            cache_namespace_fingerprint,
            compiled_steps,
            symbol_table: symbols,
            interned_globals,
            session: Mutex::new(None),
            kv_prefix_cache: Mutex::new(KvPrefixCache::default()),
            instance_cache: Mutex::new(FxHashMap::default()),
        })
    }

    /// Get the model name from the spec.
    pub fn name(&self) -> &str {
        &self.spec.name
    }

    pub fn cache_namespace_fingerprint(&self) -> &str {
        &self.cache_namespace_fingerprint
    }

    /// Get the architecture configuration.
    pub fn architecture(&self) -> &crate::spec::Architecture {
        &self.spec.architecture
    }

    /// Get access to the underlying weights bundle.
    pub fn weights(&self) -> &WeightBundle {
        &self.weights
    }

    /// Get access to the underlying loaded model metadata for tensor materialization.
    pub fn metadata(&self) -> &dyn ModelMetadata {
        self.weights.model().metadata()
    }

    fn get_or_init_cached_instance<T, F>(&self, key: &str, build: F) -> Result<std::sync::Arc<T>, MetalError>
    where
        T: 'static + Send + Sync,
        F: FnOnce() -> Result<T, MetalError>,
    {
        // PERF: The typed key avoids repeated construction of heavyweight helpers (tokenizers,
        // parsers, adapters) while keeping the cache generic for non-LLM workflows.
        let cache_key = (TypeId::of::<T>(), key.to_string());

        if let Some(existing) = self.instance_cache.lock().get(&cache_key).cloned() {
            return existing
                .downcast::<T>()
                .map_err(|_| MetalError::InvalidOperation(format!("CompiledModel instance_cache type mismatch for key '{key}'")));
        }

        // PERF: Build outside the cache lock so expensive constructors do not serialize unrelated
        // cache users. We re-check on insert to handle a concurrent winner.
        let built = std::sync::Arc::new(build()?);
        let mut cache = self.instance_cache.lock();
        if let Some(existing) = cache.get(&cache_key).cloned() {
            return existing
                .downcast::<T>()
                .map_err(|_| MetalError::InvalidOperation(format!("CompiledModel instance_cache type mismatch for key '{key}'")));
        }
        let built_any: std::sync::Arc<dyn Any + Send + Sync> = built.clone();
        cache.insert(cache_key, built_any);
        Ok(built)
    }

    /// Get the shared tokenizer for this compiled model.
    ///
    /// The tokenizer is constructed once from loaded model metadata (plus optional ModelSpec chat-template
    /// override) and then reused across workflow ops to avoid repeated reconstruction cost.
    pub fn tokenizer(&self) -> Result<std::sync::Arc<crate::BPETokenizer>, MetalError> {
        // PERF: Use the generic instance cache instead of rebuilding tokenizer + regex state for
        // every tokenize op invocation.
        self.get_or_init_cached_instance("tokenizer.default", || {
            let mut tokenizer = crate::BPETokenizer::from_metadata(self.weights.model().metadata())?;

            // Prioritize template from ModelSpec (DSL override)
            if let Some(template_override) = &self.spec.chat_template {
                tokenizer.set_chat_template(template_override.clone());
            }

            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                has_spec_chat_template = self.spec.chat_template.is_some(),
                "Initialized cached model instance key=tokenizer.default"
            );

            Ok(tokenizer)
        })
    }

    pub fn initialize_session(&self, foundry: &mut Foundry) -> Result<(), MetalError> {
        let mut session = self.session.lock();
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
            Self::decode_batch_size_or(default_decode_batch_size, MAX)
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

    /// Reset the session, clearing the KV cache and position state.
    ///
    /// Call this when switching conversations or when you need a fresh context.
    /// The next inference will re-initialize the session with `current_pos = 0`.
    pub fn reset_session(&self) {
        let mut session = self.session.lock();
        tracing::debug!(
            "Session reset requested for model {}, session is_some: {}",
            self.name(),
            session.is_some()
        );
        if session.is_some() {
            tracing::info!("Resetting model session (clearing KV cache)");
            *session = None;
            let cache_entries = self.kv_prefix_cache.lock().entries.len();
            tracing::debug!(
                target: "metallic_foundry::model::executor",
                model = self.name(),
                cache_fingerprint = self.cache_namespace_fingerprint(),
                cache_entries,
                "Session reset completed; prefix KV cache retained"
            );
        }
    }

    pub(crate) fn with_session_mut<T>(
        &self,
        foundry: &mut Foundry,
        f: impl FnOnce(&mut Foundry, &mut ModelSession) -> Result<T, MetalError>,
    ) -> Result<T, MetalError> {
        self.initialize_session(foundry)?;
        let mut session_guard = self.session.lock();
        let session = session_guard
            .as_mut()
            .ok_or_else(|| MetalError::OperationFailed("Foundry session missing after initialization".into()))?;
        f(foundry, session)
    }
}
