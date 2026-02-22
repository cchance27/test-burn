use metallic_loader::TensorNameIndex;

use super::*;

impl CompiledModel {
    fn required_prepare_vars(&self) -> FxHashSet<&str> {
        let arch = self.architecture();
        let mut vars: FxHashSet<&str> = FxHashSet::default();

        // prepare.globals expressions
        for expr in arch.prepare.globals.values() {
            for v in expr.vars() {
                let v_str: &str = v.as_ref();
                vars.insert(v_str);
            }
        }
        // derived globals expressions
        for g in &arch.prepare.derived_globals {
            for v in g.expr.vars() {
                let v_str: &str = v.as_ref();
                vars.insert(v_str);
            }
        }
        // tensor dims/strides expressions
        for t in &arch.prepare.tensors {
            for e in &t.dims {
                for v in e.vars() {
                    let v_str: &str = v.as_ref();
                    vars.insert(v_str);
                }
            }
            if let Some(strides) = &t.strides {
                for e in strides {
                    for v in e.vars() {
                        let v_str: &str = v.as_ref();
                        vars.insert(v_str);
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
                for v in (z as &IntExpr).vars() {
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

    fn seed_prepare_globals(&self, bindings: &mut TensorBindings) {
        let arch = self.architecture();
        let required = self.required_prepare_vars();

        // Seed all architecture parameters.
        for (name, val) in &arch.params {
            // Runtime overrides (e.g. memory-budget-clamped `max_seq_len`) must win over loader baseline.
            // `prepare_bindings_with_config` may seed some globals (like max_seq_len/max_prefill_chunk)
            // before calling this function.
            if bindings.get_int_global(name).is_some() {
                continue;
            }
            match val {
                ArchValue::USize(v) => self.set_global_usize(bindings, name, *v),
                ArchValue::F32(v) => self.set_global_f32(bindings, name, *v),
            }
        }

        // Derived globals often depend on d_model/n_heads (e.g. head_dim).
        // Ensure these are available as int globals for IntExpr evaluation.
        if let Some(v) = arch.params.get("d_model").and_then(|v: &ArchValue| v.as_usize()) {
            self.set_int_global(bindings, "d_model", v);
        }
        if let Some(v) = arch.params.get("n_heads").and_then(|v: &ArchValue| v.as_usize()) {
            self.set_int_global(bindings, "n_heads", v);
        }
        if let Some(v) = arch.params.get("n_kv_heads").and_then(|v: &ArchValue| v.as_usize()) {
            self.set_int_global(bindings, "n_kv_heads", v);
        }

        // Default runtime globals; workflows can override these at runtime.
        // Set these together so we keep the runtime window state coherent.
        if required.contains("m") || required.contains("seq_len") || required.contains("position_offset") {
            let m = if required.contains("m") {
                1
            } else {
                bindings.get_int_global("m").unwrap_or(1)
            };
            let seq_len = if required.contains("seq_len") {
                1
            } else {
                bindings.get_int_global("seq_len").unwrap_or(1)
            };
            let position_offset = if required.contains("position_offset") {
                0
            } else {
                bindings.get_int_global("position_offset").unwrap_or(0)
            };
            self.set_runtime_window_globals(bindings, m, seq_len, position_offset, false);
        }

        for (key, expr) in &arch.prepare.globals {
            let value: usize = expr.eval(bindings);
            self.set_global_usize(bindings, key, value);
        }

        // Derived globals may be used by tensor dims (e.g. head_dim/kv_dim).
        self.apply_derived_globals(bindings);
    }

    pub fn prepare_bindings_with_config(
        &self,
        foundry: &mut Foundry,
        context_config: &crate::model::ContextConfig,
    ) -> Result<(TensorBindings, FastBindings), MetalError> {
        let mut bindings = TensorBindings::new();
        let mut fast_bindings = FastBindings::new(self.symbol_table.len());

        let _model = self.weights.model();
        let arch = &self.spec.architecture;
        let _tensor_names = &arch.tensor_names;
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

        // Build loader-side tensor-name index for format-agnostic logical resolution.
        let tensor_name_index = TensorNameIndex::from_model(self.weights.model());

        // 2. Bind weights either via DSL-declared weight_bindings (preferred) or legacy hardcoded maps.
        // 2. Bind weights via DSL-declared weight_bindings.
        self.bind_weights_from_spec(foundry, &mut bindings, &mut fast_bindings, &tensor_name_index)?;

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
}
