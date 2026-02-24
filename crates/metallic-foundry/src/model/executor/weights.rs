use metallic_loader::TensorNameIndex;

use super::*;

impl CompiledModel {
    pub(super) fn bind_weights_from_spec(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        tensor_names: &TensorNameIndex,
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
                    self.bind_one_weight_binding(foundry, bindings, fast_bindings, tensor_names, &mut zero_cache, spec, layer_idx)?;
                }
                bindings.pop_scope();
            } else {
                if spec.key.starts_with("layer.") {
                    return Err(MetalError::InvalidShape(format!(
                        "weight_bindings key '{}' requires repeat to bind per-layer tensors",
                        spec.key
                    )));
                }
                self.bind_one_weight_binding(foundry, bindings, fast_bindings, tensor_names, &mut zero_cache, spec, None)?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_one_weight_binding(
        &self,
        foundry: &mut Foundry,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        tensor_name_index: &TensorNameIndex,
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

        if let Some(source_name) = tensor_names.resolve_with(&spec.key, layer_idx, |name| tensor_name_index.contains(name)) {
            match &spec.layout {
                WeightLayoutSpec::RowMajor => {
                    self.bind_model_tensor(bindings, fast_bindings, foundry, &source_name, &logical_name)?;
                }
                WeightLayoutSpec::Canonical { expected_k, expected_n } => {
                    let k = expected_k.eval(bindings);
                    let n = expected_n.eval(bindings);
                    self.bind_model_tensor_canonical(bindings, fast_bindings, foundry, &source_name, &logical_name, k, n)?;
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
            "Model tensor not found for weight_bindings key='{}' layer_idx={layer_idx:?} logical_name='{}'",
            spec.key, logical_name
        )))
    }

    /// Bind a model tensor to bindings under a logical name.
    fn bind_model_tensor(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        source_name: &str,
        logical_name: &str,
    ) -> Result<(), MetalError> {
        self.bind_model_tensor_with_layout(bindings, fast_bindings, foundry, source_name, logical_name, Layout::RowMajor)
    }

    /// Bind a model tensor to bindings in canonical k-block-major layout.
    /// Used for 2D weight matrices with GemvCanonical kernel.
    #[allow(clippy::too_many_arguments)]
    fn bind_model_tensor_canonical(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        source_name: &str,
        logical_name: &str,
        expected_k: usize,
        expected_n: usize,
    ) -> Result<(), MetalError> {
        self.bind_model_tensor_with_layout(
            bindings,
            fast_bindings,
            foundry,
            source_name,
            logical_name,
            Layout::Canonical { expected_k, expected_n },
        )
    }

    /// Helper to bind a model tensor using the quantization policy system.
    fn bind_model_tensor_with_layout(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        source_name: &str,
        logical_name: &str,
        layout: Layout,
    ) -> Result<(), MetalError> {
        let model = self.weights.model();
        let tensor_info = model
            .tensor_info(source_name)
            .ok_or_else(|| MetalError::InputNotFound(format!("Tensor '{}' not found in model", source_name)))?;

        let policy = resolve_policy(tensor_info.data_type);

        let loaded_tensors = policy
            .load_weights(foundry, model, source_name, logical_name, layout)
            .map_err(|e| MetalError::OperationFailed(format!("Policy load failed for '{}': {}", source_name, e)))?;

        for (name, tensor_arg) in loaded_tensors {
            self.insert_binding(bindings, fast_bindings, name.clone(), tensor_arg.clone());

            if name == logical_name {
                self.insert_binding(bindings, fast_bindings, source_name.to_string(), tensor_arg);
                tracing::trace!("Bound '{}' -> '{}' using policy {}", logical_name, source_name, policy.short_name());
            } else {
                tracing::trace!("Bound derived '{}' using policy {}", name, policy.short_name());
            }
        }

        Ok(())
    }
}
