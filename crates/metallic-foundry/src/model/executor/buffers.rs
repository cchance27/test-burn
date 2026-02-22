use metallic_env::{FoundryEnvVar, is_set};

use super::*;

impl CompiledModel {
    fn storage_mode_for(storage: StorageClass) -> MetalResourceOptions {
        match storage {
            StorageClass::Intermediate => {
                if is_set(FoundryEnvVar::IntermediatesShared) {
                    MetalResourceOptions::StorageModeShared
                } else {
                    MetalResourceOptions::StorageModePrivate
                }
            }
            StorageClass::KvCache => {
                if is_set(FoundryEnvVar::KvCacheShared) {
                    MetalResourceOptions::StorageModeShared
                } else {
                    MetalResourceOptions::StorageModePrivate
                }
            }
            StorageClass::RopeCache => {
                if is_set(FoundryEnvVar::RopeShared) {
                    MetalResourceOptions::StorageModeShared
                } else {
                    MetalResourceOptions::StorageModePrivate
                }
            }
            StorageClass::Shared => MetalResourceOptions::StorageModeShared,
            StorageClass::Private => MetalResourceOptions::StorageModePrivate,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn allocate_tensor_from_spec(
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
        if dims.contains(&0) {
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

    pub(super) fn allocate_prepare_tensors(
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
                    let dims: Vec<usize> = tensor.dims.iter().map(|e| e.eval(bindings)).collect::<Vec<usize>>();
                    let strides: Vec<usize> = if let Some(strides) = &tensor.strides {
                        strides.iter().map(|e| e.eval(bindings)).collect::<Vec<usize>>()
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
                let dims: Vec<usize> = tensor.dims.iter().map(|e| e.eval(bindings)).collect::<Vec<usize>>();
                let strides: Vec<usize> = if let Some(strides) = &tensor.strides {
                    strides.iter().map(|e| e.eval(bindings)).collect::<Vec<usize>>()
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

    pub(super) fn zero_tensor_arg(&self, foundry: &mut Foundry, size: usize) -> Result<TensorArg, MetalError> {
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

    /// Helper to insert a tensor into both string and fast bindings
    pub(super) fn insert_binding(&self, bindings: &mut TensorBindings, fast_bindings: &mut FastBindings, name: String, tensor: TensorArg) {
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

    /// Allocate a KV cache buffer for attention caching.
    /// Shape: [n_heads, max_seq_len, head_dim]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn allocate_u32_buffer(
        &self,
        foundry: &mut Foundry,
        name: &str,
        count: usize,
    ) -> Result<crate::types::MetalBuffer, MetalError> {
        let byte_size = count * 4; // u32 = 4 bytes
        let buffer = foundry
            .device
            .new_buffer(byte_size, MetalResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError::OperationFailed(format!("Failed to allocate {} bytes for '{}'", byte_size, name)))?;

        Ok(buffer)
    }
}
