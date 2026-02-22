use metallic_env::{FoundryEnvVar, is_set};

use super::*;

impl CompiledModel {
    /// Prepare runtime RoPE caches (cos/sin) sized to `rope_len`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_and_bind_rope_caches_named(
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
        let rope_freq_factors = self.load_rope_freq_factors(dim_half)?;

        self.ensure_rope_capacity_named(
            bindings,
            fast_bindings,
            foundry,
            rope_base,
            head_dim,
            dim_half,
            rope_freq_factors.as_deref(),
            rope_cos_name,
            rope_sin_name,
            0,
            rope_len,
        )?;
        if let Some(freqs) = rope_freq_factors.as_ref() {
            let (mut min_v, mut max_v) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in freqs {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
            tracing::debug!(
                "Prepared RoPE caches: [{}, {}] (rope_base={}, rope_freq_factors=len:{} min:{} max:{})",
                rope_len,
                dim_half,
                rope_base,
                freqs.len(),
                min_v,
                max_v
            );
        } else {
            tracing::debug!("Prepared RoPE caches: [{}, {}] (rope_base={})", rope_len, dim_half, rope_base);
        }
        Ok(())
    }

    pub(super) fn load_rope_freq_factors(&self, dim_half: usize) -> Result<Option<Vec<f32>>, MetalError> {
        let model = self.weights.model();
        let Some(tensor_info) = model.tensor_info("rope_freqs.weight") else {
            return Ok(None);
        };

        if tensor_info.dimensions.len() != 1 {
            return Err(MetalError::InvalidShape(format!(
                "rope_freqs.weight must be 1D, got dims {:?}",
                tensor_info.dimensions
            )));
        }

        let data = model
            .tensor_data("rope_freqs.weight")
            .map_err(|e| MetalError::OperationFailed(format!("Failed to read rope_freqs.weight: {e}")))?;
        let factors: &[f32] = bytemuck::try_cast_slice(data.as_slice())
            .map_err(|_| MetalError::OperationFailed("Invalid F32 payload for rope_freqs.weight".to_string()))?;

        if factors.len() != dim_half {
            return Err(MetalError::InvalidShape(format!(
                "rope_freqs.weight length {} does not match dim_half {}",
                factors.len(),
                dim_half
            )));
        }
        if factors.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(MetalError::InvalidShape(
                "rope_freqs.weight contains non-finite or non-positive factors".to_string(),
            ));
        }

        Ok(Some(factors.to_vec()))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ensure_rope_capacity_named(
        &self,
        bindings: &mut TensorBindings,
        fast_bindings: &mut FastBindings,
        foundry: &mut Foundry,
        rope_base: f32,
        head_dim: usize,
        dim_half: usize,
        rope_freq_factors: Option<&[f32]>,
        rope_cos_name: &str,
        rope_sin_name: &str,
        old_len: usize,
        new_len: usize,
    ) -> Result<(), MetalError> {
        if new_len == 0 || dim_half == 0 {
            return Err(MetalError::InvalidShape("RoPE cache requires new_len>0 and dim_half>0".into()));
        }
        if old_len > new_len {
            return Err(MetalError::InvalidShape(format!(
                "RoPE cache cannot shrink: {old_len} -> {new_len}"
            )));
        }
        if let Some(factors) = rope_freq_factors
            && factors.len() != dim_half
        {
            return Err(MetalError::InvalidShape(format!(
                "RoPE factor length {} does not match dim_half {}",
                factors.len(),
                dim_half
            )));
        }

        let storage_mode = if is_set(FoundryEnvVar::RopeShared) {
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

        let old_cos = bindings.get(rope_cos_name).ok();
        let old_sin = bindings.get(rope_sin_name).ok();

        let can_reuse_existing = old_len == 0
            && old_cos
                .as_ref()
                .is_some_and(|t| t.dtype == crate::tensor::Dtype::F16 && t.dims.as_slice() == [new_len, dim_half].as_slice())
            && old_sin
                .as_ref()
                .is_some_and(|t| t.dtype == crate::tensor::Dtype::F16 && t.dims.as_slice() == [new_len, dim_half].as_slice());

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

        let nested_capture = foundry.is_capturing();
        if !nested_capture {
            foundry.start_capture()?;
        }

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

        if new_len > old_len {
            const POS_CHUNK: usize = 1024;
            let mut inv_freqs = Vec::with_capacity(dim_half);
            for i in 0..dim_half {
                let exponent = (2 * i) as f32 / head_dim as f32;
                let mut inv_freq = 1.0f32 / rope_base.powf(exponent);
                if let Some(factors) = rope_freq_factors {
                    inv_freq /= factors[i];
                }
                inv_freqs.push(inv_freq);
            }

            let mut pos = old_len;
            while pos < new_len {
                let end = (pos + POS_CHUNK).min(new_len);
                let chunk_len = end - pos;
                let chunk_elems = chunk_len * dim_half;

                let mut cos_data: Vec<f16> = vec![f16::ZERO; chunk_elems];
                let mut sin_data: Vec<f16> = vec![f16::ZERO; chunk_elems];

                for p in pos..end {
                    let local_p = p - pos;
                    for (i, &inv_freq) in inv_freqs.iter().enumerate().take(dim_half) {
                        let idx = local_p * dim_half + i;
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
}
