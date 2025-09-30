use super::*;

use crate::metallic::Context;
use crate::metallic::Dtype;
use crate::metallic::MetalError;
use crate::metallic::Tensor;
use crate::metallic::TensorElement;
use crate::metallic::TensorInit;
use crate::metallic::TensorStorage;
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddInplaceOp;
use objc2_metal::MTLBlitCommandEncoder;

/// SwiGLU operation that computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
pub struct SwiGLUOp;

/// Dummy struct for SwiGLU operation since all work is done in the `new` method
pub struct SwiGLU<T: TensorElement> {
    _phantom: std::marker::PhantomData<T>,
}

pub struct SwiGLUFusedActivationOp;

struct SwiGLUFusedActivation<T: TensorElement> {
    gate: Tensor<T>,
    gate_bias: Tensor<T>,
    up_inout: Tensor<T>,
    up_bias: Tensor<T>,
    total_elements: u32,
    bias_len: u32,
    vector_width: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for SwiGLUFusedActivationOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::SwigluFusedActivation)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (gate, gate_bias, up, up_bias) = args;

        if gate.dims() != up.dims() {
            return Err(MetalError::InvalidShape(format!(
                "SwiGLU fused activation expects matching gate/up dims, got {:?} vs {:?}",
                gate.dims(),
                up.dims()
            )));
        }

        if gate_bias.dims().len() != 1 {
            return Err(MetalError::InvalidShape(format!(
                "Gate bias must be 1D, got {:?}",
                gate_bias.dims()
            )));
        }

        if up_bias.dims().len() != 1 {
            return Err(MetalError::InvalidShape(format!("Up bias must be 1D, got {:?}", up_bias.dims())));
        }

        let dims = gate.dims();
        if dims.is_empty() {
            return Err(MetalError::InvalidShape(
                "SwiGLU fused activation expects non-empty dims".to_string(),
            ));
        }

        let hidden_dim = *dims.last().expect("non-empty dims");
        if gate_bias.len() != hidden_dim {
            return Err(MetalError::DimensionMismatch {
                expected: hidden_dim,
                actual: gate_bias.len(),
            });
        }

        if up_bias.len() != hidden_dim {
            return Err(MetalError::DimensionMismatch {
                expected: hidden_dim,
                actual: up_bias.len(),
            });
        }

        ctx.prepare_tensors_for_active_cmd(&[&gate, &gate_bias, &up, &up_bias])?;

        let out = up.clone();
        let vector_width = if matches!(T::DTYPE, Dtype::F32 | Dtype::F16) && hidden_dim % 4 == 0 {
            4
        } else {
            1
        };

        let op = SwiGLUFusedActivation {
            gate,
            gate_bias,
            up_inout: out.clone(),
            up_bias,
            total_elements: out.len() as u32,
            bias_len: hidden_dim as u32,
            vector_width,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for SwiGLUFusedActivation<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let vector_width = std::cmp::max(self.vector_width as usize, 1);
        let base_threads = 256usize;
        let threads_per_tg_width = std::cmp::max(base_threads / vector_width, 1);
        let threads_per_tg = MTLSize {
            width: threads_per_tg_width,
            height: 1,
            depth: 1,
        };
        let total_threads = if self.vector_width > 1 {
            let vectorized = self.total_elements / self.vector_width;
            let remainder = self.total_elements % self.vector_width;
            vectorized + remainder
        } else {
            self.total_elements
        };

        let groups = MTLSize {
            width: ((total_threads as usize) + threads_per_tg.width - 1) / threads_per_tg.width,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.gate.buf, self.gate.offset);
        set_buffer(&encoder, 1, &self.up_inout.buf, self.up_inout.offset);
        set_buffer(&encoder, 2, &self.gate_bias.buf, self.gate_bias.offset);
        set_buffer(&encoder, 3, &self.up_bias.buf, self.up_bias.offset);
        set_bytes(&encoder, 4, &self.total_elements);
        set_bytes(&encoder, 5, &self.bias_len);
        set_bytes(&encoder, 6, &self.vector_width);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();

        Ok(())
    }
}

impl KernelInvocable for SwiGLUOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>,
        &'a Tensor<T>,
        &'a Tensor<T>,
        &'a Tensor<T>,
        &'a Tensor<T>,
        &'a Tensor<T>,
        Option<&'a Tensor<T>>,
    );

    fn function_id() -> Option<KernelFunction> {
        // This is a composite operation using existing kernels, so we don't need a specific kernel function
        None
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (x_normed_flat, ffn_gate, ffn_gate_bias, ffn_up, ffn_up_bias, ffn_down, ffn_down_bias, fused_gate_up) = args;

        // Execute the SwiGLU operation logic directly in the new method
        let output = execute_swiglu_logic(
            ctx,
            x_normed_flat,
            ffn_gate,
            ffn_gate_bias,
            ffn_up,
            ffn_up_bias,
            ffn_down,
            ffn_down_bias,
            fused_gate_up,
            cache,
        )?;

        // Create a dummy operation since all work is done in this function
        Ok((
            Box::new(SwiGLU {
                _phantom: std::marker::PhantomData::<T>,
            }),
            output,
        ))
    }
}

/// Execute the SwiGLU operation logic by calling the individual kernels in sequence
#[allow(clippy::too_many_arguments)]
fn execute_swiglu_logic<T: TensorElement>(
    ctx: &mut Context<T>,
    x_normed_flat: &Tensor<T>,
    ffn_gate: &Tensor<T>,
    ffn_gate_bias: &Tensor<T>,
    ffn_up: &Tensor<T>,
    ffn_up_bias: &Tensor<T>,
    ffn_down: &Tensor<T>,
    ffn_down_bias: &Tensor<T>,
    fused_gate_up: Option<&Tensor<T>>,
    mut cache: Option<&mut ResourceCache>,
) -> Result<Tensor<T>, MetalError> {
    ctx.prepare_tensors_for_active_cmd(&[x_normed_flat, ffn_gate_bias, ffn_up_bias, ffn_down, ffn_down_bias])?;
    if let Some(fused) = fused_gate_up {
        ctx.prepare_tensors_for_active_cmd(&[fused])?;
    }
    let d_model = x_normed_flat.dims()[1];
    let hidden_dim = ffn_gate_bias.len();

    if hidden_dim != ffn_up_bias.len() {
        return Err(MetalError::DimensionMismatch {
            expected: hidden_dim,
            actual: ffn_up_bias.len(),
        });
    }

    let (gate_temp, up_temp) = if let Some(fused_weight) = fused_gate_up {
        let fused_dims = fused_weight.dims();
        if fused_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "Fused gate/up weight must be 2D, got {:?}",
                fused_dims
            )));
        }

        let expected_cols = hidden_dim * 2;
        let fused_transpose_b = if fused_dims[0] == d_model && fused_dims[1] == expected_cols {
            false
        } else if fused_dims[0] == expected_cols && fused_dims[1] == d_model {
            true
        } else {
            return Err(MetalError::InvalidShape(format!(
                "Fused gate/up weight dims {:?} incompatible with d_model {} and ff_dim {}",
                fused_dims, d_model, hidden_dim
            )));
        };

        let fused_temp = match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(x_normed_flat, fused_weight, false, fused_transpose_b, cache)?,
            None => ctx.matmul(x_normed_flat, fused_weight, false, fused_transpose_b)?,
        };

        let gate_view = fused_temp.slice_last_dim(0..hidden_dim)?;
        let up_view = fused_temp.slice_last_dim(hidden_dim..expected_cols)?;
        let gate_temp = materialize_contiguous_if_needed(ctx, gate_view)?;
        let up_temp = materialize_contiguous_if_needed(ctx, up_view)?;
        (gate_temp, up_temp)
    } else {
        // gate_proj: [m, d_model] @ weight -> [m, ff_dim]
        ctx.prepare_tensors_for_active_cmd(&[ffn_gate])?;
        let gate_dims = ffn_gate.dims();
        let gate_transpose_b = if gate_dims[0] == d_model {
            false
        } else if gate_dims[1] == d_model {
            true
        } else {
            return Err(MetalError::DimensionMismatch {
                expected: d_model,
                actual: gate_dims[0],
            });
        };
        let gate_temp = match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(x_normed_flat, ffn_gate, false, gate_transpose_b, cache)?,
            None => ctx.matmul(x_normed_flat, ffn_gate, false, gate_transpose_b)?,
        };

        // up_proj: [m, d_model] @ weight -> [m, ff_dim]
        ctx.prepare_tensors_for_active_cmd(&[ffn_up])?;
        let up_dims = ffn_up.dims();
        let up_transpose_b = if up_dims[0] == d_model {
            false
        } else if up_dims[1] == d_model {
            true
        } else {
            return Err(MetalError::DimensionMismatch {
                expected: d_model,
                actual: up_dims[0],
            });
        };
        let up_temp = match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(x_normed_flat, ffn_up, false, up_transpose_b, cache)?,
            None => ctx.matmul(x_normed_flat, ffn_up, false, up_transpose_b)?,
        };

        (gate_temp, up_temp)
    };

    // Fuse bias additions, SiLU, and elementwise multiply
    let hidden = match cache.as_mut() {
        Some(cache) => {
            ctx.call_with_cache::<SwiGLUFusedActivationOp>((gate_temp, ffn_gate_bias.clone(), up_temp, ffn_up_bias.clone()), cache)?
        }
        None => ctx.call::<SwiGLUFusedActivationOp>((gate_temp, ffn_gate_bias.clone(), up_temp, ffn_up_bias.clone()))?,
    };

    // down_proj: [m, ff_dim] @ [ff_dim, d_model] -> [m, d_model]
    let hidden_cols = hidden.dims()[1];
    let ffn_down_rows = ffn_down.dims()[0];
    let ffn_down_cols = ffn_down.dims()[1];
    let ffn_temp = if hidden_cols == ffn_down_rows {
        // Hidden [m, ff_dim] @ ffn_down [ff_dim, d_model] -> [m, d_model]
        match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(&hidden, ffn_down, false, false, cache)?,
            None => ctx.matmul(&hidden, ffn_down, false, false)?,
        }
    } else if hidden_cols == ffn_down_cols {
        // Hidden [m, ff_dim] @ ffn_down^T [ff_dim, d_model] where stored as [d_model, ff_dim]
        match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(&hidden, ffn_down, false, true, cache)?,
            None => ctx.matmul(&hidden, ffn_down, false, true)?,
        }
    } else {
        return Err(MetalError::DimensionMismatch {
            expected: hidden_cols,
            actual: ffn_down_rows,
        });
    };

    // Add down bias to final projection output
    let ffn_out = match cache.as_mut() {
        Some(cache) => ctx.call_with_cache::<BroadcastElemwiseAddInplaceOp>((ffn_temp, ffn_down_bias.clone()), cache)?,
        None => ctx.call::<BroadcastElemwiseAddInplaceOp>((ffn_temp, ffn_down_bias.clone()))?,
    };

    Ok(ffn_out)
}

fn materialize_contiguous_if_needed<T: TensorElement>(ctx: &mut Context<T>, view: Tensor<T>) -> Result<Tensor<T>, MetalError> {
    if view.strides == Tensor::<T>::compute_strides(view.dims()) {
        return Ok(view);
    }

    let dims = view.dims().to_vec();
    let contiguous = Tensor::new(dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

    ctx.prepare_tensors_for_active_cmd(&[&view])?;

    let source_view = view.as_mps_matrix_batch_view()?;
    let dest_view = contiguous.as_mps_matrix_batch_view()?;
    let elem_size = view.dtype.size_bytes();

    let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
    let encoder = command_buffer
        .raw()
        .blitCommandEncoder()
        .ok_or(MetalError::OperationNotSupported("Blit encoder not available".to_string()))?;

    for batch_idx in 0..source_view.batch {
        for row_idx in 0..source_view.rows {
            let src_offset = view.offset + batch_idx * source_view.matrix_bytes + row_idx * source_view.row_bytes;
            let dst_offset = contiguous.offset + batch_idx * dest_view.matrix_bytes + row_idx * dest_view.row_bytes;
            let copy_bytes = dest_view.columns * elem_size;
            unsafe {
                encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &view.buf,
                    src_offset,
                    &contiguous.buf,
                    dst_offset,
                    copy_bytes,
                );
            }
        }
    }

    encoder.endEncoding();
    ctx.mark_tensor_pending(&contiguous);

    Ok(contiguous)
}

/// Execute the SwiGLU composite with an explicitly provided cache.
///
/// This helper mirrors the internal execution used by [`Context::SwiGLU`] but allows
/// benchmarks and diagnostics to control whether a [`ResourceCache`] is reused across the
/// composite's constituent kernels.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_with_optional_cache<T: TensorElement>(
    ctx: &mut Context<T>,
    x_normed_flat: &Tensor<T>,
    ffn_gate: &Tensor<T>,
    ffn_gate_bias: &Tensor<T>,
    ffn_up: &Tensor<T>,
    ffn_up_bias: &Tensor<T>,
    ffn_down: &Tensor<T>,
    ffn_down_bias: &Tensor<T>,
    fused_gate_up: Option<&Tensor<T>>,
    cache: Option<&mut ResourceCache>,
) -> Result<Tensor<T>, MetalError> {
    execute_swiglu_logic(
        ctx,
        x_normed_flat,
        ffn_gate,
        ffn_gate_bias,
        ffn_up,
        ffn_up_bias,
        ffn_down,
        ffn_down_bias,
        fused_gate_up,
        cache,
    )
}

// Implement `Operation` for the internal struct.
impl<T: TensorElement> Operation for SwiGLU<T> {
    fn encode(
        &self,
        _command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // Since all computation was done in the `new` method of KernelInvocable,
        // this method just returns Ok(())
        Ok(())
    }
}

#[cfg(test)]
mod swiglu_test;
