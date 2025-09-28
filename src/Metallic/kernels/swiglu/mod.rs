use super::*;

use crate::metallic::Context;
use crate::metallic::MetalError;
use crate::metallic::Tensor;
use crate::metallic::kernels::elemwise_add::BroadcastElemwiseAddOp;
use crate::metallic::kernels::elemwise_mul::ElemwiseMulOp;
use crate::metallic::kernels::silu::SiluOp;

/// SwiGLU operation that computes: down_proj( SiLU(gate_proj(x)) * up_proj(x) )
pub struct SwiGLUOp;

/// Dummy struct for SwiGLU operation since all work is done in the `new` method
pub struct SwiGLU;

impl KernelInvocable for SwiGLUOp {
    type Args = (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);

    fn function_id() -> Option<KernelFunction> {
        // This is a composite operation using existing kernels, so we don't need a specific kernel function
        None
    }

    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (x_normed_flat, ffn_gate, ffn_gate_bias, ffn_up, ffn_up_bias, ffn_down, ffn_down_bias) = args;

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
            cache,
        )?;

        // Create a dummy operation since all work is done in this function
        Ok((Box::new(SwiGLU), output))
    }
}

/// Execute the SwiGLU operation logic by calling the individual kernels in sequence
#[allow(clippy::too_many_arguments)]
fn execute_swiglu_logic(
    ctx: &mut Context,
    x_normed_flat: Tensor,
    ffn_gate: Tensor,
    ffn_gate_bias: Tensor,
    ffn_up: Tensor,
    ffn_up_bias: Tensor,
    ffn_down: Tensor,
    ffn_down_bias: Tensor,
    mut cache: Option<&mut ResourceCache>,
) -> Result<Tensor, MetalError> {
    let mut x_normed_flat = x_normed_flat;
    let mut ffn_gate = ffn_gate;
    let mut ffn_gate_bias = ffn_gate_bias;
    let mut ffn_up = ffn_up;
    let mut ffn_up_bias = ffn_up_bias;
    let mut ffn_down = ffn_down;
    let mut ffn_down_bias = ffn_down_bias;

    ctx.prepare_tensors_for_active_cmd(&mut [
        &mut x_normed_flat,
        &mut ffn_gate,
        &mut ffn_gate_bias,
        &mut ffn_up,
        &mut ffn_up_bias,
        &mut ffn_down,
        &mut ffn_down_bias,
    ]);
    let d_model = x_normed_flat.dims()[1];

    // gate_proj: [m, d_model] @ weight -> [m, ff_dim]
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
        Some(cache) => ctx.matmul_with_cache(&x_normed_flat, &ffn_gate, false, gate_transpose_b, *cache)?,
        None => ctx.matmul(&x_normed_flat, &ffn_gate, false, gate_transpose_b)?,
    };

    // Add gate bias (broadcast over last dim)
    let gate_out = match cache.as_mut() {
        Some(cache) => ctx.call_with_cache::<BroadcastElemwiseAddOp>((gate_temp, ffn_gate_bias), *cache)?,
        None => ctx.call::<BroadcastElemwiseAddOp>((gate_temp, ffn_gate_bias))?,
    };

    // up_proj: [m, d_model] @ weight -> [m, ff_dim]
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
        Some(cache) => ctx.matmul_with_cache(&x_normed_flat, &ffn_up, false, up_transpose_b, *cache)?,
        None => ctx.matmul(&x_normed_flat, &ffn_up, false, up_transpose_b)?,
    };

    // Add up bias
    let up_out = match cache.as_mut() {
        Some(cache) => ctx.call_with_cache::<BroadcastElemwiseAddOp>((up_temp, ffn_up_bias), *cache)?,
        None => ctx.call::<BroadcastElemwiseAddOp>((up_temp, ffn_up_bias))?,
    };

    // SiLU activation on gate_proj
    let gate_act = match cache.as_mut() {
        Some(cache) => ctx.call_with_cache::<SiluOp>(gate_out, *cache)?,
        None => ctx.call::<SiluOp>(gate_out)?,
    };

    // Element-wise multiplication: SiLU(gate_proj) * up_proj -> [m, ff_dim]
    let hidden = match cache.as_mut() {
        Some(cache) => ctx.call_with_cache::<ElemwiseMulOp>((gate_act, up_out), *cache)?,
        None => ctx.call::<ElemwiseMulOp>((gate_act, up_out))?,
    };

    // down_proj: [m, ff_dim] @ [ff_dim, d_model] -> [m, d_model]
    let hidden_cols = hidden.dims()[1];
    let ffn_down_rows = ffn_down.dims()[0];
    let ffn_down_cols = ffn_down.dims()[1];
    let ffn_temp = if hidden_cols == ffn_down_rows {
        // Hidden [m, ff_dim] @ ffn_down [ff_dim, d_model] -> [m, d_model]
        match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(&hidden, &ffn_down, false, false, *cache)?,
            None => ctx.matmul(&hidden, &ffn_down, false, false)?,
        }
    } else if hidden_cols == ffn_down_cols {
        // Hidden [m, ff_dim] @ ffn_down^T [ff_dim, d_model] where stored as [d_model, ff_dim]
        match cache.as_mut() {
            Some(cache) => ctx.matmul_with_cache(&hidden, &ffn_down, false, true, *cache)?,
            None => ctx.matmul(&hidden, &ffn_down, false, true)?,
        }
    } else {
        return Err(MetalError::DimensionMismatch {
            expected: hidden_cols,
            actual: ffn_down_rows,
        });
    };

    // Add down bias to final projection output
    let ffn_out = match cache.as_mut() {
        Some(cache) => ctx.call_with_cache::<BroadcastElemwiseAddOp>((ffn_temp, ffn_down_bias), *cache)?,
        None => ctx.call::<BroadcastElemwiseAddOp>((ffn_temp, ffn_down_bias))?,
    };

    Ok(ffn_out)
}

// Implement `Operation` for the internal struct.
impl Operation for SwiGLU {
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

mod swiglu_test;
