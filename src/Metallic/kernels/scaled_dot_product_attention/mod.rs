use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLComputePipelineState};

use super::{KernelFunction, KernelInvocable};
use crate::metallic::{cache_keys::SdpaKey, resource_cache::ResourceCache, Context, MetalError, Operation, Tensor};

mod scaled_dot_product_attention_test;

// Public, user-facing, zero-sized struct for the SDPA operation.
pub struct ScaledDotProductAttentionOp;

// Internal struct that holds the operation - we'll use existing kernels to implement it
#[allow(dead_code)]
struct ScaledDotProductAttention {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
    pub output: Tensor,
    pub causal: bool,
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub dim: usize,
    pub scale: f32,
    pub query_offset: u32,
}

// Implement `KernelInvocable` for the public struct.
impl KernelInvocable for ScaledDotProductAttentionOp {
    // Input arguments for the call - three input tensors + causal flag
    type Args = (Tensor, Tensor, Tensor, bool, u32); // (q, k, v, causal, query_offset)
                                                     // The output type

    // For composed operations that use other kernels, return None
    fn function_id() -> Option<KernelFunction> {
        None // This is a composed operation using multiple kernels
    }

    // This `new` method is called by `ctx.call()`.
    // For SDPA, we execute the entire computation here since it needs to call other kernels
    fn new(
        ctx: &mut Context,
        args: Self::Args,
        _pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (mut q, mut k, mut v, causal, query_offset) = args;

        ctx.prepare_tensors_for_active_cmd(&mut [&mut q, &mut k, &mut v]);

        // Validate dimensions
        if q.dims().len() != 3 || k.dims().len() != 3 || v.dims().len() != 3 {
            return Err(MetalError::InvalidShape("SDPA requires 3D tensors".to_string()));
        }

        let b = q.dims()[0];
        let s_q = q.dims()[1];
        let s_k = k.dims()[1];
        let d = q.dims()[2];

        // Check batch dimension compatibility
        if b != k.dims()[0] || b != v.dims()[0] {
            return Err(MetalError::DimensionMismatch {
                expected: b,
                actual: k.dims()[0].max(v.dims()[0]),
            });
        }

        // Check feature dimension compatibility
        if d != k.dims()[2] {
            return Err(MetalError::DimensionMismatch {
                expected: d,
                actual: k.dims()[2],
            });
        }

        // Check value tensor compatibility
        if s_k != v.dims()[1] || d != v.dims()[2] {
            return Err(MetalError::DimensionMismatch {
                expected: s_k * d,
                actual: v.dims()[1] * v.dims()[2],
            });
        }

        // Calculate scale factor
        let scale = 1.0 / (d as f32).sqrt();

        // Create output tensor
        let out = Tensor::create_tensor_pooled(vec![b, s_q, d], ctx)?;

        // Process each batch separately to work with 2D matmul operations
        for i in 0..b {
            // Get batch slices for each tensor
            let q_i = q.get_batch(i)?; // [s_q, d]
            let k_i = k.get_batch(i)?; // [s_k, d]
            let v_i = v.get_batch(i)?; // [s_k, d]
            let mut out_i = out.get_batch(i)?; // [s_q, d]

            // Q x K^T -> attn (for this batch)
            // Note: k.get_batch(i) gives [s_k, d], need to transpose to [d, s_k]
            let k_i_t = k_i.permute(&[1, 0], ctx)?; // [d, s_k]

            // Perform matmul: [s_q, d] @ [d, s_k] = [s_q, s_k]
            let qk_result = ctx.matmul(&q_i, &k_i_t, false, false)?; // [s_q, s_k]

            // Apply scale using element-wise multiplication to the attention matrix
            // Create a scale tensor with the same shape as attention [s_q, s_k]
            let scale_values = vec![scale; s_q * s_k]; // Fill with scale value for each element
            let scale_tensor = Tensor::create_tensor_from_slice(&scale_values, vec![s_q, s_k], ctx)?;
            let scaled_attn = ctx.call::<crate::metallic::kernels::elemwise_mul::ElemwiseMulOp>((qk_result, scale_tensor))?;

            // Apply softmax to the scaled attention
            let softmax_result = ctx.call::<crate::metallic::kernels::softmax::SoftmaxOp>((
                scaled_attn,
                s_q as u32,
                s_k as u32,
                causal as u32,
                query_offset,
            ))?;

            // attn x V -> out (for this batch)
            // [s_q, s_k] @ [s_k, d] = [s_q, d]
            let final_result = ctx.matmul(&softmax_result, &v_i, false, false)?; // [s_q, d]

            // Copy final result to output tensor
            let final_slice = final_result.as_slice();
            let out_slice = out_i.as_mut_slice();
            out_slice.copy_from_slice(final_slice);
        }

        // Create a dummy operation since all work is done in this function
        Ok((
            Box::new(ScaledDotProductAttention {
                q: q.clone(), // This is just to satisfy struct fields, actual work is done
                k: k.clone(),
                v: v.clone(),
                output: out.clone(),
                causal,
                batch: b,
                seq_q: s_q,
                seq_k: s_k,
                dim: d,
                scale,
                query_offset,
            }),
            out,
        ))
    }
}

// The implementation in the `new` method above executes the SDPA logic directly
// However, the proper way to implement this is to execute it in the `new` method
// since the operation needs to call other kernels using the Context.

// Implement `Operation` for the internal struct.
impl Operation for ScaledDotProductAttention {
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
