use super::*;
use crate::metallic::{TensorElement, TensorInit, TensorStorage};

pub struct FusedRmsNormQkvProjectionOp;

struct FusedRmsNormQkvProjection<T: TensorElement> {
    input: Tensor<T>,
    gamma: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    output: Tensor<T>,
    feature_dim: u32,
    total_out_dim: u32,
    rows: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for FusedRmsNormQkvProjectionOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>, u32, u32);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedRmsNormQkvProjection)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, gamma, weight, bias, feature_dim, total_out_dim) = args;

        let input_dims = input.dims();
        if input_dims.len() != 2 {
            return Err(MetalError::InvalidShape(format!(
                "Fused RMSNorm+QKV expects 2D input [rows, feature_dim], got {:?}",
                input_dims
            )));
        }

        if input_dims[1] != feature_dim as usize {
            return Err(MetalError::InvalidShape(format!(
                "Input feature dim {} does not match provided feature_dim {}",
                input_dims[1], feature_dim
            )));
        }

        if gamma.dims() != [feature_dim as usize] {
            return Err(MetalError::InvalidShape(format!(
                "Gamma dims {:?} do not match feature_dim {}",
                gamma.dims(),
                feature_dim
            )));
        }

        let weight_dims = weight.dims();
        if weight_dims.len() != 2 || weight_dims[0] != feature_dim as usize || weight_dims[1] != total_out_dim as usize {
            return Err(MetalError::InvalidShape(format!(
                "Weight dims {:?} incompatible with feature_dim {} and total_out_dim {}",
                weight_dims, feature_dim, total_out_dim
            )));
        }

        if bias.dims() != [total_out_dim as usize] {
            return Err(MetalError::InvalidShape(format!(
                "Bias dims {:?} do not match total_out_dim {}",
                bias.dims(),
                total_out_dim
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[&input, &gamma, &weight, &bias])?;

        let output_dims = vec![input_dims[0], total_out_dim as usize];
        let output = Tensor::new(output_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let op = FusedRmsNormQkvProjection {
            input,
            gamma,
            weight,
            bias,
            output: output.clone(),
            feature_dim,
            total_out_dim,
            rows: input_dims[0] as u32,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for FusedRmsNormQkvProjection<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: (self.rows as usize).div_ceil(threads_per_tg.width),
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(&encoder, 2, &self.gamma.buf, self.gamma.offset);
        set_buffer(&encoder, 3, &self.weight.buf, self.weight.offset);
        set_buffer(&encoder, 4, &self.bias.buf, self.bias.offset);
        set_bytes(&encoder, 5, &self.feature_dim);
        set_bytes(&encoder, 6, &self.total_out_dim);
        set_bytes(&encoder, 7, &self.rows);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();

        Ok(())
    }
}

#[cfg(test)]
mod fused_rmsnorm_qkv_test;
