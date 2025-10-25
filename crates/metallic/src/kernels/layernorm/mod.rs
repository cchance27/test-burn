use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage};

/// Public, user-facing, zero-sized struct for the LayerNorm operation.
pub struct LayerNormOp;

/// Internal struct that holds data for the Operation trait.
struct LayerNorm<T: TensorElement> {
    input: Tensor<T>,
    output: Tensor<T>,
    gamma: Tensor<T>,
    beta: Tensor<T>,
    feature_dim: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl DefaultKernelInvocable for LayerNormOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>, Tensor<T>, u32); // (input, gamma, beta, feature_dim)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::LayerNorm)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, gamma, beta, feature_dim) = args;

        // Validate dimensions
        if input.dims().last() != Some(&(feature_dim as usize)) {
            return Err(MetalError::InvalidShape(format!(
                "Input feature dimension {} does not match specified feature_dim {}",
                input.dims().last().unwrap_or(&0),
                feature_dim
            )));
        }
        if gamma.dims() != [feature_dim as usize] {
            return Err(MetalError::InvalidShape(format!(
                "Gamma shape {:?} does not match feature_dim {}",
                gamma.dims(),
                feature_dim
            )));
        }
        if beta.dims() != [feature_dim as usize] {
            return Err(MetalError::InvalidShape(format!(
                "Beta shape {:?} does not match feature_dim {}",
                beta.dims(),
                feature_dim
            )));
        }

        ctx.prepare_tensors_for_active_cmd(&[&input, &gamma, &beta])?;

        let output = Tensor::new(input.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let op = LayerNorm {
            input,
            output: output.clone(),
            gamma,
            beta,
            feature_dim,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl<T: TensorElement> Operation for LayerNorm<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let total_elements = self.input.len() as u32;
        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.input.buf, self.input.offset);
        set_buffer(&encoder, 1, &self.output.buf, self.output.offset);
        set_buffer(&encoder, 2, &self.gamma.buf, self.gamma.offset);
        set_buffer(&encoder, 3, &self.beta.buf, self.beta.offset);
        set_bytes(&encoder, 4, &self.feature_dim);
        set_bytes(&encoder, 5, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        Ok(())
    }
}

#[cfg(test)]
mod layernorm_test {
    use super::*;
    use crate::F32Element;

    #[test]
    fn test_layernorm_logic() -> Result<(), MetalError> {
        let mut ctx = Context::<F32Element>::new()?;
        // Create test data: [2, 3] tensor, so feature_dim=3
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(vec![2, 3], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data))?;

        // Create gamma and beta with all ones and zeros for simple test
        let gamma_data = vec![1.0, 1.0, 1.0];
        let gamma = Tensor::new(vec![3], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&gamma_data))?;
        let beta_data = vec![0.0, 0.0, 0.0];
        let beta = Tensor::new(vec![3], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&beta_data))?;

        let result = ctx.call::<LayerNormOp>((input, gamma, beta, 3))?;

        // The result should have mean 0 and variance 1 for each row after normalization
        let result_slice = result.as_slice();

        // Check first row (indices 0-2)
        let row1: &[f32] = &result_slice[0..3];
        let row1_mean: f32 = row1.iter().sum::<f32>() / 3.0;
        let row1_var: f32 = row1.iter().map(|x| (x - row1_mean).powi(2)).sum::<f32>() / 3.0;

        assert!(row1_mean.abs() < 1e-4, "First row mean should be near 0, got {}", row1_mean);
        assert!(
            (row1_var - 1.0).abs() < 1e-4,
            "First row variance should be near 1, got {}",
            row1_var
        );

        // Check second row (indices 3-5)
        let row2: &[f32] = &result_slice[3..6];
        let row2_mean: f32 = row2.iter().sum::<f32>() / 3.0;
        let row2_var: f32 = row2.iter().map(|x| (x - row2_mean).powi(2)).sum::<f32>() / 3.0;

        assert!(row2_mean.abs() < 1e-4, "Second row mean should be near 0, got {}", row2_mean);
        assert!(
            (row2_var - 1.0).abs() < 1e-4,
            "Second row variance should be near 1, got {}",
            row2_var
        );

        Ok(())
    }
}
