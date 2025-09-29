use super::*;
use crate::metallic::{TensorInit, TensorStorage};

/// Public, user-facing, zero-sized struct for the KV rearrange operation.
pub struct KvRearrangeOp;

/// Internal struct that holds data for the Operation trait.
struct KvRearrange {
    input: Tensor,  // [M, kv_dim]
    output: Tensor, // [batch*n_heads, seq, head_dim]
    kv_dim: u32,
    kv_head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for KvRearrangeOp {
    type Args<'a> = (Tensor, u32, u32, u32, u32, u32, u32); // (input, kv_dim, kv_head_dim, n_heads, n_kv_heads, head_dim, seq)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::KvRearrange)
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
        let (input, kv_dim, kv_head_dim, n_heads, n_kv_heads, head_dim, seq) = args;

        // Calculate output dimensions: [batch*n_heads, seq, head_dim]
        // We need to infer batch from the input dimensions: M = batch*seq
        let input_m = input.dims()[0];
        let batch = input_m / seq as usize;
        let output_dims = vec![batch * n_heads as usize, seq as usize, head_dim as usize];

        ctx.prepare_tensors_for_active_cmd(&[&input])?;

        let output = Tensor::new(output_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let op = KvRearrange {
            input,
            output: output.clone(),
            kv_dim,
            kv_head_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            seq,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), output))
    }
}

impl Operation for KvRearrange {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total_elements = self.output.len() as u32;
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
        set_bytes(&encoder, 2, &self.kv_dim);
        set_bytes(&encoder, 3, &self.kv_head_dim);
        set_bytes(&encoder, 4, &self.n_heads);
        set_bytes(&encoder, 5, &self.n_kv_heads);
        set_bytes(&encoder, 6, &self.head_dim);
        set_bytes(&encoder, 7, &self.seq);
        set_bytes(&encoder, 8, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod kv_rearrange_test {
    use super::*;

    #[test]
    fn test_kv_rearrange_logic() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        // Create a simple test tensor [batch*seq, kv_dim] = [2*3, 4] = [6, 4]
        let input_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = Tensor::new(vec![6, 4], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&input_data))?;

        // Test parameters: kv_dim=4, kv_head_dim=2, n_heads=2, n_kv_heads=1, head_dim=2, seq=3
        let result = ctx.call::<KvRearrangeOp>((input, 4, 2, 2, 1, 2, 3))?;

        // Verify dimensions: [batch*n_heads, seq, head_dim] = [2*2, 3, 2] = [4, 3, 2]
        assert_eq!(result.dims(), &[4, 3, 2]);
        Ok(())
    }
}
