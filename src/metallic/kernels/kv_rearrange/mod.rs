use super::*;
use crate::metallic::{TensorElement, TensorInit, TensorStorage};
use std::convert::TryFrom;

/// Public, user-facing, zero-sized struct for the KV rearrange operation.
pub struct KvRearrangeOp;

/// Internal struct that holds data for the Operation trait.
struct KvRearrange<T: TensorElement> {
    input: Tensor<T>,  // [M, kv_dim]
    output: Tensor<T>, // [batch*n_heads, seq, head_dim]
    kv_dim: u32,
    row_stride: u32,
    kv_head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for KvRearrangeOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, u32, u32, u32, u32, u32, u32); // (input, kv_dim, kv_head_dim, n_heads, n_kv_heads, head_dim, seq)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::KvRearrange)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, kv_dim, kv_head_dim, n_heads, n_kv_heads, head_dim, seq) = args;

        if input.dims().len() < 2 {
            return Err(MetalError::InvalidShape("KV rearrange expects at least 2D input".to_string()));
        }

        let row_stride_elems = input.strides.first().copied().unwrap_or_else(|| input.dims()[1]);
        if row_stride_elems == 0 {
            return Err(MetalError::InvalidShape(
                "Row stride for KV rearrange must be greater than zero".to_string(),
            ));
        }
        let row_stride =
            u32::try_from(row_stride_elems).map_err(|_| MetalError::InvalidShape("Row stride for KV rearrange exceeds u32".to_string()))?;

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
            row_stride,
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

impl<T: TensorElement> Operation for KvRearrange<T> {
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
        set_bytes(&encoder, 3, &self.row_stride);
        set_bytes(&encoder, 4, &self.kv_head_dim);
        set_bytes(&encoder, 5, &self.n_heads);
        set_bytes(&encoder, 6, &self.n_kv_heads);
        set_bytes(&encoder, 7, &self.head_dim);
        set_bytes(&encoder, 8, &self.seq);
        set_bytes(&encoder, 9, &total_elements);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
fn cpu_reference_rearrange(
    fused: &[f32],
    row_stride: usize,
    batch: usize,
    seq: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    kv_head_dim: usize,
    column_offset: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; batch * n_heads * seq * head_dim];
    let group_size = n_heads / n_kv_heads;
    for b in 0..batch {
        for h in 0..n_heads {
            let kv_h = h / group_size;
            for s in 0..seq {
                let src_row = b * seq + s;
                for d in 0..head_dim {
                    let base_offset = kv_h * kv_head_dim + d;
                    let src_index = src_row * row_stride + column_offset + base_offset;
                    let dst_index = ((b * n_heads + h) * seq + s) * head_dim + d;
                    output[dst_index] = fused[src_index];
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod kv_rearrange_test {
    use crate::metallic::F32Element;

    use super::*;

    #[test]
    fn test_kv_rearrange_logic() -> Result<(), MetalError> {
        let mut ctx = Context::<F32Element>::new()?;
        let batch = 2usize;
        let seq = 3usize;
        let n_heads = 4usize;
        let n_kv_heads = 2usize;
        let head_dim = 2usize;
        let kv_head_dim = 3usize;
        let d_model = n_heads * head_dim;
        let kv_dim = n_kv_heads * kv_head_dim;
        let fused_dim = d_model + 2 * kv_dim;
        let rows = batch * seq;

        // Create a fused QKV tensor so slicing produces strided views.
        let fused_data: Vec<f32> = (0..rows * fused_dim).map(|i| i as f32).collect();
        let fused = Tensor::new(
            vec![rows, fused_dim],
            TensorStorage::Dedicated(&ctx),
            TensorInit::CopyFrom(&fused_data),
        )?;

        let q_mat = fused.slice_last_dim(0..d_model)?;
        let k_mat = fused.slice_last_dim(d_model..d_model + kv_dim)?;
        let v_mat = fused.slice_last_dim(d_model + kv_dim..fused_dim)?;

        let q_result = ctx.call::<KvRearrangeOp>((
            q_mat.clone(),
            d_model as u32,
            head_dim as u32,
            n_heads as u32,
            n_heads as u32,
            head_dim as u32,
            seq as u32,
        ))?;
        let k_result = ctx.call::<KvRearrangeOp>((
            k_mat.clone(),
            kv_dim as u32,
            kv_head_dim as u32,
            n_kv_heads as u32,
            n_kv_heads as u32,
            kv_head_dim as u32,
            seq as u32,
        ))?;
        let v_result = ctx.call::<KvRearrangeOp>((
            v_mat.clone(),
            kv_dim as u32,
            kv_head_dim as u32,
            n_kv_heads as u32,
            n_kv_heads as u32,
            kv_head_dim as u32,
            seq as u32,
        ))?;

        ctx.synchronize();

        // Verify output dimensions for each projection.
        assert_eq!(q_result.dims(), &[batch * n_heads, seq, head_dim]);
        assert_eq!(k_result.dims(), &[batch * n_kv_heads, seq, kv_head_dim]);
        assert_eq!(v_result.dims(), &[batch * n_kv_heads, seq, kv_head_dim]);

        // CPU reference for verification using the original fused buffer.
        let row_stride = fused_dim;
        let q_expected = cpu_reference_rearrange(&fused_data, row_stride, batch, seq, n_heads, n_heads, head_dim, head_dim, 0);
        let k_expected = cpu_reference_rearrange(
            &fused_data,
            row_stride,
            batch,
            seq,
            n_kv_heads,
            n_kv_heads,
            kv_head_dim,
            kv_head_dim,
            d_model,
        );
        let v_expected = cpu_reference_rearrange(
            &fused_data,
            row_stride,
            batch,
            seq,
            n_kv_heads,
            n_kv_heads,
            kv_head_dim,
            kv_head_dim,
            d_model + kv_dim,
        );

        assert_eq!(q_result.as_slice(), q_expected.as_slice());
        assert_eq!(k_result.as_slice(), k_expected.as_slice());
        assert_eq!(v_result.as_slice(), v_expected.as_slice());

        Ok(())
    }
}
