use super::*;
use crate::metallic::{
    TensorElement, TensorInit, TensorStorage,
    resource_cache::{PermuteConstantKind, ResourceCache},
};

pub struct PermuteOp;

struct Permute<T: TensorElement> {
    src: Tensor<T>,
    dst: Tensor<T>,
    permute: Vec<u32>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for PermuteOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Vec<u32>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Permute)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (src, permute) = args;

        // Validate permutation
        if permute.len() != src.dims().len() {
            return Err(MetalError::InvalidShape(format!(
                "Permute: permutation length {} doesn't match tensor rank {}",
                permute.len(),
                src.dims().len()
            )));
        }

        // Calculate output dimensions based on permutation
        let mut out_dims = vec![0; src.dims().len()];
        for (i, &p_idx) in permute.iter().enumerate() {
            if p_idx as usize >= src.dims().len() {
                return Err(MetalError::InvalidShape(format!(
                    "Permute: permutation index {} out of bounds for tensor rank {}",
                    p_idx,
                    src.dims().len()
                )));
            }
            out_dims[i] = src.dims()[p_idx as usize];
        }

        ctx.prepare_tensors_for_active_cmd(&[&src])?;

        // Create the output tensor
        let dst = Tensor::new(out_dims, TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        // Validate that input and output tensors have the same number of elements
        if src.len() != dst.len() {
            return Err(MetalError::InvalidShape(format!(
                "Permute: input and output tensors must have the same length, got src={}, dst={}",
                src.len(),
                dst.len()
            )));
        }

        // Create the internal operation struct
        let op = Permute {
            src,
            dst: dst.clone(),
            permute,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        // Return the boxed operation and the output tensor
        Ok((Box::new(op), dst))
    }
}

impl<T: TensorElement> Operation for Permute<T> {
    fn encode(&self, command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>, cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let rank = self.src.dims.len() as u32;
        let num_elements = self.src.len() as u32;

        let src_strides: Vec<u32> = self.src.strides.iter().map(|&x| x as u32).collect();
        let dst_strides: Vec<u32> = self.dst.strides.iter().map(|&x| x as u32).collect();
        let dims: Vec<u32> = self.src.dims.iter().map(|&x| x as u32).collect();

        let src_strides_len = src_strides.len() * std::mem::size_of::<u32>();
        let dst_strides_len = dst_strides.len() * std::mem::size_of::<u32>();
        let dims_len = dims.len() * std::mem::size_of::<u32>();
        let permute_len = self.permute.len() * std::mem::size_of::<u32>();

        const INLINE_LIMIT: usize = 4 * 1024;
        let device = self.src.buf.device();
        let mut retained_buffers: Vec<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>> = Vec::new();

        let mut bind_slice = |index: usize, data: &[u32], length: usize, kind: PermuteConstantKind| -> Result<(), MetalError> {
            if length <= INLINE_LIMIT {
                set_bytes_slice(&encoder, index, data);
                Ok(())
            } else {
                let buffer = cache.get_or_create_permute_constant_buffer(&device, kind, length)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, buffer.contents().as_ptr() as *mut u8, length);
                }
                set_buffer(&encoder, index, &buffer, 0);
                retained_buffers.push(buffer);
                Ok(())
            }
        };

        bind_slice(2, &src_strides, src_strides_len, PermuteConstantKind::SrcStrides)?;
        bind_slice(3, &dst_strides, dst_strides_len, PermuteConstantKind::DstStrides)?;
        bind_slice(4, &dims, dims_len, PermuteConstantKind::Dims)?;
        bind_slice(5, &self.permute, permute_len, PermuteConstantKind::Permutation)?;

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.src.buf, self.src.offset);
        set_buffer(&encoder, 1, &self.dst.buf, self.dst.offset);
        set_bytes(&encoder, 6, &rank);
        set_bytes(&encoder, 7, &num_elements);

        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: num_elements.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}

#[cfg(test)]
mod permute_test;
