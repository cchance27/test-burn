use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLResourceOptions};

use super::*;
use crate::{CommandBuffer, TensorElement, TensorInit, TensorStorage, operation::{ComputeKernelEncoder}, context::GpuProfilerLabel};

pub struct PermuteOp;

struct Permute<T: TensorElement> {
    src: Tensor<T>,
    dst: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
    src_strides_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dst_strides_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    dims_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    permute_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
}

impl DefaultKernelInvocable for PermuteOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Vec<u32>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Permute)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::caching::ResourceCache>,
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

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("permute_op"));

        // Create buffers for strides, dims, and permute indices
        let src_strides: Vec<u32> = src.strides.iter().map(|&x| x as u32).collect();
        let dst_strides: Vec<u32> = dst.strides.iter().map(|&x| x as u32).collect();
        let dims: Vec<u32> = src.dims.iter().map(|&x| x as u32).collect();

        let src_strides_buf = unsafe {
            src.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(src_strides.as_ptr() as *mut u32).unwrap().cast(),
                src_strides.len() * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let dst_strides_buf = unsafe {
            src.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(dst_strides.as_ptr() as *mut u32).unwrap().cast(),
                dst_strides.len() * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let dims_buf = unsafe {
            src.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(dims.as_ptr() as *mut u32).unwrap().cast(),
                dims.len() * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let permute_buf = unsafe {
            src.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(permute.as_ptr() as *mut u32).unwrap().cast(),
                permute.len() * std::mem::size_of::<u32>(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        // Create the internal operation struct
        let op = Permute {
            src,
            dst: dst.clone(),
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
            src_strides_buf,
            dst_strides_buf,
            dims_buf,
            permute_buf,
        };

        // Return the boxed operation and the output tensor
        Ok((Box::new(op), dst))
    }
}

impl<T: TensorElement> Operation for Permute<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let _rank = self.src.dims.len() as u32;
        let num_elements = self.src.len() as u32;

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

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(groups, threads_per_tg);

        Ok(())
    }

    fn bind_to_encoder(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        set_buffer(encoder, 0, &self.src.buf, self.src.offset);
        set_buffer(encoder, 1, &self.dst.buf, self.dst.offset);
        set_buffer(encoder, 2, &self.src_strides_buf, 0);
        set_buffer(encoder, 3, &self.dst_strides_buf, 0);
        set_buffer(encoder, 4, &self.dims_buf, 0);
        set_buffer(encoder, 5, &self.permute_buf, 0);
        set_bytes(encoder, 6, &(self.src.dims.len() as u32));
        set_bytes(encoder, 7, &(self.src.len() as u32));
    }
}

#[cfg(test)]
mod permute_test;
