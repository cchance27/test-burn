use super::*;
use crate::metallic::{TensorInit, TensorStorage};
use objc2_metal::MTLResource;

pub struct PermuteOp;

struct Permute {
    src: Tensor,
    dst: Tensor,
    permute: Vec<u32>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for PermuteOp {
    type Args<'a> = (Tensor, Vec<u32>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Permute)
    }

    fn new<'a>(
        ctx: &mut Context,
        args: Self::Args<'a>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: std::option::Option<&mut crate::metallic::resource_cache::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor), MetalError> {
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

impl Operation for Permute {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let rank = self.src.dims.len() as u32;
        let num_elements = self.src.len() as u32;

        // Create buffers for the arrays
        // NOTE: We must create actual MTLBuffers for the arrays because set_bytes only works
        // for small scalar values, not arrays. The Metal kernel expects these as constant buffers.
        let src_strides: Vec<u32> = self.src.strides.iter().map(|&x| x as u32).collect();
        let dst_strides: Vec<u32> = self.dst.strides.iter().map(|&x| x as u32).collect();
        let dims: Vec<u32> = self.src.dims.iter().map(|&x| x as u32).collect();

        // Convert to byte slices and create NonNull pointers
        let src_strides_ptr = std::ptr::NonNull::new(src_strides.as_ptr() as *mut std::ffi::c_void).ok_or(MetalError::NullPointer)?;
        let dst_strides_ptr = std::ptr::NonNull::new(dst_strides.as_ptr() as *mut std::ffi::c_void).ok_or(MetalError::NullPointer)?;
        let dims_ptr = std::ptr::NonNull::new(dims.as_ptr() as *mut std::ffi::c_void).ok_or(MetalError::NullPointer)?;
        let permute_ptr = std::ptr::NonNull::new(self.permute.as_ptr() as *mut std::ffi::c_void).ok_or(MetalError::NullPointer)?;

        let src_strides_len = src_strides.len() * std::mem::size_of::<u32>();
        let dst_strides_len = dst_strides.len() * std::mem::size_of::<u32>();
        let dims_len = dims.len() * std::mem::size_of::<u32>();
        let permute_len = self.permute.len() * std::mem::size_of::<u32>();

        // Create temporary buffers
        // DEBT: These buffers are created on-demand for each permute operation and not reused.
        // This could be optimized by using a buffer pool or caching mechanism.
        let src_strides_buf = unsafe {
            self.src.buf.device().newBufferWithBytes_length_options(
                src_strides_ptr,
                src_strides_len,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let dst_strides_buf = unsafe {
            self.src.buf.device().newBufferWithBytes_length_options(
                dst_strides_ptr,
                dst_strides_len,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let dims_buf = unsafe {
            self.src
                .buf
                .device()
                .newBufferWithBytes_length_options(dims_ptr, dims_len, objc2_metal::MTLResourceOptions::StorageModeShared)
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        let permute_buf = unsafe {
            self.src.buf.device().newBufferWithBytes_length_options(
                permute_ptr,
                permute_len,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferFromBytesCreationFailed)?;

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.src.buf, self.src.offset);
        set_buffer(&encoder, 1, &self.dst.buf, self.dst.offset);
        set_buffer(&encoder, 2, &src_strides_buf, 0);
        set_buffer(&encoder, 3, &dst_strides_buf, 0);
        set_buffer(&encoder, 4, &dims_buf, 0);
        set_buffer(&encoder, 5, &permute_buf, 0);
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
