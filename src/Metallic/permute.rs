use super::{Context, MetalError, Operation, Tensor, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder as _, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice as _, MTLLibrary as _, MTLResource, MTLSize,
};

use crate::metallic::encoder::{
    dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state,
};

/// DEBT: The permute kernel creates temporary buffers for each operation which could be inefficient.
/// These buffers are created on-demand and not reused, which may lead to memory allocation overhead.
/// In the future, we should consider using a buffer pool or caching mechanism for these temporary arrays.
/// Ensure the permute compute pipeline is compiled and cached on the Context.
pub fn ensure_permute_pipeline(ctx: &mut Context) -> Result<(), MetalError> {
    if ctx.permute_pipeline.is_some() {
        return Ok(());
    }

    let source = r#"
    using namespace metal;

    // Kernel for permuting tensor dimensions.
    // This kernel takes arrays of strides and permutation indices as constant buffers.
    // NOTE: The arrays (src_strides, dst_strides, dims, permute) must be passed as proper MTLBuffers,
    // not as inline bytes, because set_bytes only works for small scalar values, not arrays.
    kernel void permute_kernel(device const float* src [[buffer(0)]],
                               device float* dst [[buffer(1)]],
                               constant const uint* src_strides [[buffer(2)]],
                               constant const uint* dst_strides [[buffer(3)]],
                               constant const uint* dims [[buffer(4)]],
                               constant const uint* permute [[buffer(5)]],
                               constant const uint& rank [[buffer(6)]],
                               constant const uint& num_elements [[buffer(7)]],
                               uint gid [[thread_position_in_grid]]) {
        if (gid >= num_elements) return;

        uint src_idx = gid;
        uint temp_idx = src_idx;

        uint src_coords[8];
        for (uint i = 0; i < rank; ++i) {
            src_coords[i] = temp_idx / src_strides[i];
            temp_idx %= src_strides[i];
        }

        uint dst_coords[8];
        for (uint i = 0; i < rank; ++i) {
            dst_coords[i] = src_coords[permute[i]];
        }

        uint dst_idx = 0;
        for (uint i = 0; i < rank; ++i) {
            dst_idx += dst_coords[i] * dst_strides[i];
        }

        dst[dst_idx] = src[src_idx];
    }
    "#;

    let source_ns = NSString::from_str(source);
    let library = ctx
        .device
        .newLibraryWithSource_options_error(&source_ns, None)
        .map_err(|err| MetalError::LibraryCompilationFailed(err.to_string()))?;

    let fn_name = NSString::from_str("permute_kernel");
    let function = library
        .newFunctionWithName(&fn_name)
        .ok_or_else(|| MetalError::FunctionCreationFailed(fn_name.to_string()))?;
    let pipeline = ctx
        .device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|_err| MetalError::PipelineCreationFailed)?;

    ctx.permute_pipeline = Some(pipeline);
    Ok(())
}

/// An operation that runs permute over an input tensor.
pub struct Permute {
    pub src: Tensor,
    pub dst: Tensor,
    pub permute: Vec<u32>,
    pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Permute {
    pub fn new(
        src: Tensor,
        dst: Tensor,
        permute: Vec<u32>,
        pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    ) -> Result<Self, MetalError> {
        if src.len() != dst.len() {
            return Err(MetalError::InvalidShape(format!(
                "Permute: input and output tensors must have the same length, got src={}, dst={}",
                src.len(),
                dst.len()
            )));
        }

        Ok(Self {
            src,
            dst,
            permute,
            pipeline,
        })
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
        let src_strides_ptr = std::ptr::NonNull::new(src_strides.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MetalError::NullPointer)?;
        let dst_strides_ptr = std::ptr::NonNull::new(dst_strides.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MetalError::NullPointer)?;
        let dims_ptr = std::ptr::NonNull::new(dims.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MetalError::NullPointer)?;
        let permute_ptr = std::ptr::NonNull::new(self.permute.as_ptr() as *mut std::ffi::c_void)
            .ok_or(MetalError::NullPointer)?;

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
            self.src.buf.device().newBufferWithBytes_length_options(
                dims_ptr,
                dims_len,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
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
