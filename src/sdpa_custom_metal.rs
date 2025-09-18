use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder as _, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder as _, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;

pub fn scaled_dot_product_attention_custom_metal(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
) -> Vec<f32> {
    objc2::rc::autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice().expect("failed to get default system device");

        let command_queue = device.newCommandQueue().unwrap();

        let q_buf = unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(query.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(query),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let k_buf = unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(key.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(key),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let v_buf = unsafe {
            device
                .newBufferWithBytes_length_options(
                    NonNull::new(value.as_ptr() as *mut c_void).unwrap(),
                    std::mem::size_of_val(value),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let out_byte_length = batch * seq_q * dim * std::mem::size_of::<f32>();
        let out_buf = device
            .newBufferWithLength_options(out_byte_length, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let source = NSString::from_str(include_str!("sdpa.metal"));
        let options = MTLCompileOptions::new();
        let library = device
            .newLibraryWithSource_options_error(&source, Some(&options))
            .unwrap();

        let function_name = NSString::from_str("sdpa_kernel");
        let function = library.newFunctionWithName(&function_name).unwrap();

        let pipeline_state = device
            .newComputePipelineStateWithFunction_error(&function)
            .unwrap();

        let command_buffer = command_queue.commandBuffer().unwrap();

        let compute_encoder = command_buffer.computeCommandEncoder().unwrap();

        compute_encoder.setComputePipelineState(&pipeline_state);
        unsafe {
            compute_encoder.setBuffer_offset_atIndex(Some(&q_buf), 0, 0);
            compute_encoder.setBuffer_offset_atIndex(Some(&k_buf), 0, 1);
            compute_encoder.setBuffer_offset_atIndex(Some(&v_buf), 0, 2);
            compute_encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 3);
        }

        let mut batch_u32 = batch as u32;
        let mut seq_q_u32 = seq_q as u32;
        let mut seq_k_u32 = seq_k as u32;
        let mut dim_u32 = dim as u32;

        unsafe {
            compute_encoder.setBytes_length_atIndex(
                NonNull::new(&mut batch_u32 as *mut _ as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                4,
            );
            compute_encoder.setBytes_length_atIndex(
                NonNull::new(&mut seq_q_u32 as *mut _ as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                5,
            );
            compute_encoder.setBytes_length_atIndex(
                NonNull::new(&mut seq_k_u32 as *mut _ as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                6,
            );
            compute_encoder.setBytes_length_atIndex(
                NonNull::new(&mut dim_u32 as *mut _ as *mut c_void).unwrap(),
                std::mem::size_of::<u32>(),
                7,
            );
        }

        let threadgroups_per_grid = MTLSize {
            width: seq_q,
            height: batch,
            depth: 1,
        };

        let threads_per_threadgroup = MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        };

        compute_encoder.dispatchThreadgroups_threadsPerThreadgroup(
            threadgroups_per_grid,
            threads_per_threadgroup,
        );

        compute_encoder.endEncoding();

        command_buffer.commit();

        unsafe { command_buffer.waitUntilCompleted() };

        let mut out_data = vec![0.0f32; batch * seq_q * dim];
        let ptr = out_buf.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, out_data.as_mut_ptr(), out_data.len());
        }

        out_data
    })
}
