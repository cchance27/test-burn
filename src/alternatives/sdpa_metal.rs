use objc2::AnyThread;
use objc2::rc::autoreleasepool;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandQueue;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication, MPSMatrixSoftMax,
};
use std::ffi::c_void;
use std::ptr::NonNull;

pub fn scaled_dot_product_attention_metal(
    query_ptr: NonNull<c_void>,
    key_ptr: NonNull<c_void>,
    value_ptr: NonNull<c_void>,
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
) -> Vec<f32> {
    autoreleasepool(|_| {
        let device = MTLCreateSystemDefaultDevice().expect("failed to get default system device");
        let command_queue = device.newCommandQueue().unwrap();

        let d_k = dim as f32;
        let scale = 1.0 / d_k.sqrt();

        let q_buf = unsafe {
            device
                .newBufferWithBytes_length_options(
                    query_ptr,
                    batch * seq_q * dim * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let k_buf = unsafe {
            device
                .newBufferWithBytes_length_options(
                    key_ptr,
                    batch * seq_q * dim * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };
        let v_buf = unsafe {
            device
                .newBufferWithBytes_length_options(
                    value_ptr,
                    batch * seq_q * dim * std::mem::size_of::<f32>(),
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        };

        let attn_byte_length = batch * seq_q * seq_k * std::mem::size_of::<f32>();
        let attn_buf = device
            .newBufferWithLength_options(attn_byte_length, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let out_byte_length = batch * seq_q * dim * std::mem::size_of::<f32>();
        let out_buf = device
            .newBufferWithLength_options(out_byte_length, MTLResourceOptions::StorageModeShared)
            .unwrap();

        let command_buffer = command_queue.commandBuffer().unwrap();

        let qk_gemm = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            &device,
            false,
            true,
            seq_q,
            seq_k,
            dim,
            scale.into(),
            0.0,
        )
        };

        let out_gemm = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
            MPSMatrixMultiplication::alloc(),
            &device,
            false,
            false,
            seq_q,
            dim,
            seq_k,
            1.0,
            0.0,
        )
        };

        let softmax =
            unsafe { MPSMatrixSoftMax::initWithDevice(MPSMatrixSoftMax::alloc(), &device) };

        for i in 0..batch {
            let q_offset = i * seq_q * dim * std::mem::size_of::<f32>();
            let k_offset = i * seq_k * dim * std::mem::size_of::<f32>();
            let v_offset = i * seq_k * dim * std::mem::size_of::<f32>();
            let attn_offset = i * seq_q * seq_k * std::mem::size_of::<f32>();
            let out_offset = i * seq_q * dim * std::mem::size_of::<f32>();

            let q_desc = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    seq_q,
                    dim,
                    dim * std::mem::size_of::<f32>(),
                    MPSDataType::Float32,
                )
            };
            let q_matrix = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &q_buf,
                    q_offset,
                    &q_desc,
                )
            };

            let k_desc = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    seq_k,
                    dim,
                    dim * std::mem::size_of::<f32>(),
                    MPSDataType::Float32,
                )
            };
            let k_matrix = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &k_buf,
                    k_offset,
                    &k_desc,
                )
            };

            let attn_desc = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    seq_q,
                    seq_k,
                    seq_k * std::mem::size_of::<f32>(),
                    MPSDataType::Float32,
                )
            };
            let attn_matrix = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &attn_buf,
                    attn_offset,
                    &attn_desc,
                )
            };

            let v_desc = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    seq_k,
                    dim,
                    dim * std::mem::size_of::<f32>(),
                    MPSDataType::Float32,
                )
            };
            let v_matrix = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &v_buf,
                    v_offset,
                    &v_desc,
                )
            };

            let out_desc = unsafe {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                    seq_q,
                    dim,
                    dim * std::mem::size_of::<f32>(),
                    MPSDataType::Float32,
                )
            };
            let out_matrix = unsafe {
                MPSMatrix::initWithBuffer_offset_descriptor(
                    MPSMatrix::alloc(),
                    &out_buf,
                    out_offset,
                    &out_desc,
                )
            };

            unsafe {
                qk_gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &command_buffer,
                    &q_matrix,
                    &k_matrix,
                    &attn_matrix,
                )
            };
            unsafe {
                softmax.encodeToCommandBuffer_inputMatrix_resultMatrix(
                    &command_buffer,
                    &attn_matrix,
                    &attn_matrix,
                )
            };
            unsafe {
                out_gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                    &command_buffer,
                    &attn_matrix,
                    &v_matrix,
                    &out_matrix,
                )
            };
        }

        command_buffer.commit();
        unsafe {
            command_buffer.waitUntilCompleted();
        }

        // Wrap the output buffer in our Tensor API and copy out via to_vec for consistency
        let out_tensor = crate::metallic::Tensor::from_existing_buffer(
            out_buf.clone(),
            vec![batch, seq_q, dim],
            &device,
            0,
        )
        .expect("failed to wrap out_buf as Tensor");
        out_tensor.to_vec()
    })
}
