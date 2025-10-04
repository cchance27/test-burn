use super::*;
use crate::metallic::{TensorInit, TensorStorage};
use metallic_macros::metal_kernel;

#[cfg(test)]
mod elemwise_add_test;
#[cfg(test)]
mod elemwise_broadcast_add_test;

metal_kernel! {
    library ElemwiseAdd {
        source: include_str!("kernel.metal"),
        functions: {
            ElemwiseAdd => {
                F32 => "add_kernel_f32",
                F16 => "add_kernel_f16",
            },
            ElemwiseBroadcastAdd => {
                F32 => "broadcast_add_kernel_f32",
                F16 => "broadcast_add_kernel_f16",
            },
        },
        operations: {
            ElemwiseAddOp => {
                function: Some(KernelFunction::ElemwiseAdd),
                args: (Tensor<T>, Tensor<T>),
                pipeline: required,
                state: {
                    a: Tensor<T>,
                    b: Tensor<T>,
                    out: Tensor<T>,
                    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
                },
                new: |ctx, (a, b), pipeline, _cache| {
                    if a.dims() != b.dims() {
                        return Err(MetalError::InvalidShape(format!(
                            "ElemwiseAdd: input shapes must match, got a={:?}, b={:?}",
                            a.dims(),
                            b.dims(),
                        )));
                    }
                    ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;
                    let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
                    Ok((ElemwiseAddOpState { a, b, out: out.clone(), pipeline }, out))
                },
                encode: |command_buffer, _cache, state| {
                    let encoder = command_buffer
                        .computeCommandEncoder()
                        .ok_or(MetalError::ComputeEncoderCreationFailed)?;

                    let total_elements = state.a.len() as u32;
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

                    set_compute_pipeline_state(&encoder, &state.pipeline);
                    set_buffer(&encoder, 0, &state.a.buf, state.a.offset);
                    set_buffer(&encoder, 1, &state.b.buf, state.b.offset);
                    set_buffer(&encoder, 2, &state.out.buf, state.out.offset);
                    set_bytes(&encoder, 3, &total_elements);

                    dispatch_threadgroups(&encoder, groups, threads_per_tg);
                    encoder.endEncoding();
                    Ok(())
                },
            },
            BroadcastElemwiseAddOp => {
                function: Some(KernelFunction::ElemwiseBroadcastAdd),
                args: (Tensor<T>, Tensor<T>),
                pipeline: required,
                state: {
                    a: Tensor<T>,
                    b: Tensor<T>,
                    out: Tensor<T>,
                    b_len: usize,
                    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
                },
                new: |ctx, (a, b), pipeline, _cache| {
                    let b_len = b.len();
                    if b_len == 0 {
                        return Err(MetalError::InvalidShape("Broadcast b cannot be empty".to_string()));
                    }
                    if b.dims().len() != 1 {
                        return Err(MetalError::InvalidShape(format!("Broadcast b must be 1D, got {:?}", b.dims())));
                    }

                    ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;

                    let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
                    Ok((BroadcastElemwiseAddOpState {
                        a,
                        b,
                        out: out.clone(),
                        b_len,
                        pipeline,
                    }, out))
                },
                encode: |command_buffer, _cache, state| {
                    let encoder = command_buffer
                        .computeCommandEncoder()
                        .ok_or(MetalError::ComputeEncoderCreationFailed)?;

                    let total_elements = state.a.len() as u32;
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

                    set_compute_pipeline_state(&encoder, &state.pipeline);
                    set_buffer(&encoder, 0, &state.a.buf, state.a.offset);
                    set_buffer(&encoder, 1, &state.b.buf, state.b.offset);
                    set_buffer(&encoder, 2, &state.out.buf, state.out.offset);
                    set_bytes(&encoder, 3, &total_elements);
                    set_bytes(&encoder, 4, &(state.b_len as u32));

                    dispatch_threadgroups(&encoder, groups, threads_per_tg);
                    encoder.endEncoding();
                    Ok(())
                },
            },
            BroadcastElemwiseAddInplaceOp => {
                function: Some(KernelFunction::ElemwiseBroadcastAdd),
                args: (Tensor<T>, Tensor<T>),
                pipeline: required,
                state: {
                    a: Tensor<T>,
                    b: Tensor<T>,
                    b_len: usize,
                    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
                },
                new: |ctx, (a, b), pipeline, _cache| {
                    let b_len = b.len();
                    if b_len == 0 {
                        return Err(MetalError::InvalidShape("Broadcast b cannot be empty".to_string()));
                    }
                    if b.dims().len() != 1 {
                        return Err(MetalError::InvalidShape(format!("Broadcast b must be 1D, got {:?}", b.dims())));
                    }

                    ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;

                    let out = a.clone();
                    Ok((BroadcastElemwiseAddInplaceOpState {
                        a,
                        b,
                        b_len,
                        pipeline,
                    }, out))
                },
                encode: |command_buffer, _cache, state| {
                    let encoder = command_buffer
                        .computeCommandEncoder()
                        .ok_or(MetalError::ComputeEncoderCreationFailed)?;

                    let total_elements = state.a.len() as u32;
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

                    set_compute_pipeline_state(&encoder, &state.pipeline);
                    set_buffer(&encoder, 0, &state.a.buf, state.a.offset);
                    set_buffer(&encoder, 1, &state.b.buf, state.b.offset);
                    set_buffer(&encoder, 2, &state.a.buf, state.a.offset);
                    set_bytes(&encoder, 3, &total_elements);
                    set_bytes(&encoder, 4, &(state.b_len as u32));

                    dispatch_threadgroups(&encoder, groups, threads_per_tg);
                    encoder.endEncoding();
                    Ok(())
                },
            },
        },
    }
}
