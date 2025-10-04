use super::*;
use crate::metallic::encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state};
use crate::metallic::{
    TensorElement, TensorInit, TensorStorage,
    instrumentation::{MatMulDispatchKind, MatMulDispatchTiming, MatmulDims},
    kernels::{KernelFunction, KernelInvocable, matmul::MatMulBackend},
};
use objc2::msg_send;
use objc2::rc::Retained;
use objc2::runtime::{Bool, ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputePipelineState, MTLCounterSampleBuffer, MTLSize};

#[repr(C)]
struct GemvParams {
    k: u32,
    n: u32,
}

const THREADGROUP_WIDTH: usize = 256;
const TILE_N: usize = THREADGROUP_WIDTH;

pub struct GemvOp;

struct Gemv<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    a: Tensor<T>,
    x: Tensor<T>,
    y: Tensor<T>,
    params: GemvParams,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
    dispatch_timing: Option<MatMulDispatchTiming>,
}

impl<T: TensorElement> Operation for Gemv<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        if let Some(timing) = &self.dispatch_timing {
            if matches!(timing.kind(), MatMulDispatchKind::Compute) {
                let sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer> = timing.sample_buffer();
                unsafe {
                    let _: () = msg_send![
                        &*encoder,
                        sampleCountersInBuffer: sample_buffer,
                        atSampleIndex: timing.start_index(),
                        withBarrier: Bool::YES
                    ];
                }
            }
        }

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(&encoder, 1, &self.x.buf, self.x.offset);
        set_buffer(&encoder, 2, &self.y.buf, self.y.offset);
        set_bytes(&encoder, 3, &self.params);

        dispatch_threadgroups(&encoder, self.grid_size, self.threadgroup_size);

        if let Some(timing) = &self.dispatch_timing {
            if matches!(timing.kind(), MatMulDispatchKind::Compute) {
                let sample_buffer: &ProtocolObject<dyn MTLCounterSampleBuffer> = timing.sample_buffer();
                unsafe {
                    let _: () = msg_send![
                        &*encoder,
                        sampleCountersInBuffer: sample_buffer,
                        atSampleIndex: timing.end_index(),
                        withBarrier: Bool::NO
                    ];
                }
            }
        }

        encoder.endEncoding();
        Ok(())
    }
}

impl KernelInvocable for GemvOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>); // (x, A)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::Gemv)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (x, a) = args;
        let pipeline = pipeline.ok_or(MetalError::PipelineCreationFailed)?;

        // x is (1, K), a is (K, N)
        let x_dims = x.dims();
        let a_dims = a.dims();

        if x_dims.len() != 2 || x_dims[0] != 1 {
            return Err(MetalError::InvalidShape(format!(
                "Invalid shape for GEMV vector x: {:?}, expected (1, K)",
                x_dims
            )));
        }

        if a_dims.len() != 2 || a_dims[0] != x_dims[1] {
            return Err(MetalError::InvalidShape(format!(
                "Invalid shape for GEMV matrix A: {:?}, expected (K, N) where K={}",
                a_dims, x_dims[1]
            )));
        }

        let k = x_dims[1];
        let n = a_dims[1];

        let y = Tensor::new(vec![1, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        ctx.prepare_tensors_for_active_cmd(&[x, a, &y])?;

        let params = GemvParams { k: k as u32, n: n as u32 };

        let threadgroup_size = MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };

        let grid_size = MTLSize {
            width: ((n + TILE_N - 1) / TILE_N),
            height: 1,
            depth: 1,
        };

        let dims = MatmulDims { batch: 1, m: 1, n, k };

        let dispatch_timing = {
            let command_buffer = {
                let command_buffer = ctx.active_command_buffer_mut_without_cache()?;
                command_buffer.clone()
            };
            ctx.register_matmul_dispatch(&command_buffer, MatMulBackend::Gemv, Some(dims), MatMulDispatchKind::Compute)
                .timing()
                .cloned()
        };

        let op = Gemv {
            pipeline,
            a: a.clone(),
            x: x.clone(),
            y: y.clone(),
            params,
            grid_size,
            threadgroup_size,
            dispatch_timing,
        };

        Ok((Box::new(op), y))
    }
}
