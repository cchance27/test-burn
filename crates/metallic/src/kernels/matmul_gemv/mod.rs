use metallic_instrumentation::GpuProfiler;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, encoder::{dispatch_threadgroups, set_buffer, set_bytes, set_compute_pipeline_state}, kernels::{KernelFunction, KernelInvocable}
};

#[repr(C)]
struct GemvParams {
    k: u32,
    n: u32,
}

const THREADGROUP_WIDTH: usize = 256;
const TILE_N: usize = THREADGROUP_WIDTH;

pub struct MatmulGemvOp;

struct MatMulGemv<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    a: Tensor<T>,
    x: Tensor<T>,
    y: Tensor<T>,
    params: GemvParams,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for MatMulGemv<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let encoder = command_buffer.get_compute_encoder()?;

        let label = self.profiler_label.clone();
        let _scope = GpuProfiler::profile_compute(command_buffer.raw(), &encoder, label.op_name, label.backend);

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(&encoder, 1, &self.x.buf, self.x.offset);
        set_buffer(&encoder, 2, &self.y.buf, self.y.offset);
        set_bytes(&encoder, 3, &self.params);

        dispatch_threadgroups(&encoder, self.grid_size, self.threadgroup_size);

        Ok(())
    }
}

impl KernelInvocable for MatmulGemvOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>); // (x, A)

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemv)
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
            width: n.div_ceil(TILE_N),
            height: 1,
            depth: 1,
        };

        let profiler_label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_op"));

        let op = MatMulGemv {
            pipeline,
            a: a.clone(),
            x: x.clone(),
            y: y.clone(),
            params,
            grid_size,
            threadgroup_size,
            profiler_label,
        };

        Ok((Box::new(op), y))
    }
}
