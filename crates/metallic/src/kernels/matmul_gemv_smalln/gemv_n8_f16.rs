use half::f16;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSUInteger;
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, Context, MetalError, Operation, ResourceCache, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::{ComputeKernelEncoder}
};

// Public, user-facing, zero-sized struct for the operation.
pub struct MatmulGemvSmallN8Op;

// Internal struct that holds data for the `Operation` trait.
struct MatmulGemvSmallN8<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    m: u32,
    k: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

impl DefaultKernelInvocable for MatmulGemvSmallN8Op {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvSmallN8)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;
        let (m, k) = (a.dims()[0], a.dims()[1]);
        let n = b.dims()[1];

        if n != 8 {
            return Err(MetalError::InvalidShape(format!("MatmulGemvSmallN8Op requires N=8, got N={}", n)));
        }

        ctx.prepare_tensors_for_active_cmd(&[a, b])?;

        let out = Tensor::new(vec![m, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("matmul_gemv_small_n8_op"));

        let op = MatmulGemvSmallN8 {
            a: a.clone(),
            b: b.clone(),
            out: out.clone(),
            m: m as u32,
            k: k as u32,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
            profiler_label,
        };

        Ok((Box::new(op), out))
    }
}

impl<T: TensorElement> Operation for MatmulGemvSmallN8<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        let m = self.a.dims()[0] as u32;
        
        const ROWS_PER_TG: u32 = 8;
        const COLS_PER_TG: u32 = 8;
        const TILE_K_SIZE: u32 = 64;

        let threadgroups = MTLSize {
            width: m.div_ceil(ROWS_PER_TG) as usize,
            height: 1,
            depth: 1,
        };

        let threads_per_threadgroup = MTLSize {
            width: ROWS_PER_TG as usize * COLS_PER_TG as usize,
            height: 1,
            depth: 1,
        };

        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .with_encoder(|encoder| {
                let smem_b_size = (TILE_K_SIZE as usize * COLS_PER_TG as usize * std::mem::size_of::<f16>()) as NSUInteger;
                unsafe {
                    encoder.setThreadgroupMemoryLength_atIndex(smem_b_size, 0);
                }
            })
            .dispatch_custom(threadgroups, threads_per_threadgroup);

        Ok(())
    }

    fn bind_to_encoder(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        set_buffer(encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(encoder, 2, &self.out.buf, self.out.offset);
        set_bytes(encoder, 3, &self.m);
        set_bytes(encoder, 4, &self.k);
    }
}
