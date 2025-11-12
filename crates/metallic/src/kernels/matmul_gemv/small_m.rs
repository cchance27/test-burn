use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use super::helpers::{GEMV_COLS_PER_THREAD, THREADGROUP_WIDTH};
use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::quantized::CanonicalQuantTensor
};

#[repr(C)]
#[derive(Clone, Copy)]
struct RowsGemvParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
    has_bias: u32,
}

pub struct MatmulGemvSmallMOp;

struct GemvRowsOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    data: CanonicalQuantTensor,
    bias: Option<Tensor<T>>,
    a: Tensor<T>,
    y: Tensor<T>,
    params: RowsGemvParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for GemvRowsOp<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.data.data.buf, self.data.data.offset);
        set_buffer(encoder, 1, &self.data.scales.buf, self.data.scales.offset);
        set_buffer(encoder, 2, &self.a.buf, self.a.offset);
        set_buffer(encoder, 3, &self.y.buf, self.y.offset);
        set_bytes(encoder, 4, &self.params);
        if let Some(bias) = &self.bias {
            set_buffer(encoder, 5, &bias.buf, bias.offset);
        } else {
            set_buffer(encoder, 5, &self.a.buf, self.a.offset);
        }
    }
}

impl DefaultKernelInvocable for MatmulGemvSmallMOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a crate::tensor::QuantizedQ8_0Tensor, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvSmallM)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulGemvSmallM",
                dtype: T::DTYPE,
            });
        }
        let (a, q8, bias) = args;
        let dims = a.dims();
        if dims.len() != 2 {
            return Err(MetalError::InvalidShape("MatmulGemvSmallM expects A with 2 dims".into()));
        }
        let m = dims[0];
        let k = dims[1];
        if m == 0 || m > 4 {
            return Err(MetalError::OperationNotSupported("MatmulGemvSmallM supports 1..=4 rows".into()));
        }

        let canonical = CanonicalQuantTensor::from_split_q8_tensor(q8)
            .map_err(|e| MetalError::InvalidOperation(format!("Failed to canonicalize Q8 tensor: {e}")))?;
        if canonical.logical_dims.len() != 2 {
            return Err(MetalError::InvalidShape("Quant tensor must be 2D".into()));
        }
        let (dim0, dim1) = (canonical.logical_dims[0], canonical.logical_dims[1]);
        let (canon_k, n) = if dim0 == k {
            (dim0, dim1)
        } else if dim1 == k {
            (dim1, dim0)
        } else {
            return Err(MetalError::InvalidShape(format!(
                "Quant dims {:?} do not match k={}",
                canonical.logical_dims, k
            )));
        };
        if canon_k != k {
            return Err(MetalError::InvalidShape("Canonical K mismatch".into()));
        }

        if let Some(b) = &bias {
            if b.len() != n {
                return Err(MetalError::InvalidShape(format!("Bias len {} != N {}", b.len(), n)));
            }
        }

        let y = Tensor::new(vec![m, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        if let Some(b) = bias {
            ctx.prepare_tensors_for_active_cmd(&[a, &y, b])?;
        } else {
            ctx.prepare_tensors_for_active_cmd(&[a, &y])?;
        }

        let params = RowsGemvParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: k as u32,
            ldc: n as u32,
            // Derive blocks_per_k from k to support both [K,N] and [N,K]
            blocks_per_k: k.div_ceil(canonical.weights_per_block) as u32,
            weights_per_block: canonical.weights_per_block as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
        };

        let tile_n = THREADGROUP_WIDTH * GEMV_COLS_PER_THREAD;
        let grid = MTLSize {
            width: n.div_ceil(tile_n),
            height: m.div_ceil(4),
            depth: 1,
        };
        let tg = MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };

        let label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_q8_rows"))
        } else {
            GpuProfilerLabel::fallback("gemv_q8_rows")
        };

        let pipeline = pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("matmulGemvSmallMOp".to_string()))?;
        let op = GemvRowsOp {
            pipeline,
            data: canonical,
            bias: bias.cloned(),
            a: a.clone(),
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
        };
        Ok((Box::new(op), y))
    }
}
