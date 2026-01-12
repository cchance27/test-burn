use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::quantized::CanonicalQuantTensor
};

const MAX_ROWS_TILE: usize = 4;
const TILE_COLS: usize = 128;

pub const MATMUL_Q8_NT_MAX_ROWS: usize = MAX_ROWS_TILE;

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmQ8NtParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
    has_bias: u32,
}

pub struct MatmulQ8NtOp;

struct GemmQ8Nt<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    data: CanonicalQuantTensor,
    bias: Option<Tensor<T>>,
    a: Tensor<T>,
    y: Tensor<T>,
    params: GemmQ8NtParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for GemmQ8Nt<T> {
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

impl DefaultKernelInvocable for MatmulQ8NtOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a crate::tensor::QuantizedQ8_0Tensor, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulQ8Nt)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulQ8Nt",
                dtype: T::DTYPE,
            });
        }
        let (a, q8, bias) = args;
        let dims = a.dims();
        if dims.len() != 2 {
            return Err(MetalError::InvalidShape("MatmulQ8Nt expects A with 2 dims".into()));
        }
        let m = dims[0];
        let k = dims[1];
        if m == 0 {
            return Err(MetalError::InvalidShape("MatmulQ8Nt: m must be > 0".into()));
        }
        if m > MAX_ROWS_TILE {
            return Err(MetalError::OperationNotSupported(format!(
                "MatmulQ8Nt currently supports up to {} rows (got {})",
                MAX_ROWS_TILE, m
            )));
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
                "Quant tensor dims {:?} do not match k={}",
                canonical.logical_dims, k
            )));
        };
        if canon_k != k {
            return Err(MetalError::InvalidShape("Canonical K mismatch".into()));
        }

        if let Some(bias_tensor) = &bias
            && bias_tensor.len() != n {
                return Err(MetalError::InvalidShape(format!(
                    "Bias len {} does not match N {}",
                    bias_tensor.len(),
                    n
                )));
            }

        let y = Tensor::new(vec![m, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        if let Some(bias_tensor) = bias {
            ctx.prepare_tensors_for_active_cmd(&[a, &y, bias_tensor])?;
        } else {
            ctx.prepare_tensors_for_active_cmd(&[a, &y])?;
        }

        let params = GemmQ8NtParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: k as u32,
            ldc: n as u32,
            // Derive blocks_per_k from k to support both [K,N] and [N,K] logical dims
            blocks_per_k: k.div_ceil(canonical.weights_per_block) as u32,
            weights_per_block: canonical.weights_per_block as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
        };

        let grid = MTLSize {
            width: n.div_ceil(TILE_COLS),
            height: m.div_ceil(MAX_ROWS_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: TILE_COLS,
            height: 1,
            depth: 1,
        };

        let profiler_label = if crate::profiling_state::get_profiling_state() {
            let mut label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("matmul_q8_nt"));
            // Append to parent scope for TUI visibility
            label.op_name = format!("{}/matmul/gemm_q8_nt (Q)", label.op_name);
            label.backend = "gemm_q8_nt".to_string();
            if crate::profiling_state::get_profiling_state() {
                let mut data = rustc_hash::FxHashMap::default();
                data.insert("op".to_string(), "matmul".to_string());
                data.insert("backend".to_string(), "gemm_q8_nt".to_string());
                data.insert("batch".to_string(), "1".to_string());
                data.insert("m".to_string(), m.to_string());
                data.insert("n".to_string(), n.to_string());
                data.insert("k".to_string(), k.to_string());
                data.insert("tA".to_string(), "0".to_string());
                data.insert("tB".to_string(), "1".to_string());
                label.data = Some(data);
            }
            label
        } else {
            let mut label = GpuProfilerLabel::fallback("gemm_q8_nt");
            label.op_name = format!("{}/matmul/gemm_q8_nt (Q)", label.op_name);
            label.backend = "gemm_q8_nt".to_string();
            label
        };

        let pipeline = if let Some(p) = pipeline {
            p
        } else {
            ctx.kernel_manager.get_pipeline(KernelFunction::MatmulQ8Nt, T::DTYPE, &ctx.device)?
        };

        let op = GemmQ8Nt {
            pipeline,
            data: canonical,
            bias: bias.cloned(),
            a: a.clone(),
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label,
        };

        Ok((Box::new(op), y))
    }
}
