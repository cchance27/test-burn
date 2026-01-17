use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::{CanonicalF16Tensor, F16_CANONICAL_WEIGHTS_PER_BLOCK}
};

const TILE_COLS_TOTAL: usize = 128;
const TILE_COLS_PER_TG: usize = TILE_COLS_TOTAL / 2;
const ROWS_PER_TILE: usize = 32;
const COLS_PER_THREAD: usize = 2;
const TG_ROWS: usize = 4;
const TG_COL_LANES: usize = TILE_COLS_PER_TG / COLS_PER_THREAD;

#[repr(C)]
#[derive(Clone, Copy)]
struct GemmF16CanonicalParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldc: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
    has_bias: u32,
}

pub struct MatmulF16CanonicalOp;
pub struct MatmulF16CanonicalRows16Op;

struct GemmF16Canonical<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    data: CanonicalF16Tensor<T>,
    bias: Option<Tensor<T>>,
    a: Tensor<T>,
    y: Tensor<T>,
    params: GemmF16CanonicalParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
}

impl<T: TensorElement> Operation for GemmF16Canonical<T> {
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
        set_buffer(encoder, 1, &self.a.buf, self.a.offset);
        set_buffer(encoder, 2, &self.y.buf, self.y.offset);
        set_bytes(encoder, 3, &self.params);
        if let Some(bias) = &self.bias {
            set_buffer(encoder, 4, &bias.buf, bias.offset);
        } else {
            set_buffer(encoder, 4, &self.a.buf, self.a.offset);
        }
    }
}

impl DefaultKernelInvocable for MatmulF16CanonicalOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a CanonicalF16Tensor<T>, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulF16CanonicalLargeN)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulF16Canonical",
                dtype: T::DTYPE,
            });
        }

        let (a, canon, bias) = args;
        let dims = a.dims();
        if dims.len() != 2 {
            return Err(MetalError::InvalidShape("MatmulF16Canonical expects A with 2 dims".into()));
        }
        let m = dims[0];
        let k = dims[1];
        if m == 0 {
            return Err(MetalError::InvalidShape("MatmulF16Canonical: m must be > 0".into()));
        }

        if m == 1 {
            use super::base::MatmulGemvOp;
            use crate::tensor::TensorType;
            let rhs = TensorType::DenseCanonical(canon);
            return MatmulGemvOp::new(ctx, (a, rhs, false, bias), None, _cache);
        }

        if canon.logical_dims.len() != 2 {
            return Err(MetalError::InvalidShape("Canonical tensor must be 2D".into()));
        }
        let (canon_k, n) = (canon.logical_dims[0], canon.logical_dims[1]);
        if canon_k != k {
            return Err(MetalError::InvalidShape(format!(
                "Canonical K mismatch: canonical k={} vs A k={}",
                canon_k, k
            )));
        }
        let expected_blocks = k.div_ceil(F16_CANONICAL_WEIGHTS_PER_BLOCK);
        if canon.blocks_per_k != expected_blocks {
            return Err(MetalError::InvalidShape(format!(
                "Canonical blocks_per_k {} does not match expected {}",
                canon.blocks_per_k, expected_blocks
            )));
        }

        if F16_CANONICAL_WEIGHTS_PER_BLOCK != 32 {
            return Err(MetalError::InvalidOperation(
                "F16 canonical kernel assumes 32 weights per block".into(),
            ));
        }

        if let Some(bias_tensor) = &bias
            && bias_tensor.len() != n
        {
            return Err(MetalError::InvalidShape(format!(
                "Bias len {} does not match N {}",
                bias_tensor.len(),
                n
            )));
        }

        let y = Tensor::new(vec![m, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        if let Some(bias_tensor) = bias {
            ctx.prepare_tensors_for_active_cmd(&[a, &y, &canon.data, bias_tensor])?;
        } else {
            ctx.prepare_tensors_for_active_cmd(&[a, &y, &canon.data])?;
        }

        let params = GemmF16CanonicalParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: k as u32,
            ldc: n as u32,
            blocks_per_k: canon.blocks_per_k as u32,
            weights_per_block: F16_CANONICAL_WEIGHTS_PER_BLOCK as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
        };

        let grid = MTLSize {
            width: n.div_ceil(TILE_COLS_PER_TG),
            height: m.div_ceil(ROWS_PER_TILE),
            depth: 1,
        };
        let tg = MTLSize {
            width: TG_COL_LANES,
            height: TG_ROWS,
            depth: 1,
        };

        let mut label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemm_f16_canonical_n"))
        } else {
            GpuProfilerLabel::fallback("gemm_f16_canonical_n")
        };
        label.op_name = format!("{}/matmul/gemm_f16_canonical_n (F16)", label.op_name);
        label.backend = "gemm_f16_canonical_n".to_string();
        if crate::profiling_state::get_profiling_state() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemm_f16_canonical_n".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), m.to_string());
            data.insert("n".to_string(), n.to_string());
            data.insert("k".to_string(), k.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "0".to_string());
            label.data = Some(data);
        }

        let pipeline = pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("MatmulF16CanonicalOp".to_string()))?;
        let op = GemmF16Canonical {
            pipeline,
            data: canon.clone(),
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

impl DefaultKernelInvocable for MatmulF16CanonicalRows16Op {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a CanonicalF16Tensor<T>, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulF16CanonicalRows16LargeN)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulF16CanonicalRows16",
                dtype: T::DTYPE,
            });
        }

        let (a, canon, bias) = args;
        let dims = a.dims();
        if dims.len() != 2 {
            return Err(MetalError::InvalidShape("MatmulF16CanonicalRows16 expects A with 2 dims".into()));
        }
        let m = dims[0];
        let k = dims[1];
        if m == 0 {
            return Err(MetalError::InvalidShape("MatmulF16CanonicalRows16: m must be > 0".into()));
        }

        if canon.logical_dims.len() != 2 {
            return Err(MetalError::InvalidShape("Canonical tensor must be 2D".into()));
        }
        let (canon_k, n) = (canon.logical_dims[0], canon.logical_dims[1]);
        if canon_k != k {
            return Err(MetalError::InvalidShape(format!(
                "Canonical K mismatch: canonical k={} vs A k={}",
                canon_k, k
            )));
        }
        let expected_blocks = k.div_ceil(F16_CANONICAL_WEIGHTS_PER_BLOCK);
        if canon.blocks_per_k != expected_blocks {
            return Err(MetalError::InvalidShape(format!(
                "Canonical blocks_per_k {} does not match expected {}",
                canon.blocks_per_k, expected_blocks
            )));
        }

        if let Some(bias_tensor) = &bias
            && bias_tensor.len() != n
        {
            return Err(MetalError::InvalidShape(format!(
                "Bias len {} does not match N {}",
                bias_tensor.len(),
                n
            )));
        }

        let y = Tensor::new(vec![m, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        if let Some(bias_tensor) = bias {
            ctx.prepare_tensors_for_active_cmd(&[a, &y, &canon.data, bias_tensor])?;
        } else {
            ctx.prepare_tensors_for_active_cmd(&[a, &y, &canon.data])?;
        }

        let params = GemmF16CanonicalParams {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: k as u32,
            ldc: n as u32,
            blocks_per_k: canon.blocks_per_k as u32,
            weights_per_block: F16_CANONICAL_WEIGHTS_PER_BLOCK as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
        };

        const ROWS_PER_TILE_16: usize = 16;
        let grid = MTLSize {
            width: n.div_ceil(TILE_COLS_PER_TG),
            height: m.div_ceil(ROWS_PER_TILE_16),
            depth: 1,
        };
        let tg = MTLSize {
            width: TG_COL_LANES,
            height: TG_ROWS,
            depth: 1,
        };

        let mut label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemm_f16_canonical_n_rows16"))
        } else {
            GpuProfilerLabel::fallback("gemm_f16_canonical_n_rows16")
        };
        label.op_name = format!("{}/matmul/gemm_f16_canonical_n_rows16 (F16)", label.op_name);
        label.backend = "gemm_f16_canonical_n_rows16".to_string();
        if crate::profiling_state::get_profiling_state() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemm_f16_canonical_n_rows16".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), m.to_string());
            data.insert("n".to_string(), n.to_string());
            data.insert("k".to_string(), k.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "0".to_string());
            label.data = Some(data);
        }

        let pipeline = pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("MatmulF16CanonicalRows16Op".to_string()))?;
        let op = GemmF16Canonical {
            pipeline,
            data: canon.clone(),
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
