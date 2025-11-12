use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use super::*;
use crate::{
    CommandBuffer, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::{QuantizedTensor, TensorType, quantized::CanonicalQuantTensor}
};

#[repr(C)]
#[derive(Clone, Copy)]
struct GemvParams {
    k: u32,
    n: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
}

pub const THREADGROUP_WIDTH: usize = 256;
const TILE_N: usize = THREADGROUP_WIDTH;

pub struct MatmulGemvOp;
pub struct MatmulGemvAddmmOp;

const GEMV_LOADER_DENSE: u32 = 0;
const GEMV_LOADER_DENSE_BIAS: u32 = 1;
const GEMV_LOADER_Q8_CANONICAL: u32 = 2;
const GEMV_LOADER_Q8_CANONICAL_BIAS: u32 = 3;
const GEMV_LOADER_Q8_CANONICAL_DEBUG: u32 = 4;

enum GemvMatrix<T: TensorElement> {
    Dense(Tensor<T>),
    QuantCanonical(CanonicalQuantTensor),
}

struct MatMulGemv<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matrix: GemvMatrix<T>,
    bias: Option<Tensor<T>>,
    residual: Option<Tensor<T>>,
    alpha: f32,
    beta: f32,
    needs_bias_buffer: bool,
    loader_mode: u32,
    x: Tensor<T>,
    y: Tensor<T>,
    params: GemvParams,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
    profiler_label: GpuProfilerLabel,
    diag_col: u32,
}

impl<T: TensorElement> Operation for MatMulGemv<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid_size, self.threadgroup_size);

        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};

        match &self.matrix {
            GemvMatrix::Dense(mat) => {
                set_buffer(encoder, 0, &mat.buf, mat.offset);
                set_buffer(encoder, 6, &self.x.buf, self.x.offset);
            }
            GemvMatrix::QuantCanonical(c) => {
                set_buffer(encoder, 0, &c.data.buf, c.data.offset);
                set_buffer(encoder, 6, &c.scales.buf, c.scales.offset);
            }
        }
        set_buffer(encoder, 1, &self.x.buf, self.x.offset);
        set_buffer(encoder, 2, &self.y.buf, self.y.offset);
        set_bytes(encoder, 3, &self.params);

        // Unified: always bind bias (or x as dummy) and loader_mode.
        if self.needs_bias_buffer {
            let bias = self.bias.as_ref().expect("bias tensor required for bias kernel");
            set_buffer(encoder, 4, &bias.buf, bias.offset);
        } else {
            set_buffer(encoder, 4, &self.x.buf, self.x.offset);
        }
        set_bytes(encoder, 5, &self.loader_mode);
        // Optional: pass diag column for debug mode
        set_bytes(encoder, 8, &self.diag_col);
        // Residual C (for epilogue) and alpha/beta scalars
        if let Some(resid) = &self.residual {
            set_buffer(encoder, 7, &resid.buf, resid.offset);
        } else {
            // Bind y as placeholder; kernel checks beta to skip read
            set_buffer(encoder, 7, &self.y.buf, self.y.offset);
        }
        let alpha = self.alpha;
        let beta = self.beta;
        set_bytes(encoder, 9, &alpha);
        set_bytes(encoder, 10, &beta);
    }
}

impl DefaultKernelInvocable for MatmulGemvOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, TensorType<'a, T>, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        // Default pipeline is the generic GEMV. We may override to a quant-only pipeline
        // within new() when the RHS is quantized to avoid any loader-mode ambiguity.
        Some(KernelFunction::MatmulGemv)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (x, rhs, bias) = args;

        let x_dims = x.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            return Err(MetalError::InvalidShape(format!(
                "Invalid shape for GEMV vector x: {:?}, expected (1, K)",
                x_dims
            )));
        }
        let k = x_dims[1];

        let bias_tensor = bias.cloned();

        let mut canonical_blocks_per_k = 0u32;
        let mut canonical_weights_per_block = 0u32;

        let (matrix, n, loader_mode, needs_bias_buffer) = match rhs {
            TensorType::Dense(a) => {
                let a_dims = a.dims();
                if a_dims.len() != 2 || a_dims[0] != k {
                    return Err(MetalError::InvalidShape(format!(
                        "Invalid shape for GEMV matrix A: {:?}, expected (K, N) where K={}",
                        a_dims, k
                    )));
                }
                let needs_bias = bias_tensor.is_some();
                (
                    GemvMatrix::Dense(a.clone()),
                    a_dims[1],
                    if needs_bias { GEMV_LOADER_DENSE_BIAS } else { GEMV_LOADER_DENSE },
                    needs_bias,
                )
            }
            TensorType::Quant(qrhs) => {
                if T::DTYPE != crate::tensor::Dtype::F16 {
                    return Err(MetalError::UnsupportedDtype {
                        operation: "MatmulGemv/Q8",
                        dtype: T::DTYPE,
                    });
                }
                match qrhs {
                    QuantizedTensor::Q8_0(q8) => {
                        if q8.logical_dims.len() != 2 {
                            return Err(MetalError::InvalidShape(format!(
                                "Q8 GEMV expects weight dims [*,*], got {:?}",
                                q8.logical_dims
                            )));
                        }
                        let canonical = CanonicalQuantTensor::from_split_q8_tensor(q8)
                            .map_err(|e| MetalError::InvalidOperation(format!("Failed to canonicalize Q8 tensor: {e}")))?;
                        let d0 = canonical.logical_dims[0];
                        let d1 = canonical.logical_dims[1];
                        let d0_is_k = d0 == k;
                        let d1_is_k = d1 == k;
                        if !d0_is_k && !d1_is_k {
                            return Err(MetalError::InvalidShape(format!(
                                "Q8 GEMV expects one dim = K ({}), got {:?}",
                                k, canonical.logical_dims
                            )));
                        }
                        let n = if d0_is_k { d1 } else { d0 };
                        let needs_bias = bias_tensor.is_some();
                        canonical_blocks_per_k = canonical.blocks_per_k as u32;
                        canonical_weights_per_block = canonical.weights_per_block as u32;
                        if canonical_weights_per_block == 0 {
                            return Err(MetalError::InvalidOperation(
                                "Q8 canonical params invalid: weights_per_block == 0".to_string(),
                            ));
                        }
                        let mut loader_mode = if needs_bias {
                            GEMV_LOADER_Q8_CANONICAL_BIAS
                        } else {
                            GEMV_LOADER_Q8_CANONICAL
                        };
                        if let Ok(col_str) = std::env::var("METALLIC_GEMV_DEBUG_COL") {
                            if let Ok(_col) = col_str.parse::<u32>() {
                                loader_mode = GEMV_LOADER_Q8_CANONICAL_DEBUG;
                            }
                        }
                        (GemvMatrix::QuantCanonical(canonical), n, loader_mode, needs_bias)
                    }
                }
            }
        };

        let y = Tensor::new(vec![1, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        match &matrix {
            GemvMatrix::Dense(a_tensor) => {
                if let Some(bias) = &bias_tensor {
                    ctx.prepare_tensors_for_active_cmd(&[x, a_tensor, &y, bias])?;
                } else {
                    ctx.prepare_tensors_for_active_cmd(&[x, a_tensor, &y])?;
                }
            }
            GemvMatrix::QuantCanonical(_) => {
                if let Some(bias) = &bias_tensor {
                    ctx.prepare_tensors_for_active_cmd(&[x, &y, bias])?;
                } else {
                    ctx.prepare_tensors_for_active_cmd(&[x, &y])?;
                }
            }
        }

        let params = GemvParams {
            k: k as u32,
            n: n as u32,
            blocks_per_k: canonical_blocks_per_k,
            weights_per_block: canonical_weights_per_block,
        };

        //if cfg!(debug_assertions) {
        //    match &matrix {
        //        GemvMatrix::Dense(_) => {
        //            eprintln!(
        //                "[GEMV PARAMS] dense k={} n={} loader_mode={} bias={}",
        //                params.k, params.n, loader_mode, needs_bias_buffer
        //            );
        //        }
        //        GemvMatrix::QuantCanonical(_) => {
        //            eprintln!(
        //                "[GEMV PARAMS] q8 k={} n={} blocks_per_k={} wpb={} loader_mode={} bias={}",
        //                params.k, params.n, params.blocks_per_k, params.weights_per_block, loader_mode, needs_bias_buffer
        //            );
        //        }
        //    }
        //}

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

        // Keep profiler label creation minimal in hot path when profiling is off
        // Dense vs Quant suffix for visibility
        let dq_suffix = match &matrix {
            GemvMatrix::Dense(_) => " (D)",
            GemvMatrix::QuantCanonical(_) => " (Q)",
        };
        let profiler_label = if crate::profiling_state::get_profiling_state() {
            let mut label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("matmul"));
            // Make GEMV visible in the TUI hierarchy similar to MLX: append op/backend to name
            label.op_name = format!("{}/matmul/gemv{}", label.op_name, dq_suffix);
            label.backend = "gemv".to_string();
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("n".to_string(), n.to_string());
            data.insert("k".to_string(), k.to_string());
            data.insert("tA".to_string(), "0".to_string());
            // When GEMV is used for matmul A*[K,N], we logically have tB=true
            data.insert("tB".to_string(), "1".to_string());
            label.data = Some(data);
            label
        } else {
            let mut label = GpuProfilerLabel::fallback("matmul");
            label.op_name = format!("{}/matmul/gemv{}", label.op_name, dq_suffix);
            label.backend = "gemv".to_string();
            label
        };

        // Select pipeline: Dense path reuses provided pipeline. Quantized path always uses the generic GEMV kernel
        // (which internally routes to the canonical Q8 implementation based on params).
        let pipeline = match &matrix {
            GemvMatrix::Dense(_) => pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("matmulGemvOp".to_string()))?,
            GemvMatrix::QuantCanonical(_) => {
                if let Some(existing) = pipeline {
                    existing
                } else {
                    ctx.kernel_manager.get_pipeline(KernelFunction::MatmulGemv, T::DTYPE, &ctx.device)?
                }
            }
        };

        let diag_col = std::env::var("METALLIC_GEMV_DEBUG_COL")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(u32::MAX);
        let op = MatMulGemv {
            pipeline,
            matrix,
            bias: bias_tensor,
            residual: None,
            alpha: 1.0,
            beta: 0.0,
            needs_bias_buffer,
            loader_mode,
            x: x.clone(),
            y: y.clone(),
            params,
            grid_size,
            threadgroup_size,
            profiler_label,
            diag_col,
        };

        Ok((Box::new(op), y))
    }
}

// Addmm-style GEMV with optional bias and residual: y = alpha * (A*x [+ bias]) + beta * residual
impl DefaultKernelInvocable for MatmulGemvAddmmOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        TensorType<'a, T>,
        Option<&'a Tensor<T>>, // bias
        Option<&'a Tensor<T>>, // residual C
        f32,                   // alpha
        f32,                   // beta
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemv)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (x, rhs, bias_opt, residual_opt, alpha, beta) = args;

        let x_dims = x.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            return Err(MetalError::InvalidShape(format!(
                "Invalid shape for GEMV vector x: {:?}, expected (1, K)",
                x_dims
            )));
        }
        let k = x_dims[1];

        let bias_tensor = bias_opt.cloned();
        let residual_tensor = residual_opt.cloned();

        let mut canonical_blocks_per_k = 0u32;
        let mut canonical_weights_per_block = 0u32;

        let (matrix, n, loader_mode, needs_bias_buffer) = match rhs {
            TensorType::Dense(a) => {
                let a_dims = a.dims();
                if a_dims.len() != 2 || a_dims[0] != k {
                    return Err(MetalError::InvalidShape(format!(
                        "Invalid shape for GEMV matrix A: {:?}, expected (K, N) where K={}",
                        a_dims, k
                    )));
                }
                let needs_bias = bias_tensor.is_some();
                (
                    GemvMatrix::Dense(a.clone()),
                    a_dims[1],
                    if needs_bias { GEMV_LOADER_DENSE_BIAS } else { GEMV_LOADER_DENSE },
                    needs_bias,
                )
            }
            TensorType::Quant(qrhs) => {
                if T::DTYPE != crate::tensor::Dtype::F16 {
                    return Err(MetalError::UnsupportedDtype {
                        operation: "MatmulGemvAddmm/Q8",
                        dtype: T::DTYPE,
                    });
                }
                match qrhs {
                    QuantizedTensor::Q8_0(q8) => {
                        if q8.logical_dims.len() != 2 {
                            return Err(MetalError::InvalidShape(format!(
                                "Q8 GEMV expects weight dims [*,*], got {:?}",
                                q8.logical_dims
                            )));
                        }
                        let canonical = CanonicalQuantTensor::from_split_q8_tensor(q8)
                            .map_err(|e| MetalError::InvalidOperation(format!("Failed to canonicalize Q8 tensor: {e}")))?;
                        let d0 = canonical.logical_dims[0];
                        let d1 = canonical.logical_dims[1];
                        let d0_is_k = d0 == k;
                        let d1_is_k = d1 == k;
                        if !d0_is_k && !d1_is_k {
                            return Err(MetalError::InvalidShape(format!(
                                "Q8 GEMV expects one dim = K ({}), got {:?}",
                                k, canonical.logical_dims
                            )));
                        }
                        let n = if d0_is_k { d1 } else { d0 };
                        let needs_bias = bias_tensor.is_some();
                        canonical_blocks_per_k = canonical.blocks_per_k as u32;
                        canonical_weights_per_block = canonical.weights_per_block as u32;
                        if canonical_weights_per_block == 0 {
                            return Err(MetalError::InvalidOperation(
                                "Q8 canonical params invalid: weights_per_block == 0".to_string(),
                            ));
                        }
                        let mut loader_mode = if needs_bias {
                            GEMV_LOADER_Q8_CANONICAL_BIAS
                        } else {
                            GEMV_LOADER_Q8_CANONICAL
                        };
                        if let Ok(col_str) = std::env::var("METALLIC_GEMV_DEBUG_COL") {
                            if let Ok(_col) = col_str.parse::<u32>() {
                                loader_mode = GEMV_LOADER_Q8_CANONICAL_DEBUG;
                            }
                        }
                        (GemvMatrix::QuantCanonical(canonical), n, loader_mode, needs_bias)
                    }
                }
            }
        };

        if let Some(bias) = &bias_tensor {
            if bias.len() != n {
                return Err(MetalError::InvalidShape(format!("Bias len {} does not match N {}", bias.len(), n)));
            }
        }
        if let Some(resid) = &residual_tensor {
            let rd = resid.dims();
            if rd.len() != 2 || rd[0] != 1 || rd[1] != n {
                return Err(MetalError::InvalidShape(format!("Residual shape {:?} must be [1, N={}]", rd, n)));
            }
        }

        let y = Tensor::new(vec![1, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        match &matrix {
            GemvMatrix::Dense(a_tensor) => {
                let mut inputs: Vec<&Tensor<T>> = vec![x, a_tensor, &y];
                if let Some(b) = &bias_tensor {
                    inputs.push(b);
                }
                if let Some(r) = &residual_tensor {
                    inputs.push(r);
                }
                ctx.prepare_tensors_for_active_cmd(&inputs)?;
            }
            GemvMatrix::QuantCanonical(_) => {
                let mut inputs: Vec<&Tensor<T>> = vec![x, &y];
                if let Some(b) = &bias_tensor {
                    inputs.push(b);
                }
                if let Some(r) = &residual_tensor {
                    inputs.push(r);
                }
                ctx.prepare_tensors_for_active_cmd(&inputs)?;
            }
        }

        let params = GemvParams {
            k: k as u32,
            n: n as u32,
            blocks_per_k: canonical_blocks_per_k,
            weights_per_block: canonical_weights_per_block,
        };

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

        let dq_suffix = match &matrix {
            GemvMatrix::Dense(_) => " (D)",
            GemvMatrix::QuantCanonical(_) => " (Q)",
        };
        let profiler_label = if crate::profiling_state::get_profiling_state() {
            let mut label = ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("matmul"));
            label.op_name = format!("{}/matmul/gemv_addmm{}", label.op_name, dq_suffix);
            label.backend = "gemv".to_string();
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("n".to_string(), n.to_string());
            data.insert("k".to_string(), k.to_string());
            data.insert("tA".to_string(), "0".to_string());
            data.insert("tB".to_string(), "1".to_string());
            label.data = Some(data);
            label
        } else {
            let mut label = GpuProfilerLabel::fallback("matmul");
            label.op_name = format!("{}/matmul/gemv_addmm{}", label.op_name, dq_suffix);
            label.backend = "gemv".to_string();
            label
        };

        let pipeline = match &matrix {
            GemvMatrix::Dense(_) => pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("MatmulGemvAddmmOp".to_string()))?,
            GemvMatrix::QuantCanonical(_) => {
                if let Some(existing) = pipeline {
                    existing
                } else {
                    ctx.kernel_manager.get_pipeline(KernelFunction::MatmulGemv, T::DTYPE, &ctx.device)?
                }
            }
        };

        let op = MatMulGemv {
            pipeline,
            matrix,
            bias: bias_tensor,
            residual: residual_tensor,
            alpha,
            beta,
            needs_bias_buffer,
            loader_mode,
            x: x.clone(),
            y: y.clone(),
            params,
            grid_size,
            threadgroup_size,
            profiler_label,
            diag_col: u32::MAX,
        };

        Ok((Box::new(op), y))
    }
}
