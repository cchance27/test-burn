use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};
use rustc_hash::FxHashMap;

use super::{
    helpers::{GemvDispatch, GemvLoaderMode, GemvParams, GemvRhsBinding, ResolvedGemvRhs, THREADGROUP_WIDTH, TILE_N, resolve_rhs}, q8_nt::MatmulQ8NtOp
};
use crate::{
    CommandBuffer, Context, GemvError, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, caching::ResourceCache, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::TensorType
};

#[derive(Clone, Copy, Debug)]
enum GemvColsVariant {
    Cols2,
    Cols4,
    Cols8,
}

fn gemv_cols_variant() -> GemvColsVariant {
    use std::sync::OnceLock;
    static VARIANT: OnceLock<GemvColsVariant> = OnceLock::new();
    *VARIANT.get_or_init(|| match std::env::var("METALLIC_GEMV_COLS_PER_TG").ok().as_deref() {
        Some("2") => GemvColsVariant::Cols2,
        Some("8") => GemvColsVariant::Cols8,
        _ => GemvColsVariant::Cols4,
    })
}

pub struct MatmulGemvOp;
pub struct MatmulGemvAddmmOp;
pub struct MatmulGemvRmsnormOp;

struct MatMulGemv<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rhs: GemvRhsBinding<T>,
    bias: Option<Tensor<T>>,
    residual: Option<Tensor<T>>,
    alpha: f32,
    beta: f32,
    dispatch: GemvDispatch,
    x: Tensor<T>,
    y: Tensor<T>,
    params: GemvParams,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
    profiler_label: GpuProfilerLabel,
}

struct MatMulGemvRmsnorm<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    rhs: GemvRhsBinding<T>,
    bias: Option<Tensor<T>>,
    residual: Option<Tensor<T>>,
    alpha: f32,
    beta: f32,
    dispatch: GemvDispatch,
    x: Tensor<T>,
    gamma: Tensor<T>,
    y: Tensor<T>,
    params: GemvParams,
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
    profiler_label: GpuProfilerLabel,
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

        match &self.rhs {
            GemvRhsBinding::Dense(mat) => {
                set_buffer(encoder, 0, &mat.buf, mat.offset);
                set_buffer(encoder, 6, &self.x.buf, self.x.offset);
            }
            GemvRhsBinding::DenseCanonical(canon) => {
                set_buffer(encoder, 0, &canon.data.buf, canon.data.offset);
                set_buffer(encoder, 6, &self.x.buf, self.x.offset);
            }
            GemvRhsBinding::QuantCanonical(c) => {
                set_buffer(encoder, 0, &c.data.buf, c.data.offset);
                set_buffer(encoder, 6, &c.scales.buf, c.scales.offset);
            }
        }

        set_buffer(encoder, 1, &self.x.buf, self.x.offset);
        set_buffer(encoder, 2, &self.y.buf, self.y.offset);
        set_bytes(encoder, 3, &self.params);

        if self.dispatch.needs_bias_buffer() {
            let bias = self.bias.as_ref().expect("bias tensor required for bias kernel");
            set_buffer(encoder, 4, &bias.buf, bias.offset);
        } else {
            set_buffer(encoder, 4, &self.x.buf, self.x.offset);
        }

        let loader_mode = self.dispatch.loader_id();
        set_bytes(encoder, 5, &loader_mode);
        let diag = self.dispatch.diag_col();
        set_bytes(encoder, 8, &diag);

        if let Some(resid) = &self.residual {
            set_buffer(encoder, 7, &resid.buf, resid.offset);
        } else {
            set_buffer(encoder, 7, &self.y.buf, self.y.offset);
        }

        let alpha = self.alpha;
        let beta = self.beta;
        set_bytes(encoder, 9, &alpha);
        set_bytes(encoder, 10, &beta);
    }
}

impl<T: TensorElement> Operation for MatMulGemvRmsnorm<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid_size, self.threadgroup_size);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};

        match &self.rhs {
            GemvRhsBinding::Dense(mat) => {
                set_buffer(encoder, 0, &mat.buf, mat.offset);
                set_buffer(encoder, 6, &self.x.buf, self.x.offset);
            }
            GemvRhsBinding::DenseCanonical(canon) => {
                set_buffer(encoder, 0, &canon.data.buf, canon.data.offset);
                set_buffer(encoder, 6, &self.x.buf, self.x.offset);
            }
            GemvRhsBinding::QuantCanonical(c) => {
                set_buffer(encoder, 0, &c.data.buf, c.data.offset);
                set_buffer(encoder, 6, &c.scales.buf, c.scales.offset);
            }
        }

        set_buffer(encoder, 1, &self.x.buf, self.x.offset);
        set_buffer(encoder, 2, &self.y.buf, self.y.offset);
        set_bytes(encoder, 3, &self.params);

        if self.dispatch.needs_bias_buffer() {
            let bias = self.bias.as_ref().expect("bias tensor required for bias kernel");
            set_buffer(encoder, 4, &bias.buf, bias.offset);
        } else {
            set_buffer(encoder, 4, &self.x.buf, self.x.offset);
        }

        let loader_mode = self.dispatch.loader_id();
        set_bytes(encoder, 5, &loader_mode);
        let diag = self.dispatch.diag_col();
        set_bytes(encoder, 8, &diag);

        if let Some(resid) = &self.residual {
            set_buffer(encoder, 7, &resid.buf, resid.offset);
        } else {
            set_buffer(encoder, 7, &self.y.buf, self.y.offset);
        }

        let alpha = self.alpha;
        let beta = self.beta;
        set_bytes(encoder, 9, &alpha);
        set_bytes(encoder, 10, &beta);

        set_buffer(encoder, 11, &self.gamma.buf, self.gamma.offset);
    }
}

fn validate_x_shape<T: TensorElement>(x: &Tensor<T>) -> Result<usize, MetalError> {
    let x_dims = x.dims();
    if x_dims.len() != 2 || x_dims[0] != 1 {
        return Err(GemvError::VectorShape { actual: x_dims.to_vec() }.into());
    }
    Ok(x_dims[1])
}

fn build_operation<'a, T: TensorElement>(
    ctx: &mut Context<T>,
    x: &'a Tensor<T>,
    k: usize,
    rhs: ResolvedGemvRhs<'a, T>,
    bias: Option<&'a Tensor<T>>,
    residual: Option<&'a Tensor<T>>,
    alpha: f32,
    beta: f32,
    pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    profiler_tag: &'static str,
    diag_col: u32,
    transpose_right: bool,
) -> Result<(MatMulGemv<T>, Tensor<T>), MetalError> {
    let ResolvedGemvRhs {
        binding,
        n,
        loader_mode,
        needs_bias_buffer,
        quant_meta,
        blocks_per_k,
        weights_per_block,
    } = rhs;

    if let Some(bias_tensor) = &bias {
        if bias_tensor.len() != n {
            return Err(GemvError::BiasLengthMismatch {
                expected: n,
                actual: bias_tensor.len(),
            }
            .into());
        }
    }

    if let Some(residual_tensor) = &residual {
        let rd = residual_tensor.dims();
        if rd.len() != 2 || rd[0] != 1 || rd[1] != n {
            return Err(GemvError::ResidualShapeMismatch {
                expected: n,
                actual: rd.to_vec(),
            }
            .into());
        }
    }

    let y = Tensor::new(vec![1, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

    let mut inputs: Vec<&Tensor<T>> = vec![x, &y];
    match &binding {
        GemvRhsBinding::Dense(rhs_tensor) => inputs.push(rhs_tensor),
        GemvRhsBinding::DenseCanonical(rhs_tensor) => inputs.push(&rhs_tensor.data),
        GemvRhsBinding::QuantCanonical(_) => {}
    }
    if let Some(b) = &bias {
        inputs.push(b);
    }
    if let Some(r) = &residual {
        inputs.push(r);
    }
    ctx.prepare_tensors_for_active_cmd(&inputs)?;

    let _ = quant_meta;

    let params = GemvParams {
        k: k as u32,
        n: n as u32,
        blocks_per_k,
        weights_per_block,
    };
    let dispatch = GemvDispatch::new(loader_mode, needs_bias_buffer, diag_col);

    // Grid configuration depends on the kernel loader mode.
    // Q8Canonical kernels use our new SIMD-Parallel logic (4 cols per 128-thread TG).
    // Grid configuration depends on the kernel loader mode.
    // Q8Canonical kernels use our new SIMD-Parallel logic (4 cols per 128-thread TG).
    // Dense kernels use legacy Thread-Per-Col logic (256 cols per 256-thread TG).
    // UPDATE: Dense now uses SIMD-Parallel logic too (via run_simd_f16_gemv).
    let lid = dispatch.loader_id();
    let is_simd_q8 = lid == (GemvLoaderMode::Q8Canonical as u32)
        || lid == (GemvLoaderMode::Q8CanonicalBias as u32)
        || lid == (GemvLoaderMode::Q8CanonicalDebug as u32);

    // Dense FP16 with transpose_right MUST use SIMD kernel (layout [Out, In]).
    // Dense FP16 without transpose_right MUST use Legacy kernel (layout [In, Out]).
    let use_simd_dense = matches!(binding, GemvRhsBinding::Dense(_)) && transpose_right;
    let use_simd_canonical = matches!(binding, GemvRhsBinding::DenseCanonical(_));

    let (tg_width, tile_cols) = if is_simd_q8 || use_simd_dense || use_simd_canonical {
        let cols = if use_simd_dense || use_simd_canonical {
            8usize
        } else {
            match gemv_cols_variant() {
                GemvColsVariant::Cols2 => 2usize,
                GemvColsVariant::Cols4 => 4usize,
                GemvColsVariant::Cols8 => 8usize,
            }
        };
        (cols * 32usize, cols)
    } else {
        (THREADGROUP_WIDTH, TILE_N)
    };

    let threadgroup_size = MTLSize {
        width: tg_width as usize,
        height: 1,
        depth: 1,
    };
    let grid_size = MTLSize {
        width: n.div_ceil(tile_cols) as usize,
        height: 1,
        depth: 1,
    };

    let dq_suffix = match &binding {
        GemvRhsBinding::Dense(_) => " (D)",
        GemvRhsBinding::DenseCanonical(_) => " (C)",
        GemvRhsBinding::QuantCanonical(_) => " (Q)",
    };

    let mut profiler_label = if crate::profiling_state::get_profiling_state() {
        ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback(profiler_tag))
    } else {
        GpuProfilerLabel::fallback(profiler_tag)
    };
    profiler_label.op_name = format!("{}/matmul/{}{}", profiler_label.op_name, profiler_tag, dq_suffix);
    profiler_label.backend = "gemv".to_string();

    if crate::profiling_state::get_profiling_state() {
        let mut data = FxHashMap::default();
        data.insert("op".to_string(), "matmul".to_string());
        data.insert("backend".to_string(), "gemv".to_string());
        data.insert("batch".to_string(), "1".to_string());
        data.insert("m".to_string(), "1".to_string());
        data.insert("n".to_string(), n.to_string());
        data.insert("k".to_string(), k.to_string());
        data.insert("tA".to_string(), "0".to_string());
        data.insert("tB".to_string(), "1".to_string());
        profiler_label.data = Some(data);
    }

    let kernel_fn = match (&binding, gemv_cols_variant()) {
        (GemvRhsBinding::Dense(_), _) => {
            if transpose_right {
                KernelFunction::MatmulGemvCols8
            } else {
                KernelFunction::MatmulGemv
            }
        }
        (GemvRhsBinding::DenseCanonical(_), _) => KernelFunction::MatmulGemvCols8,
        (GemvRhsBinding::QuantCanonical(_), GemvColsVariant::Cols2) => KernelFunction::MatmulGemvQ8Cols2,
        (GemvRhsBinding::QuantCanonical(_), GemvColsVariant::Cols4) => KernelFunction::MatmulGemvQ8,
        (GemvRhsBinding::QuantCanonical(_), GemvColsVariant::Cols8) => KernelFunction::MatmulGemvQ8Cols8,
    };

    let pipeline = match &binding {
        GemvRhsBinding::Dense(_) | GemvRhsBinding::DenseCanonical(_) => {
            if let Some(existing) = pipeline {
                if kernel_fn == KernelFunction::MatmulGemv {
                    existing
                } else {
                    ctx.kernel_manager.get_pipeline(kernel_fn, T::DTYPE, &ctx.device)?
                }
            } else {
                ctx.kernel_manager.get_pipeline(kernel_fn, T::DTYPE, &ctx.device)?
            }
        }
        GemvRhsBinding::QuantCanonical(_) => ctx.kernel_manager.get_pipeline(kernel_fn, T::DTYPE, &ctx.device)?,
    };

    let op = MatMulGemv {
        pipeline,
        rhs: binding,
        bias: bias.cloned(),
        residual: residual.cloned(),
        alpha,
        beta,
        dispatch,
        x: x.clone(),
        y: y.clone(),
        params,
        grid_size,
        threadgroup_size,
        profiler_label,
    };

    Ok((op, y))
}

fn build_operation_rmsnorm<'a, T: TensorElement>(
    ctx: &mut Context<T>,
    x: &'a Tensor<T>,
    gamma: &'a Tensor<T>,
    k: usize,
    rhs: ResolvedGemvRhs<'a, T>,
    bias: Option<&'a Tensor<T>>,
    residual: Option<&'a Tensor<T>>,
    alpha: f32,
    beta: f32,
    pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    profiler_tag: &'static str,
) -> Result<(MatMulGemvRmsnorm<T>, Tensor<T>), MetalError> {
    let ResolvedGemvRhs {
        binding,
        n,
        loader_mode,
        needs_bias_buffer,
        quant_meta,
        blocks_per_k,
        weights_per_block,
    } = rhs;

    if gamma.dims() != [k] {
        return Err(MetalError::InvalidShape(format!(
            "Gamma shape {:?} does not match K={}",
            gamma.dims(),
            k
        )));
    }

    if let Some(bias_tensor) = &bias {
        if bias_tensor.len() != n {
            return Err(GemvError::BiasLengthMismatch {
                expected: n,
                actual: bias_tensor.len(),
            }
            .into());
        }
    }

    if let Some(residual_tensor) = &residual {
        let rd = residual_tensor.dims();
        if rd.len() != 2 || rd[0] != 1 || rd[1] != n {
            return Err(GemvError::ResidualShapeMismatch {
                expected: n,
                actual: rd.to_vec(),
            }
            .into());
        }
    }

    let y = Tensor::new(vec![1, n], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

    let mut inputs: Vec<&Tensor<T>> = vec![x, gamma, &y];
    match &binding {
        GemvRhsBinding::Dense(rhs_tensor) => inputs.push(rhs_tensor),
        GemvRhsBinding::DenseCanonical(rhs_tensor) => inputs.push(&rhs_tensor.data),
        GemvRhsBinding::QuantCanonical(_) => {}
    }
    if let Some(b) = &bias {
        inputs.push(b);
    }
    if let Some(r) = &residual {
        inputs.push(r);
    }
    ctx.prepare_tensors_for_active_cmd(&inputs)?;

    let _ = quant_meta;

    let params = GemvParams {
        k: k as u32,
        n: n as u32,
        blocks_per_k,
        weights_per_block,
    };
    let dispatch = GemvDispatch::new(loader_mode, needs_bias_buffer, u32::MAX);

    let lid = dispatch.loader_id();
    let is_simd_q8 = lid == (GemvLoaderMode::Q8Canonical as u32)
        || lid == (GemvLoaderMode::Q8CanonicalBias as u32)
        || lid == (GemvLoaderMode::Q8CanonicalDebug as u32)
        || lid == (GemvLoaderMode::Dense as u32)
        || lid == (GemvLoaderMode::DenseBias as u32)
        || lid == (GemvLoaderMode::DenseCanonical as u32)
        || lid == (GemvLoaderMode::DenseCanonicalBias as u32);

    let (tg_width, tile_cols) = if is_simd_q8 {
        (128usize, 4usize)
    } else {
        (THREADGROUP_WIDTH, TILE_N)
    };

    let threadgroup_size = MTLSize {
        width: tg_width as usize,
        height: 1,
        depth: 1,
    };
    let grid_size = MTLSize {
        width: n.div_ceil(tile_cols) as usize,
        height: 1,
        depth: 1,
    };

    let dq_suffix = match &binding {
        GemvRhsBinding::Dense(_) => " (D)",
        GemvRhsBinding::DenseCanonical(_) => " (C)",
        GemvRhsBinding::QuantCanonical(_) => " (Q)",
    };

    let mut profiler_label = if crate::profiling_state::get_profiling_state() {
        ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback(profiler_tag))
    } else {
        GpuProfilerLabel::fallback(profiler_tag)
    };
    profiler_label.op_name = format!("{}/matmul/{}{}", profiler_label.op_name, profiler_tag, dq_suffix);
    profiler_label.backend = "gemv_rmsnorm".to_string();

    if crate::profiling_state::get_profiling_state() {
        let mut data = FxHashMap::default();
        data.insert("op".to_string(), "matmul".to_string());
        data.insert("backend".to_string(), "gemv_rmsnorm".to_string());
        data.insert("batch".to_string(), "1".to_string());
        data.insert("m".to_string(), "1".to_string());
        data.insert("n".to_string(), n.to_string());
        data.insert("k".to_string(), k.to_string());
        data.insert("tA".to_string(), "0".to_string());
        data.insert("tB".to_string(), "1".to_string());
        profiler_label.data = Some(data);
    }

    let pipeline = match &binding {
        GemvRhsBinding::Dense(_) | GemvRhsBinding::DenseCanonical(_) => {
            if let Some(existing) = pipeline {
                existing
            } else {
                ctx.kernel_manager
                    .get_pipeline(KernelFunction::MatmulGemvRmsnorm, T::DTYPE, &ctx.device)?
            }
        }
        GemvRhsBinding::QuantCanonical(_) => {
            if let Some(existing) = pipeline {
                existing
            } else {
                ctx.kernel_manager
                    .get_pipeline(KernelFunction::MatmulGemvQ8Rmsnorm, T::DTYPE, &ctx.device)?
            }
        }
    };

    let op = MatMulGemvRmsnorm {
        pipeline,
        rhs: binding,
        bias: bias.cloned(),
        residual: residual.cloned(),
        alpha,
        beta,
        dispatch,
        x: x.clone(),
        gamma: gamma.clone(),
        y: y.clone(),
        params,
        grid_size,
        threadgroup_size,
        profiler_label,
    };

    Ok((op, y))
}

fn gemv_debug_column() -> u32 {
    std::env::var("METALLIC_GEMV_DEBUG_COL")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}

impl DefaultKernelInvocable for MatmulGemvOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, TensorType<'a, T>, bool, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemv)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (x, rhs, transpose_right, bias) = args;
        let k = validate_x_shape(x)?;
        let bias_tensor = bias.cloned();
        let resolved = resolve_rhs(rhs, k, bias_tensor.is_some())?;

        if resolved.quant_meta.is_some() && !matches!(resolved.loader_mode, GemvLoaderMode::Q8CanonicalDebug) {
            let k_u = k as u32;
            let n_u = resolved.n as u32;
            if k_u <= 1024 && n_u <= 1024 {
                if let Some(meta) = resolved.quant_meta {
                    let (delegated_op, out_tensor) =
                        <MatmulQ8NtOp as DefaultKernelInvocable>::new(ctx, (&x, meta.source, bias), None, None)?;
                    return Ok((delegated_op, out_tensor));
                }
            }
        }

        let (op, y) = build_operation(
            ctx,
            x,
            k,
            resolved,
            bias,
            None,
            1.0,
            0.0,
            pipeline,
            "gemv",
            gemv_debug_column(),
            transpose_right,
        )?;
        Ok((Box::new(op), y))
    }
}

impl DefaultKernelInvocable for MatmulGemvAddmmOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        TensorType<'a, T>,
        Option<&'a Tensor<T>>,
        Option<&'a Tensor<T>>,
        bool,
        f32,
        f32,
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
        let (x, rhs, bias, residual, transpose_right, alpha, beta) = args;
        let k = validate_x_shape(x)?;
        let bias_tensor = bias.cloned();
        let resolved = resolve_rhs(rhs, k, bias_tensor.is_some())?;

        let (op, y) = build_operation(
            ctx,
            x,
            k,
            resolved,
            bias,
            residual,
            alpha,
            beta,
            pipeline,
            "gemv_addmm",
            u32::MAX,
            transpose_right,
        )?;
        Ok((Box::new(op), y))
    }
}

impl DefaultKernelInvocable for MatmulGemvRmsnormOp {
    type Args<'a, T: TensorElement> = (&'a Tensor<T>, &'a Tensor<T>, TensorType<'a, T>, Option<&'a Tensor<T>>);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvRmsnorm)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != crate::tensor::Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulGemvRmsnorm",
                dtype: T::DTYPE,
            });
        }

        let (x, gamma, rhs, bias) = args;
        let k = validate_x_shape(x)?;
        let bias_tensor = bias.cloned();
        let resolved = resolve_rhs(rhs, k, bias_tensor.is_some())?;
        let use_pipeline = matches!(&resolved.binding, GemvRhsBinding::Dense(_));
        let pipeline = if use_pipeline { pipeline } else { None };
        let (op, y) = build_operation_rmsnorm(ctx, x, gamma, k, resolved, bias, None, 1.0, 0.0, pipeline, "gemv_rmsnorm")?;
        Ok((Box::new(op), y))
    }
}
