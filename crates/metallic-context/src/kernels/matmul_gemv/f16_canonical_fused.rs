use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{CustomKernelInvocable, DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::{
        Dtype, canonical::{CanonicalF16Tensor, F16_CANONICAL_WEIGHTS_PER_BLOCK}
    }
};

#[repr(C)]
#[derive(Clone, Copy)]
struct QkvFusedParams {
    k: u32,
    nq: u32,
    nk: u32,
    nv: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
    has_bias_q: u32,
    has_bias_k: u32,
    has_bias_v: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Q2FusedParams {
    k: u32,
    n0: u32,
    n1: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
    has_bias0: u32,
    has_bias1: u32,
}

pub struct MatmulF16CanonicalQkvFusedOp;
pub struct MatmulF16CanonicalQkvFusedRmsnormOp;
pub struct MatmulF16CanonicalSwiGluOp;
pub struct MatmulF16CanonicalSwiGluRmsnormOp;

// =================================================================================================
// QKV Fused Ops
// =================================================================================================

struct F16CanonicalQkvFusedOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    w_q: CanonicalF16Tensor<T>,
    w_k: CanonicalF16Tensor<T>,
    w_v: CanonicalF16Tensor<T>,
    bias_q: Option<Tensor<T>>,
    bias_k: Option<Tensor<T>>,
    bias_v: Option<Tensor<T>>,
    out_q: Tensor<T>,
    out_k: Tensor<T>,
    out_v: Tensor<T>,
    params: QkvFusedParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
}

struct F16CanonicalQkvFusedRmsnormOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    gamma: Tensor<T>,
    w_q: CanonicalF16Tensor<T>,
    w_k: CanonicalF16Tensor<T>,
    w_v: CanonicalF16Tensor<T>,
    bias_q: Option<Tensor<T>>,
    bias_k: Option<Tensor<T>>,
    bias_v: Option<Tensor<T>>,
    out_q: Tensor<T>,
    out_k: Tensor<T>,
    out_v: Tensor<T>,
    params: QkvFusedParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
    eps: f32,
}

impl<T: TensorElement> Operation for F16CanonicalQkvFusedOp<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        let encoder = ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?;
        encoder
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.w_q.data.buf, self.w_q.data.offset);
        set_buffer(encoder, 1, &self.w_k.data.buf, self.w_k.data.offset);
        set_buffer(encoder, 2, &self.w_v.data.buf, self.w_v.data.offset);
        set_buffer(encoder, 3, &self.x.buf, self.x.offset);
        set_buffer(encoder, 4, &self.out_q.buf, self.out_q.offset);
        set_buffer(encoder, 5, &self.out_k.buf, self.out_k.offset);
        set_buffer(encoder, 6, &self.out_v.buf, self.out_v.offset);
        set_bytes(encoder, 7, &self.params);

        let fallback = &self.x;
        let (bq, bq_off) = self
            .bias_q
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));
        let (bk, bk_off) = self
            .bias_k
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));
        let (bv, bv_off) = self
            .bias_v
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));

        set_buffer(encoder, 8, bq, bq_off);
        set_buffer(encoder, 9, bk, bk_off);
        set_buffer(encoder, 10, bv, bv_off);
    }
}

impl<T: TensorElement> Operation for F16CanonicalQkvFusedRmsnormOp<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        let encoder = ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?;
        encoder
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.w_q.data.buf, self.w_q.data.offset);
        set_buffer(encoder, 1, &self.w_k.data.buf, self.w_k.data.offset);
        set_buffer(encoder, 2, &self.w_v.data.buf, self.w_v.data.offset);
        set_buffer(encoder, 3, &self.x.buf, self.x.offset);
        set_buffer(encoder, 4, &self.out_q.buf, self.out_q.offset);
        set_buffer(encoder, 5, &self.out_k.buf, self.out_k.offset);
        set_buffer(encoder, 6, &self.out_v.buf, self.out_v.offset);
        set_bytes(encoder, 7, &self.params);

        let fallback = &self.x;
        let (bq, bq_off) = self
            .bias_q
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));
        let (bk, bk_off) = self
            .bias_k
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));
        let (bv, bv_off) = self
            .bias_v
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));

        set_buffer(encoder, 8, bq, bq_off);
        set_buffer(encoder, 9, bk, bk_off);
        set_buffer(encoder, 10, bv, bv_off);
        set_buffer(encoder, 11, &self.gamma.buf, self.gamma.offset);
        set_bytes(encoder, 12, &self.eps);
    }
}

// =================================================================================================
// SwiGLU Fused Ops
// =================================================================================================

struct F16CanonicalSwiGluOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    w_gate: CanonicalF16Tensor<T>,
    w_up: CanonicalF16Tensor<T>,
    bias_gate: Option<Tensor<T>>,
    bias_up: Option<Tensor<T>>,
    y: Tensor<T>,
    params: Q2FusedParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
}

struct F16CanonicalSwiGluRmsnormOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    gamma: Tensor<T>,
    w_gate: CanonicalF16Tensor<T>,
    w_up: CanonicalF16Tensor<T>,
    bias_gate: Option<Tensor<T>>,
    bias_up: Option<Tensor<T>>,
    y: Tensor<T>,
    params: Q2FusedParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: GpuProfilerLabel,
    eps: f32,
}

impl<T: TensorElement> Operation for F16CanonicalSwiGluOp<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        let encoder = ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?;
        encoder
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.w_gate.data.buf, self.w_gate.data.offset);
        set_buffer(encoder, 1, &self.w_up.data.buf, self.w_up.data.offset);
        set_buffer(encoder, 2, &self.x.buf, self.x.offset);
        set_buffer(encoder, 3, &self.y.buf, self.y.offset);
        set_bytes(encoder, 4, &self.params);

        let fallback = &self.x;
        let (bg, bg_off) = self
            .bias_gate
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));
        let (bu, bu_off) = self
            .bias_up
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));

        set_buffer(encoder, 5, bg, bg_off);
        set_buffer(encoder, 6, bu, bu_off);
    }
}

impl<T: TensorElement> Operation for F16CanonicalSwiGluRmsnormOp<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        let encoder = ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?;
        encoder
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.w_gate.data.buf, self.w_gate.data.offset);
        set_buffer(encoder, 1, &self.w_up.data.buf, self.w_up.data.offset);
        set_buffer(encoder, 2, &self.x.buf, self.x.offset);
        set_buffer(encoder, 3, &self.y.buf, self.y.offset);
        set_bytes(encoder, 4, &self.params);

        let fallback = &self.x;
        let (bg, bg_off) = self
            .bias_gate
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));
        let (bu, bu_off) = self
            .bias_up
            .as_ref()
            .map(|b| (&b.buf, b.offset))
            .unwrap_or((&fallback.buf, fallback.offset));

        set_buffer(encoder, 5, bg, bg_off);
        set_buffer(encoder, 6, bu, bu_off);
        set_buffer(encoder, 7, &self.gamma.buf, self.gamma.offset);
        set_bytes(encoder, 8, &self.eps);
    }
}

// =================================================================================================
// DefaultKernelInvocable Implementations
// =================================================================================================

impl CustomKernelInvocable for MatmulF16CanonicalQkvFusedOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        (&'a CanonicalF16Tensor<T>, &'a CanonicalF16Tensor<T>, &'a CanonicalF16Tensor<T>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    type OutputTuple<T: TensorElement> = (T, T, T);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulF16CanonicalQkvFused)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, (Tensor<T>, Tensor<T>, Tensor<T>)), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulF16CanonicalQkvFused",
                dtype: T::DTYPE,
            });
        }
        let (x, (wq, wk, wv), (bq, bk, bv)) = args;
        let k = x.dims().last().copied().unwrap_or(0);

        let nq = wq.logical_dims[1];
        let nk = wk.logical_dims[1];
        let nv = wv.logical_dims[1];

        // Ensure preparation
        let out_q = Tensor::new(vec![1, nq], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let out_k = Tensor::new(vec![1, nk], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let out_v = Tensor::new(vec![1, nv], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let mut inputs = vec![x, &wq.data, &wk.data, &wv.data, &out_q, &out_k, &out_v];
        if let Some(b) = bq {
            inputs.push(b);
        }
        if let Some(b) = bk {
            inputs.push(b);
        }
        if let Some(b) = bv {
            inputs.push(b);
        }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = QkvFusedParams {
            k: k as u32,
            nq: nq as u32,
            nk: nk as u32,
            nv: nv as u32,
            blocks_per_k: wq.blocks_per_k as u32,
            weights_per_block: F16_CANONICAL_WEIGHTS_PER_BLOCK as u32,
            has_bias_q: bq.is_some() as u32,
            has_bias_k: bk.is_some() as u32,
            has_bias_v: bv.is_some() as u32,
        };

        let tile_n = 8;
        let max_n = nq.max(nk).max(nv);
        let grid = MTLSize {
            width: max_n.div_ceil(tile_n),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let pipeline = pipeline.ok_or(MetalError::PipelineCreationFailed("MatmulF16CanonicalQkvFused".into()))?;

        let label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_f16_canonical_qkv"))
        } else {
            GpuProfilerLabel::fallback("gemv_f16_canonical_qkv")
        };
        // label.op_name = ... (optional customization)

        let op = F16CanonicalQkvFusedOp {
            pipeline,
            x: x.clone(),
            w_q: wq.clone(),
            w_k: wk.clone(),
            w_v: wv.clone(),
            bias_q: bq.cloned(),
            bias_k: bk.cloned(),
            bias_v: bv.cloned(),
            out_q: out_q.clone(),
            out_k: out_k.clone(),
            out_v: out_v.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
        };
        Ok((Box::new(op), (out_q, out_k, out_v)))
    }
}

impl CustomKernelInvocable for MatmulF16CanonicalQkvFusedRmsnormOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>, // Gamma
        f32,           // Epsilon
        (&'a CanonicalF16Tensor<T>, &'a CanonicalF16Tensor<T>, &'a CanonicalF16Tensor<T>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    type OutputTuple<T: TensorElement> = (T, T, T);

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulF16CanonicalQkvFusedRmsnorm)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, (Tensor<T>, Tensor<T>, Tensor<T>)), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulF16CanonicalQkvFusedRmsnorm",
                dtype: T::DTYPE,
            });
        }
        let (x, gamma, eps, (wq, wk, wv), (bq, bk, bv)) = args;
        let k = x.dims().last().copied().unwrap_or(0);
        let nq = wq.logical_dims[1];
        let nk = wk.logical_dims[1];
        let nv = wv.logical_dims[1];

        let out_q = Tensor::new(vec![1, nq], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let out_k = Tensor::new(vec![1, nk], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let out_v = Tensor::new(vec![1, nv], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let mut inputs = vec![x, gamma, &wq.data, &wk.data, &wv.data, &out_q, &out_k, &out_v];
        if let Some(b) = bq {
            inputs.push(b);
        }
        if let Some(b) = bk {
            inputs.push(b);
        }
        if let Some(b) = bv {
            inputs.push(b);
        }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = QkvFusedParams {
            k: k as u32,
            nq: nq as u32,
            nk: nk as u32,
            nv: nv as u32,
            blocks_per_k: wq.blocks_per_k as u32,
            weights_per_block: F16_CANONICAL_WEIGHTS_PER_BLOCK as u32,
            has_bias_q: bq.is_some() as u32,
            has_bias_k: bk.is_some() as u32,
            has_bias_v: bv.is_some() as u32,
        };

        let tile_n = 8;
        let max_n = nq.max(nk).max(nv);
        let grid = MTLSize {
            width: max_n.div_ceil(tile_n),
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let pipeline = pipeline.ok_or(MetalError::PipelineCreationFailed("MatmulF16CanonicalQkvFusedRmsnorm".into()))?;
        let label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_f16_canonical_qkv_rmsnorm"))
        } else {
            GpuProfilerLabel::fallback("gemv_f16_canonical_qkv_rmsnorm")
        };

        let op = F16CanonicalQkvFusedRmsnormOp {
            pipeline,
            x: x.clone(),
            gamma: gamma.clone(),
            w_q: wq.clone(),
            w_k: wk.clone(),
            w_v: wv.clone(),
            bias_q: bq.cloned(),
            bias_k: bk.cloned(),
            bias_v: bv.cloned(),
            out_q: out_q.clone(),
            out_k: out_k.clone(),
            out_v: out_v.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
            eps,
        };
        Ok((Box::new(op), (out_q, out_k, out_v)))
    }
}

impl DefaultKernelInvocable for MatmulF16CanonicalSwiGluOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        (&'a CanonicalF16Tensor<T>, &'a CanonicalF16Tensor<T>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulF16CanonicalSwiGlu)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulF16CanonicalSwiGlu",
                dtype: T::DTYPE,
            });
        }
        let (x, (wg, wu), (bg, bu)) = args;
        let k = x.dims().last().copied().unwrap_or(0);
        let n0 = wg.logical_dims[1];
        let n1 = wu.logical_dims[1];

        let y = Tensor::new(vec![1, n0], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let mut inputs = vec![x, &wg.data, &wu.data, &y];
        if let Some(b) = bg {
            inputs.push(b);
        }
        if let Some(b) = bu {
            inputs.push(b);
        }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = Q2FusedParams {
            k: k as u32,
            n0: n0 as u32,
            n1: n1 as u32,
            blocks_per_k: wg.blocks_per_k as u32,
            weights_per_block: F16_CANONICAL_WEIGHTS_PER_BLOCK as u32,
            has_bias0: bg.is_some() as u32,
            has_bias1: bu.is_some() as u32,
        };

        let col_groups = n0.div_ceil(8);
        let grid = MTLSize {
            width: col_groups,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let pipeline = pipeline.ok_or(MetalError::PipelineCreationFailed("MatmulF16CanonicalSwiGlu".into()))?;
        let label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_f16_canonical_swiglu"))
        } else {
            GpuProfilerLabel::fallback("gemv_f16_canonical_swiglu")
        };

        let op = F16CanonicalSwiGluOp {
            pipeline,
            x: x.clone(),
            w_gate: wg.clone(),
            w_up: wu.clone(),
            bias_gate: bg.cloned(),
            bias_up: bu.cloned(),
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
        };
        Ok((Box::new(op), y))
    }
}

impl DefaultKernelInvocable for MatmulF16CanonicalSwiGluRmsnormOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>,
        f32,
        (&'a CanonicalF16Tensor<T>, &'a CanonicalF16Tensor<T>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulF16CanonicalSwiGluRmsnorm)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulF16CanonicalSwiGluRmsnorm",
                dtype: T::DTYPE,
            });
        }
        let (x, gamma, eps, (wg, wu), (bg, bu)) = args;
        let k = x.dims().last().copied().unwrap_or(0);
        let n0 = wg.logical_dims[1];
        let n1 = wu.logical_dims[1];

        let y = Tensor::new(vec![1, n0], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let mut inputs = vec![x, gamma, &wg.data, &wu.data, &y];
        if let Some(b) = bg {
            inputs.push(b);
        }
        if let Some(b) = bu {
            inputs.push(b);
        }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = Q2FusedParams {
            k: k as u32,
            n0: n0 as u32,
            n1: n1 as u32,
            blocks_per_k: wg.blocks_per_k as u32,
            weights_per_block: F16_CANONICAL_WEIGHTS_PER_BLOCK as u32,
            has_bias0: bg.is_some() as u32,
            has_bias1: bu.is_some() as u32,
        };

        let col_groups = n0.div_ceil(8);
        let grid = MTLSize {
            width: col_groups,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };

        let pipeline = pipeline.ok_or(MetalError::PipelineCreationFailed("MatmulF16CanonicalSwiGluRmsnorm".into()))?;
        let label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_f16_canonical_swiglu_rmsnorm"))
        } else {
            GpuProfilerLabel::fallback("gemv_f16_canonical_swiglu_rmsnorm")
        };

        let op = F16CanonicalSwiGluRmsnormOp {
            pipeline,
            x: x.clone(),
            gamma: gamma.clone(),
            w_gate: wg.clone(),
            w_up: wu.clone(),
            bias_gate: bg.cloned(),
            bias_up: bu.cloned(),
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
            eps,
        };
        Ok((Box::new(op), y))
    }
}
