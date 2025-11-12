use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, caching::ResourceCache, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::{Dtype, QuantizedTensor, quantized::CanonicalQuantTensor}
};

#[repr(C)]
#[derive(Clone, Copy)]
struct QkvParams {
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

// Reuse GEMV tiling constants to stay consistent with kernel.metal TILE_N
use crate::kernels::matmul_gemv::{GEMV_COLS_PER_THREAD, THREADGROUP_WIDTH};

pub struct MatmulGemvQkvFusedOp;

struct FusedQkvOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    wq: CanonicalQuantTensor,
    wk: CanonicalQuantTensor,
    wv: CanonicalQuantTensor,
    bias_q: Option<Tensor<T>>,
    bias_k: Option<Tensor<T>>,
    bias_v: Option<Tensor<T>>,
    y: Tensor<T>, // packed [nq + nk + nv]
    params: QkvParams,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: crate::context::GpuProfilerLabel,
}

impl<T: TensorElement> Operation for FusedQkvOp<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        // Buffers layout must match gemv_q8_fused3_f16
        set_buffer(encoder, 0, &self.wq.data.buf, self.wq.data.offset);
        set_buffer(encoder, 1, &self.wk.data.buf, self.wk.data.offset);
        set_buffer(encoder, 2, &self.wv.data.buf, self.wv.data.offset);
        set_buffer(encoder, 3, &self.x.buf, self.x.offset);
        let elem = T::DTYPE.size_bytes();
        let nq = self.params.nq as usize;
        let nk = self.params.nk as usize;
        // nv implied by total length offset
        let off_q = self.y.offset + 0;
        let off_k = self.y.offset + nq * elem;
        let off_v = self.y.offset + (nq + nk) * elem;
        set_buffer(encoder, 4, &self.y.buf, off_q);
        set_buffer(encoder, 5, &self.y.buf, off_k);
        set_buffer(encoder, 6, &self.y.buf, off_v);
        set_bytes(encoder, 7, &self.params);
        set_buffer(encoder, 8, &self.wq.scales.buf, self.wq.scales.offset);
        set_buffer(encoder, 9, &self.wk.scales.buf, self.wk.scales.offset);
        set_buffer(encoder, 10, &self.wv.scales.buf, self.wv.scales.offset);
        let fallback = &self.x;
        let (bias_q_buf, bias_q_offset) = if let Some(bias) = &self.bias_q {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };
        set_buffer(encoder, 11, bias_q_buf, bias_q_offset);
        let (bias_k_buf, bias_k_offset) = if let Some(bias) = &self.bias_k {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };
        set_buffer(encoder, 12, bias_k_buf, bias_k_offset);
        let (bias_v_buf, bias_v_offset) = if let Some(bias) = &self.bias_v {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };
        set_buffer(encoder, 13, bias_v_buf, bias_v_offset);
    }
}

impl DefaultKernelInvocable for MatmulGemvQkvFusedOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        (&'a QuantizedTensor<'a>, &'a QuantizedTensor<'a>, &'a QuantizedTensor<'a>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvQkvFused)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulGemvQkvFused",
                dtype: T::DTYPE,
            });
        }
        let (x, (wq, wk, wv), (bias_q_opt, bias_k_opt, bias_v_opt)) = args;
        let x_dims = x.dims();
        if x_dims.len() != 2 || x_dims[0] != 1 {
            return Err(MetalError::InvalidShape("x must be [1,K]".into()));
        }
        let k = x_dims[1];

        // Canonicalize the three Q8 tensors
        let canonize = |qt: &QuantizedTensor| -> Result<CanonicalQuantTensor, MetalError> {
            match qt {
                QuantizedTensor::Q8_0(q8) => CanonicalQuantTensor::from_split_q8_tensor(q8),
            }
        };
        let wq = canonize(wq)?;
        let wk = canonize(wk)?;
        let wv = canonize(wv)?;

        // Validate shapes (K, N)
        let (nq, nk, nv) = (wq.logical_dims[1], wk.logical_dims[1], wv.logical_dims[1]);
        if wq.logical_dims[0] != k || wk.logical_dims[0] != k || wv.logical_dims[0] != k {
            return Err(MetalError::InvalidShape("QKV K mismatch".into()));
        }
        // Allocate packed output as a flat buffer [nq + nk + nv]
        let total = nq + nk + nv;
        let y = Tensor::new(vec![total], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let bias_q = bias_q_opt.map(Tensor::clone);
        let bias_k = bias_k_opt.map(Tensor::clone);
        let bias_v = bias_v_opt.map(Tensor::clone);

        // Prepare inputs for the active command buffer, including biases when present
        {
            let mut inputs: Vec<&Tensor<T>> = vec![x, &y];
            if let Some(b) = &bias_q {
                inputs.push(b);
            }
            if let Some(b) = &bias_k {
                inputs.push(b);
            }
            if let Some(b) = &bias_v {
                inputs.push(b);
            }
            ctx.prepare_tensors_for_active_cmd(&inputs)?;
        }

        let params = QkvParams {
            k: k as u32,
            nq: nq as u32,
            nk: nk as u32,
            nv: nv as u32,
            // Derive blocks_per_k from k to be robust to logical dim order
            blocks_per_k: k.div_ceil(wq.weights_per_block) as u32,
            weights_per_block: wq.weights_per_block as u32,
            has_bias_q: bias_q.is_some() as u32,
            has_bias_k: bias_k.is_some() as u32,
            has_bias_v: bias_v.is_some() as u32,
        };

        let tg = MTLSize {
            width: THREADGROUP_WIDTH,
            height: 1,
            depth: 1,
        };
        let max_n = nq.max(nk.max(nv));
        let tile_n = THREADGROUP_WIDTH * GEMV_COLS_PER_THREAD;
        let grid = MTLSize {
            width: max_n.div_ceil(tile_n),
            height: 1,
            depth: 1,
        };

        let pipeline = pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("MatmulGemvQkvFusedOp".to_string()))?;

        // Build profiler label under current scope for TUI hierarchy
        let mut profiler_label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| crate::context::GpuProfilerLabel::fallback("gemv_qkv_fused"))
        } else {
            crate::context::GpuProfilerLabel::fallback("gemv_qkv_fused")
        };
        profiler_label.op_name = format!("{}/matmul/gemv_qkv_fused (Q)", profiler_label.op_name);
        profiler_label.backend = "gemv_qkv_fused".to_string();
        if crate::profiling_state::get_profiling_state() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv_qkv_fused".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("k".to_string(), k.to_string());
            data.insert("nq".to_string(), nq.to_string());
            data.insert("nk".to_string(), nk.to_string());
            data.insert("nv".to_string(), nv.to_string());
            profiler_label.data = Some(data);
        }

        let op: FusedQkvOp<T> = FusedQkvOp {
            pipeline,
            x: x.clone(),
            wq,
            wk,
            wv,
            bias_q,
            bias_k,
            bias_v,
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label,
        };

        Ok((Box::new(op), y))
    }
}
