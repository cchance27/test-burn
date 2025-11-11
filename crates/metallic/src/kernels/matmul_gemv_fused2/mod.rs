use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLComputeCommandEncoder, MTLComputePipelineState, MTLSize};

use crate::{
    CommandBuffer, MetalError, Operation, Tensor, TensorElement, TensorInit, TensorStorage, context::GpuProfilerLabel, kernels::{DefaultKernelInvocable, KernelFunction}, operation::ComputeKernelEncoder, tensor::{Dtype, QuantizedTensor, quantized::CanonicalQuantTensor}
};

#[repr(C)]
#[derive(Clone, Copy)]
struct Q2Params {
    k: u32,
    n0: u32,
    n1: u32,
    blocks_per_k: u32,
    weights_per_block: u32,
    has_bias0: u32,
    has_bias1: u32,
}

pub struct MatmulGemvQ2FusedOp;

struct FusedQ2Op<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    w0: CanonicalQuantTensor,
    w1: CanonicalQuantTensor,
    bias0: Option<Tensor<T>>,
    bias1: Option<Tensor<T>>,
    y: Tensor<T>, // packed [n0 + n1]
    params: Q2Params,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: crate::context::GpuProfilerLabel,
}

impl<T: TensorElement> Operation for FusedQ2Op<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut crate::caching::ResourceCache) -> Result<(), MetalError> {
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)
            .dispatch_custom(self.grid, self.tg);
        Ok(())
    }

    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        set_buffer(encoder, 0, &self.w0.data.buf, self.w0.data.offset);
        set_buffer(encoder, 1, &self.w1.data.buf, self.w1.data.offset);
        set_buffer(encoder, 2, &self.x.buf, self.x.offset);
        // outputs split from packed y
        let elem = T::DTYPE.size_bytes();
        let n0 = self.params.n0 as usize;
        let off0 = self.y.offset;
        let off1 = self.y.offset + n0 * elem;
        set_buffer(encoder, 3, &self.y.buf, off0);
        set_buffer(encoder, 4, &self.y.buf, off1);
        set_bytes(encoder, 5, &self.params);
        set_buffer(encoder, 6, &self.w0.scales.buf, self.w0.scales.offset);
        set_buffer(encoder, 7, &self.w1.scales.buf, self.w1.scales.offset);
        let fallback = &self.x;
        let (b0_buf, b0_off) = if let Some(bias) = &self.bias0 { (&bias.buf, bias.offset) } else { (&fallback.buf, fallback.offset) };
        let (b1_buf, b1_off) = if let Some(bias) = &self.bias1 { (&bias.buf, bias.offset) } else { (&fallback.buf, fallback.offset) };
        set_buffer(encoder, 8, b0_buf, b0_off);
        set_buffer(encoder, 9, b1_buf, b1_off);
    }
}

impl DefaultKernelInvocable for MatmulGemvQ2FusedOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        (&'a QuantizedTensor<'a>, &'a QuantizedTensor<'a>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvQ2Fused)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype { operation: "MatmulGemvQ2Fused", dtype: T::DTYPE });
        }
        let (x, (w0, w1), (b0_opt, b1_opt)) = args;
        let dims = x.dims();
        if dims.len() != 2 || dims[0] != 1 { return Err(MetalError::InvalidShape("x must be [1,K]".into())); }
        let k = dims[1];
        let canonize = |qt: &QuantizedTensor| -> Result<CanonicalQuantTensor, MetalError> {
            match qt { QuantizedTensor::Q8_0(q8) => CanonicalQuantTensor::from_split_q8_tensor(q8), }
        };
        let w0 = canonize(w0)?;
        let w1 = canonize(w1)?;
        let (n0, n1);
        if w0.logical_dims.len()!=2 || w1.logical_dims.len()!=2 { return Err(MetalError::InvalidShape("Quant tensors must be 2D".into())); }
        if w0.logical_dims[0]!=k || w1.logical_dims[0]!=k { return Err(MetalError::InvalidShape("K mismatch for fused2".into())); }
        n0 = w0.logical_dims[1];
        n1 = w1.logical_dims[1];
        if let Some(b) = &b0_opt { if b.len()!=n0 { return Err(MetalError::InvalidShape("bias0 len mismatch".into())); } }
        if let Some(b) = &b1_opt { if b.len()!=n1 { return Err(MetalError::InvalidShape("bias1 len mismatch".into())); } }

        let y = Tensor::new(vec![n0 + n1], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
        let mut inputs: Vec<&Tensor<T>> = vec![x, &y];
        if let Some(b) = &b0_opt { inputs.push(b); }
        if let Some(b) = &b1_opt { inputs.push(b); }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = Q2Params { k: k as u32, n0: n0 as u32, n1: n1 as u32, blocks_per_k: w0.blocks_per_k as u32, weights_per_block: w0.weights_per_block as u32, has_bias0: (b0_opt.is_some() as u32), has_bias1: (b1_opt.is_some() as u32) };
        let tg = MTLSize { width: 256, height: 1, depth: 1 };
        let grid = MTLSize { width: (n0.max(n1) + 255) / 256, height: 1, depth: 1 };

        let mut label = if crate::profiling_state::get_profiling_state() { ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_q8_fused2")) } else { GpuProfilerLabel::fallback("gemv_q8_fused2") };
        label.op_name = format!("{}/matmul/gemv_q8_fused2 (Q)", label.op_name);
        label.backend = "gemv_q8_fused2".to_string();
        if crate::profiling_state::get_profiling_state() {
            let mut data = rustc_hash::FxHashMap::default();
            data.insert("op".to_string(), "matmul".to_string());
            data.insert("backend".to_string(), "gemv_q8_fused2".to_string());
            data.insert("batch".to_string(), "1".to_string());
            data.insert("m".to_string(), "1".to_string());
            data.insert("k".to_string(), k.to_string());
            data.insert("n0".to_string(), n0.to_string());
            data.insert("n1".to_string(), n1.to_string());
            label.data = Some(data);
        }

        let pipeline = pipeline.ok_or(MetalError::PipelineCreationFailed)?;
        let op = FusedQ2Op { pipeline, x: x.clone(), w0, w1, bias0: b0_opt.cloned(), bias1: b1_opt.cloned(), y: y.clone(), params, grid, tg, profiler_label: label };
        Ok((Box::new(op), y))
    }
}

