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

pub struct MatmulGemvQ8SwiGluOp;
pub struct MatmulGemvQ8SwiGluRmsnormOp;

struct SwiGluQ8Op<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    w_gate: CanonicalQuantTensor,
    w_up: CanonicalQuantTensor,
    bias_gate: Option<Tensor<T>>,
    bias_up: Option<Tensor<T>>,
    y: Tensor<T>,
    params: Q2Params,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: crate::context::GpuProfilerLabel,
}

struct SwiGluQ8RmsnormOp<T: TensorElement> {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    x: Tensor<T>,
    gamma: Tensor<T>,
    w_gate: CanonicalQuantTensor,
    w_up: CanonicalQuantTensor,
    bias_gate: Option<Tensor<T>>,
    bias_up: Option<Tensor<T>>,
    y: Tensor<T>,
    params: Q2Params,
    grid: MTLSize,
    tg: MTLSize,
    profiler_label: crate::context::GpuProfilerLabel,
}

impl<T: TensorElement> Operation for SwiGluQ8Op<T> {
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
        // buffer(0): data_g
        set_buffer(encoder, 0, &self.w_gate.data.buf, self.w_gate.data.offset);
        // buffer(1): data_u
        set_buffer(encoder, 1, &self.w_up.data.buf, self.w_up.data.offset);
        // buffer(2): vector_x
        set_buffer(encoder, 2, &self.x.buf, self.x.offset);
        // buffer(3): out_res
        set_buffer(encoder, 3, &self.y.buf, self.y.offset);
        // buffer(4): params
        set_bytes(encoder, 4, &self.params);
        // buffer(5): scales_g
        set_buffer(encoder, 5, &self.w_gate.scales.buf, self.w_gate.scales.offset);
        // buffer(6): scales_u
        set_buffer(encoder, 6, &self.w_up.scales.buf, self.w_up.scales.offset);

        let fallback = &self.x; // Just a valid buffer reference if bias missing
        let (bg_buf, bg_off) = if let Some(bias) = &self.bias_gate {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };
        let (bu_buf, bu_off) = if let Some(bias) = &self.bias_up {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };

        // buffer(7): bias_g
        set_buffer(encoder, 7, bg_buf, bg_off);
        // buffer(8): bias_u
        set_buffer(encoder, 8, bu_buf, bu_off);
    }
}

impl<T: TensorElement> Operation for SwiGluQ8RmsnormOp<T> {
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
        set_buffer(encoder, 5, &self.w_gate.scales.buf, self.w_gate.scales.offset);
        set_buffer(encoder, 6, &self.w_up.scales.buf, self.w_up.scales.offset);

        let fallback = &self.x;
        let (bg_buf, bg_off) = if let Some(bias) = &self.bias_gate {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };
        let (bu_buf, bu_off) = if let Some(bias) = &self.bias_up {
            (&bias.buf, bias.offset)
        } else {
            (&fallback.buf, fallback.offset)
        };

        set_buffer(encoder, 7, bg_buf, bg_off);
        set_buffer(encoder, 8, bu_buf, bu_off);
        set_buffer(encoder, 9, &self.gamma.buf, self.gamma.offset);
    }
}

impl DefaultKernelInvocable for MatmulGemvQ8SwiGluOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        (&'a QuantizedTensor<'a>, &'a QuantizedTensor<'a>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvQ8SwiGlu)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulGemvQ8SwiGlu",
                dtype: T::DTYPE,
            });
        }
        let (x, (w_gate, w_up), (bias_gate, bias_up)) = args;
        let dims = x.dims();
        // X must be [1, K] or [B, K] but we only support batch 1 for now really efficiently
        // Actually the kernel supports any batch if we iterate Z, but grid is set for 1.
        // We enforce [1, K] or flat K.
        let k = dims.last().copied().unwrap_or(0);

        // Helper to canonicalize
        let canonize = |qt: &QuantizedTensor| -> Result<CanonicalQuantTensor, MetalError> {
            match qt {
                QuantizedTensor::Q8_0(q8) => CanonicalQuantTensor::from_split_q8_tensor(q8),
            }
        };
        let w_gate = canonize(w_gate)?;
        let w_up = canonize(w_up)?;

        // Validate shapes
        if w_gate.logical_dims.len() != 2 || w_up.logical_dims.len() != 2 {
            return Err(MetalError::InvalidShape("Quant tensors must be 2D".into()));
        }
        if w_gate.logical_dims[0] != k || w_up.logical_dims[0] != k {
            return Err(MetalError::InvalidShape("K mismatch for SwiGLU".into()));
        }
        let n_gate = w_gate.logical_dims[1];
        let n_up = w_up.logical_dims[1];

        if n_gate != n_up {
            return Err(MetalError::InvalidShape("Gate and Up dims must match for SwiGLU".into()));
        }

        if let Some(b) = &bias_gate
            && b.len() != n_gate {
                return Err(MetalError::InvalidShape("Gate bias len mismatch".into()));
            }
        if let Some(b) = &bias_up
            && b.len() != n_up {
                return Err(MetalError::InvalidShape("Up bias len mismatch".into()));
            }

        // Output size: [1, N] (rank 2 required by MatmulGemvOp downstream)
        let y = Tensor::new(vec![1, n_gate], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        // Prepare tensors
        let mut inputs: Vec<&Tensor<T>> = vec![x, &y];
        if let Some(b) = &bias_gate {
            inputs.push(b);
        }
        if let Some(b) = &bias_up {
            inputs.push(b);
        }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = Q2Params {
            k: k as u32,
            n0: n_gate as u32,
            n1: n_up as u32,
            blocks_per_k: w_gate.blocks_per_k as u32,
            weights_per_block: w_gate.weights_per_block as u32,
            has_bias0: (bias_gate.is_some() as u32),
            has_bias1: (bias_up.is_some() as u32),
        };

        // Grid setup
        // SIMD-Parallel: 128 threads (4 Warps) process 4 output columns (N) per ThreadGroup
        // Logic: logical_col < N.
        // Logic: logical_col < N.
        let tile_n = 4;
        let col_groups = n_gate.div_ceil(tile_n);

        let grid = MTLSize {
            width: col_groups,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let mut label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope().unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_q8_swiglu"))
        } else {
            GpuProfilerLabel::fallback("gemv_q8_swiglu")
        };
        label.op_name = format!("{}/matmul/gemv_q8_swiglu (Q)", label.op_name);
        label.backend = "gemv_q8_swiglu".to_string();

        let pipeline = pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("MatmulGemvQ8SwiGluOp".to_string()))?;

        let op = SwiGluQ8Op {
            pipeline,
            x: x.clone(),
            w_gate,
            w_up,
            bias_gate: bias_gate.cloned(),
            bias_up: bias_up.cloned(),
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
        };

        Ok((Box::new(op), y))
    }
}

impl DefaultKernelInvocable for MatmulGemvQ8SwiGluRmsnormOp {
    type Args<'a, T: TensorElement> = (
        &'a Tensor<T>,
        &'a Tensor<T>,
        (&'a QuantizedTensor<'a>, &'a QuantizedTensor<'a>),
        (Option<&'a Tensor<T>>, Option<&'a Tensor<T>>),
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::MatmulGemvQ8SwiGluRmsnorm)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut crate::Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut crate::caching::ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        if T::DTYPE != Dtype::F16 {
            return Err(MetalError::UnsupportedDtype {
                operation: "MatmulGemvQ8SwiGluRmsnorm",
                dtype: T::DTYPE,
            });
        }
        let (x, gamma, (w_gate, w_up), (bias_gate, bias_up)) = args;
        let dims = x.dims();
        let k = dims.last().copied().unwrap_or(0);
        if gamma.dims() != [k] {
            return Err(MetalError::InvalidShape("Gamma shape does not match K".into()));
        }

        let canonize = |qt: &QuantizedTensor| -> Result<CanonicalQuantTensor, MetalError> {
            match qt {
                QuantizedTensor::Q8_0(q8) => CanonicalQuantTensor::from_split_q8_tensor(q8),
            }
        };
        let w_gate = canonize(w_gate)?;
        let w_up = canonize(w_up)?;

        if w_gate.logical_dims.len() != 2 || w_up.logical_dims.len() != 2 {
            return Err(MetalError::InvalidShape("Quant tensors must be 2D".into()));
        }
        if w_gate.logical_dims[0] != k || w_up.logical_dims[0] != k {
            return Err(MetalError::InvalidShape("K mismatch for SwiGLU".into()));
        }
        let n_gate = w_gate.logical_dims[1];
        let n_up = w_up.logical_dims[1];
        if n_gate != n_up {
            return Err(MetalError::InvalidShape("Gate and Up dims must match for SwiGLU".into()));
        }
        if let Some(b) = &bias_gate
            && b.len() != n_gate {
                return Err(MetalError::InvalidShape("Gate bias len mismatch".into()));
            }
        if let Some(b) = &bias_up
            && b.len() != n_up {
                return Err(MetalError::InvalidShape("Up bias len mismatch".into()));
            }

        let y = Tensor::new(vec![1, n_gate], TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        let mut inputs: Vec<&Tensor<T>> = vec![x, gamma, &y];
        if let Some(b) = &bias_gate {
            inputs.push(b);
        }
        if let Some(b) = &bias_up {
            inputs.push(b);
        }
        ctx.prepare_tensors_for_active_cmd(&inputs)?;

        let params = Q2Params {
            k: k as u32,
            n0: n_gate as u32,
            n1: n_up as u32,
            blocks_per_k: w_gate.blocks_per_k as u32,
            weights_per_block: w_gate.weights_per_block as u32,
            has_bias0: (bias_gate.is_some() as u32),
            has_bias1: (bias_up.is_some() as u32),
        };

        let tile_n = 4;
        let col_groups = n_gate.div_ceil(tile_n);
        let grid = MTLSize {
            width: col_groups,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        };

        let mut label = if crate::profiling_state::get_profiling_state() {
            ctx.take_gpu_scope()
                .unwrap_or_else(|| GpuProfilerLabel::fallback("gemv_q8_swiglu_rmsnorm"))
        } else {
            GpuProfilerLabel::fallback("gemv_q8_swiglu_rmsnorm")
        };
        label.op_name = format!("{}/matmul/gemv_q8_swiglu_rmsnorm (Q)", label.op_name);
        label.backend = "gemv_q8_swiglu_rmsnorm".to_string();

        let pipeline = pipeline.ok_or_else(|| MetalError::PipelineCreationFailed("MatmulGemvQ8SwiGluRmsnormOp".to_string()))?;

        let op = SwiGluQ8RmsnormOp {
            pipeline,
            x: x.clone(),
            gamma: gamma.clone(),
            w_gate,
            w_up,
            bias_gate: bias_gate.cloned(),
            bias_up: bias_up.cloned(),
            y: y.clone(),
            params,
            grid,
            tg,
            profiler_label: label,
        };

        Ok((Box::new(op), y))
    }
}
