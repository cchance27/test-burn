use super::*;
use crate::metallic::TensorElement;

pub struct FusedQkvOp;

struct FusedQkv<T: TensorElement> {
    fused: Tensor<T>,
    q_out: Tensor<T>,
    k_out: Tensor<T>,
    v_out: Tensor<T>,
    cos: Tensor<T>,
    sin: Tensor<T>,
    row_stride: u32,
    d_model: u32,
    kv_dim: u32,
    head_dim: u32,
    kv_head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq: u32,
    apply_rope: u32,
    position_offset: u32,
    total_q: u32,
    total_k: u32,
    total_v: u32,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for FusedQkvOp {
    #[allow(clippy::type_complexity)]
    type Args<'a, T: TensorElement> = (
        Tensor<T>,
        Tensor<T>,
        Tensor<T>,
        Tensor<T>,
        Option<Tensor<T>>,
        Option<Tensor<T>>,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::FusedQkv)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (
            fused,
            q_out,
            k_out,
            v_out,
            cos,
            sin,
            row_stride,
            d_model,
            kv_dim,
            head_dim,
            kv_head_dim,
            n_heads,
            n_kv_heads,
            seq,
            position_offset,
        ) = args;

        if n_heads == 0 || n_kv_heads == 0 {
            return Err(MetalError::InvalidShape("fused_qkv requires non-zero head counts".to_string()));
        }
        if head_dim == 0 || kv_head_dim == 0 {
            return Err(MetalError::InvalidShape("fused_qkv requires non-zero head dimensions".to_string()));
        }

        let (cos_tensor, sin_tensor, apply_rope) = match (cos, sin) {
            (Some(cos), Some(sin)) => (cos, sin, 1),
            _ => (q_out.clone(), q_out.clone(), 0),
        };

        let tensors: Vec<&Tensor<T>> = if apply_rope == 1 {
            vec![&fused, &q_out, &k_out, &v_out, &cos_tensor, &sin_tensor]
        } else {
            vec![&fused, &q_out, &k_out, &v_out]
        };
        ctx.prepare_tensors_for_active_cmd(&tensors)?;

        let total_q = q_out.len() as u32;
        let total_k = k_out.len() as u32;
        let total_v = v_out.len() as u32;

        let op = FusedQkv {
            fused,
            q_out: q_out.clone(),
            k_out: k_out.clone(),
            v_out: v_out.clone(),
            cos: cos_tensor.clone(),
            sin: sin_tensor.clone(),
            row_stride,
            d_model,
            kv_dim,
            head_dim,
            kv_head_dim,
            n_heads,
            n_kv_heads,
            seq,
            apply_rope,
            position_offset,
            total_q,
            total_k,
            total_v,
            pipeline: pipeline.expect("Kernel Library supplied for MetalKernels"),
        };

        Ok((Box::new(op), q_out))
    }
}

impl<T: TensorElement> Operation for FusedQkv<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let total = self.total_q + self.total_k + self.total_v;
        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: total.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.fused.buf, self.fused.offset);
        set_buffer(&encoder, 1, &self.q_out.buf, self.q_out.offset);
        set_buffer(&encoder, 2, &self.k_out.buf, self.k_out.offset);
        set_buffer(&encoder, 3, &self.v_out.buf, self.v_out.offset);
        set_buffer(&encoder, 4, &self.cos.buf, self.cos.offset);
        set_buffer(&encoder, 5, &self.sin.buf, self.sin.offset);
        set_bytes(&encoder, 6, &self.row_stride);
        set_bytes(&encoder, 7, &self.d_model);
        set_bytes(&encoder, 8, &self.kv_dim);
        set_bytes(&encoder, 9, &self.head_dim);
        set_bytes(&encoder, 10, &self.kv_head_dim);
        set_bytes(&encoder, 11, &self.n_heads);
        set_bytes(&encoder, 12, &self.n_kv_heads);
        set_bytes(&encoder, 13, &self.seq);
        set_bytes(&encoder, 14, &self.apply_rope);
        set_bytes(&encoder, 15, &self.position_offset);
        set_bytes(&encoder, 16, &self.total_q);
        set_bytes(&encoder, 17, &self.total_k);
        set_bytes(&encoder, 18, &self.total_v);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
