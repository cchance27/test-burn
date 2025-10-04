use super::*;
use crate::metallic::TensorElement;

pub struct KvCacheWriteOp;

#[derive(Clone, Debug)]
pub struct KvCacheWriteConfig {
    pub canonical_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub step: u32,
    pub group_size: u32,
    pub src_head_stride: u32,
    pub src_seq_stride: u32,
    pub dst_head_stride: u32,
    pub dst_seq_stride: u32,
    pub total_threads: u32,
    pub repeated_head_stride: u32,
    pub repeated_seq_stride: u32,
    pub has_repeated: u32,
}

struct KvCacheWrite<T: TensorElement> {
    k_src: Tensor<T>,
    v_src: Tensor<T>,
    k_dst: Tensor<T>,
    v_dst: Tensor<T>,
    repeated_k_dst: Option<Tensor<T>>,
    repeated_v_dst: Option<Tensor<T>>,
    params: KvCacheWriteConfig,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelInvocable for KvCacheWriteOp {
    type Args<'a, T: TensorElement> = (
        Tensor<T>,
        Tensor<T>,
        Tensor<T>,
        Tensor<T>,
        Option<(Tensor<T>, Tensor<T>)>,
        KvCacheWriteConfig,
    );

    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::KvCacheWrite)
    }

    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (k_src, v_src, k_dst, v_dst, repeated, params) = args;

        if params.canonical_heads == 0 {
            return Err(MetalError::InvalidShape(
                "kv_cache_write requires a positive head count".to_string(),
            ));
        }
        if params.head_dim == 0 {
            return Err(MetalError::InvalidShape(
                "kv_cache_write requires a positive head dimension".to_string(),
            ));
        }
        if params.seq_len == 0 {
            return Err(MetalError::InvalidShape(
                "kv_cache_write expects at least one active sequence element".to_string(),
            ));
        }
        if params.group_size == 0 {
            return Err(MetalError::InvalidShape(
                "kv_cache_write requires a non-zero group size".to_string(),
            ));
        }

        if params.has_repeated > 1 {
            return Err(MetalError::InvalidShape(
                "kv_cache_write expects has_repeated flag to be 0 or 1".to_string(),
            ));
        }

        let mut tensors: Vec<&Tensor<T>> = vec![&k_src, &v_src, &k_dst, &v_dst];
        if let Some((ref repeated_k, ref repeated_v)) = repeated {
            tensors.push(repeated_k);
            tensors.push(repeated_v);
        }

        ctx.prepare_tensors_for_active_cmd(&tensors)?;

        let pipeline = pipeline.expect("Kernel Library supplied for MetalKernels");
        let op = KvCacheWrite {
            k_src: k_src.clone(),
            v_src,
            k_dst: k_dst.clone(),
            v_dst,
            repeated_k_dst: repeated.as_ref().map(|(k, _)| k.clone()),
            repeated_v_dst: repeated.map(|(_, v)| v),
            params,
            pipeline,
        };

        Ok((Box::new(op), k_dst))
    }
}

impl<T: TensorElement> Operation for KvCacheWrite<T> {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(MetalError::ComputeEncoderCreationFailed)?;

        let threads_per_tg = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let groups = MTLSize {
            width: self.params.total_threads.div_ceil(256) as usize,
            height: 1,
            depth: 1,
        };

        set_compute_pipeline_state(&encoder, &self.pipeline);
        set_buffer(&encoder, 0, &self.k_src.buf, self.k_src.offset);
        set_buffer(&encoder, 1, &self.v_src.buf, self.v_src.offset);
        set_buffer(&encoder, 2, &self.k_dst.buf, self.k_dst.offset);
        set_buffer(&encoder, 3, &self.v_dst.buf, self.v_dst.offset);
        let repeated_k = self.repeated_k_dst.as_ref().unwrap_or(&self.k_dst);
        let repeated_v = self.repeated_v_dst.as_ref().unwrap_or(&self.v_dst);
        set_buffer(&encoder, 4, &repeated_k.buf, repeated_k.offset);
        set_buffer(&encoder, 5, &repeated_v.buf, repeated_v.offset);
        set_bytes(&encoder, 6, &self.params.canonical_heads);
        set_bytes(&encoder, 7, &self.params.head_dim);
        set_bytes(&encoder, 8, &self.params.seq_len);
        set_bytes(&encoder, 9, &self.params.step);
        set_bytes(&encoder, 10, &self.params.group_size);
        set_bytes(&encoder, 11, &self.params.src_head_stride);
        set_bytes(&encoder, 12, &self.params.src_seq_stride);
        set_bytes(&encoder, 13, &self.params.dst_head_stride);
        set_bytes(&encoder, 14, &self.params.dst_seq_stride);
        set_bytes(&encoder, 15, &self.params.repeated_head_stride);
        set_bytes(&encoder, 16, &self.params.repeated_seq_stride);
        set_bytes(&encoder, 17, &self.params.has_repeated);
        set_bytes(&encoder, 18, &self.params.total_threads);

        dispatch_threadgroups(&encoder, groups, threads_per_tg);
        encoder.endEncoding();
        Ok(())
    }
}
