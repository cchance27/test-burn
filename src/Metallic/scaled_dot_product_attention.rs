use super::{
    Context, MetalError, Tensor,
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
};
use crate::metallic::CommandBuffer;
use crate::metallic::matmul::MatMulOperation;
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::softmax::{SoftmaxOperation, ensure_fused_softmax_pipeline};
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLComputePipelineState, MTLDevice};

impl Context {
    pub fn scaled_dot_product_attention(
        &mut self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor, MetalError> {
        self.pool.reset();

        autoreleasepool(|_| {
            // Validate input tensors for numerical stability
            q.validate_numerical_stability()?;
            k.validate_numerical_stability()?;
            v.validate_numerical_stability()?;

            let b = q.dims[0];
            let s_q = q.dims[1];
            let s_k = k.dims[1];
            let d = q.dims[2];

            let out = self.pool.alloc_tensor(vec![b, s_q, d])?;
            let attn = self.pool.alloc_tensor(vec![b, s_q, s_k])?;

            // Create a local cache for this operation
            let mut cache = ResourceCache::new();

            // Get or create the SDPA operation
            let sdpa_op = cache.get_or_create_sdpa(b, s_q, s_k, d);

            // Get the fused softmax pipeline
            ensure_fused_softmax_pipeline(self)?;
            let softmax_pipeline = self.fused_softmax_pipeline.as_ref().unwrap().clone();

            scaled_dot_product_attention_impl(
                q,
                k,
                v,
                causal,
                &mut cache,
                &self.device,
                &self.command_queue,
                &softmax_pipeline,
                sdpa_op.scale,
                &out,
                &attn,
            )
        })
    }
}

/// Standalone implementation of scaled dot product attention that doesn't depend on Context.
///
/// This function can be used independently of the Context struct, allowing for better
/// decoupling and more flexible usage patterns.
#[allow(clippy::too_many_arguments)]
pub fn scaled_dot_product_attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    cache: &mut ResourceCache,
    device: &Retained<ProtocolObject<dyn MTLDevice>>,
    command_queue: &Retained<ProtocolObject<dyn MTLCommandQueue>>,
    softmax_pipeline: &Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    scale: f32,
    out: &Tensor,
    attn: &Tensor,
) -> Result<Tensor, MetalError> {
    // Validate dimensions
    if q.dims.len() != 3 || k.dims.len() != 3 || v.dims.len() != 3 {
        return Err(MetalError::InvalidShape(
            "SDPA requires 3D tensors".to_string(),
        ));
    }

    let b = q.dims[0];
    let s_q = q.dims[1];
    let s_k = k.dims[1];
    let d = q.dims[2];

    // Check batch dimension compatibility
    if b != k.dims[0] || b != v.dims[0] {
        return Err(MetalError::DimensionMismatch {
            expected: b,
            actual: k.dims[0].max(v.dims[0]),
        });
    }

    // Check feature dimension compatibility
    if d != k.dims[2] {
        return Err(MetalError::DimensionMismatch {
            expected: d,
            actual: k.dims[2],
        });
    }

    // Check value tensor compatibility
    if s_k != v.dims[1] || d != v.dims[2] {
        return Err(MetalError::DimensionMismatch {
            expected: s_k * d,
            actual: v.dims[1] * v.dims[2],
        });
    }

    // Get cached GEMM operations
    let qk_gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: true,
        result_rows: s_q,
        result_columns: s_k,
        interior_columns: d,
        alpha: scale,
        beta: 0.0,
    };
    let _qk_gemm_op = cache.get_or_create_gemm(qk_gemm_key.clone(), device)?;

    let out_gemm_key = MpsGemmKey {
        transpose_left: false,
        transpose_right: false,
        result_rows: s_q,
        result_columns: d,
        interior_columns: s_k,
        alpha: 1.0,
        beta: 0.0,
    };
    let _out_gemm_op = cache.get_or_create_gemm(out_gemm_key.clone(), device)?;

    // Get cached matrix descriptors
    let bytes_per_elem: usize = core::mem::size_of::<f32>();
    let row_bytes_feat = d * bytes_per_elem;
    let row_bytes_attn = s_k * bytes_per_elem;

    let desc_q_key = MpsMatrixDescriptorKey {
        rows: s_q,
        columns: d,
        row_bytes: row_bytes_feat,
    };
    let desc_q = cache.get_or_create_descriptor(desc_q_key, device)?;

    let desc_k_key = MpsMatrixDescriptorKey {
        rows: s_k,
        columns: d,
        row_bytes: row_bytes_feat,
    };
    let desc_k = cache.get_or_create_descriptor(desc_k_key, device)?;

    let desc_v_key = MpsMatrixDescriptorKey {
        rows: s_k,
        columns: d,
        row_bytes: row_bytes_feat,
    };
    let desc_v = cache.get_or_create_descriptor(desc_v_key, device)?;

    let desc_out_key = MpsMatrixDescriptorKey {
        rows: s_q,
        columns: d,
        row_bytes: row_bytes_feat,
    };
    let desc_out = cache.get_or_create_descriptor(desc_out_key, device)?;

    let desc_attn_key = MpsMatrixDescriptorKey {
        rows: s_q,
        columns: s_k,
        row_bytes: row_bytes_attn,
    };
    let desc_attn = cache.get_or_create_descriptor(desc_attn_key, device)?;

    let mut command_buffers: Vec<CommandBuffer> = Vec::with_capacity(b);

    for i in 0..b {
        let mut cmd = CommandBuffer::new(command_queue)?;

        let q_i = q.get_batch(i)?;
        let k_i = k.get_batch(i)?;
        let v_i = v.get_batch(i)?;
        let attn_i = attn.get_batch(i)?;
        let out_i = out.get_batch(i)?;

        // Q x K^T -> attn
        let qk_gemm = cache.get_or_create_gemm(qk_gemm_key.clone(), device)?;
        let qk_op = MatMulOperation {
            left_buf: q_i.buf.clone(),
            left_offset: q_i.offset,
            right_buf: k_i.buf.clone(),
            right_offset: k_i.offset,
            result_buf: attn_i.buf.clone(),
            result_offset: attn_i.offset,
            left_desc: desc_q.clone(),
            right_desc: desc_k.clone(),
            result_desc: desc_attn.clone(),
            gemm: qk_gemm,
        };
        cmd.record(&qk_op, cache)?;

        // Softmax(attn)
        let sm_op = SoftmaxOperation {
            attn_buf: attn_i.buf.clone(),
            attn_offset: attn_i.offset,
            seq_q: s_q as u32,
            seq_k: s_k as u32,
            causal: causal as u32,
            pipeline: softmax_pipeline.clone(),
        };
        cmd.record(&sm_op, cache)?;

        // attn x V -> out
        let out_gemm = cache.get_or_create_gemm(out_gemm_key.clone(), device)?;
        let out_op = MatMulOperation {
            left_buf: attn_i.buf.clone(),
            left_offset: attn_i.offset,
            right_buf: v_i.buf.clone(),
            right_offset: v_i.offset,
            result_buf: out_i.buf.clone(),
            result_offset: out_i.offset,
            left_desc: desc_attn.clone(),
            right_desc: desc_v.clone(),
            result_desc: desc_out.clone(),
            gemm: out_gemm,
        };
        cmd.record(&out_op, cache)?;

        cmd.commit();
        command_buffers.push(cmd);
    }

    for cb in &command_buffers {
        cb.wait();
    }

    Ok(out.clone())
}
