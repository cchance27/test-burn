use super::{
    cache_keys::{MpsGemmKey, MpsMatrixDescriptorKey},
    Context, MetalError, Tensor,
};
use crate::metallic::softmax::{ensure_fused_softmax_pipeline, SoftmaxOperation};
use crate::metallic::matmul::MatMulOperation;
use crate::metallic::resource_cache::ResourceCache;
use crate::metallic::CommandBuffer;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandQueue, MTLDevice, MTLComputePipelineState,
};

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
                q, k, v, causal,
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
    let b = q.dims[0];
    let s_q = q.dims[1];
    let s_k = k.dims[1];
    let d = q.dims[2];

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

#[cfg(test)]
mod tests {
    const PYTORCH_ARANGE_NONCAUSAL: [f32; 256] = [
        112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
        125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
        122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
        119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0,
        116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0,
        113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0,
        126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
        123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
        120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0,
        117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 240.0, 241.0,
        242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0,
        255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0,
        252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0,
        249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0,
        246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0,
        243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0,
        240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0,
        253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0,
        250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0,
        247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0,
    ];

    const DIMENSIONS: [usize; 3] = [2, 8, 16];
    const NUM_ELEMENTS: i64 = 2 * 8 * 16;

    #[test]
    fn arange_sdpa_burn_vs_pytorch_causal() {
        use burn::prelude::*;
        type MyBackend = burn::backend::Metal;
        let pytorch_arange_causal = (0..256).map(|x| x as f32).collect::<Vec<_>>();

        let device = <MyBackend as Backend>::Device::default();
        let query = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
            .float()
            .reshape(DIMENSIONS);
        let key = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
            .float()
            .reshape(DIMENSIONS);
        let value = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
            .float()
            .reshape(DIMENSIONS);
        let output =
            crate::sdpa_burn::scaled_dot_product_attention_burn(query, key, value, None, true);
        assert_eq!(output.dims(), DIMENSIONS);
        assert_eq!(
            output.to_data().as_slice::<f32>().unwrap(),
            &pytorch_arange_causal
        );
    }

    #[test]
    fn arange_sdpa_ours_vs_pytorch_causal() {
        use crate::metallic::{Context, Tensor};
        let pytorch_arange_causal = (0..256).map(|x| x as f32).collect::<Vec<_>>();

        let mut context = Context::new().unwrap();
        let query: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
        let key: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
        let value: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();

        // Clone the device so the created tensors don't hold an immutable borrow on `context`
        let q_tensor = Tensor::create_tensor_from_slice(&query, vec![2, 8, 16], &context).unwrap();
        let k_tensor = Tensor::create_tensor_from_slice(&key, vec![2, 8, 16], &context).unwrap();
        let v_tensor = Tensor::create_tensor_from_slice(&value, vec![2, 8, 16], &context).unwrap();

        let output = context
            .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)
            .unwrap();
        assert_eq!(output.dims(), DIMENSIONS);
        assert_eq!(output.as_slice(), &pytorch_arange_causal);
    }

    #[test]
    fn arange_sdpa_burn_vs_pytorch_noncausal() {
        use burn::prelude::*;
        type MyBackend = burn::backend::Metal;

        let device = <MyBackend as Backend>::Device::default();
        let query = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
            .float()
            .reshape(DIMENSIONS);
        let key = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
            .float()
            .reshape(DIMENSIONS);
        let value = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
            .float()
            .reshape(DIMENSIONS);
        let output =
            crate::sdpa_burn::scaled_dot_product_attention_burn(query, key, value, None, false);
        assert_eq!(output.dims(), DIMENSIONS);
        assert_eq!(
            output.to_data().as_slice::<f32>().unwrap(),
            &PYTORCH_ARANGE_NONCAUSAL
        );
    }

    #[test]
    fn arange_sdpa_ours_vs_pytorch_noncausal() {
        use crate::metallic::{Context, Tensor};

        let mut context = Context::new().unwrap();
        let query: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
        let key: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
        let value: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();

        // Clone the device so the created tensors don't hold an immutable borrow on `context`
        let q_tensor = Tensor::create_tensor_from_slice(&query, vec![2, 8, 16], &context).unwrap();
        let k_tensor = Tensor::create_tensor_from_slice(&key, vec![2, 8, 16], &context).unwrap();
        let v_tensor = Tensor::create_tensor_from_slice(&value, vec![2, 8, 16], &context).unwrap();

        let output = context
            .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)
            .unwrap();
        assert_eq!(output.dims(), DIMENSIONS);
        assert_eq!(output.as_slice(), &PYTORCH_ARANGE_NONCAUSAL);
    }

    #[test]
    fn large_sdpa_ours_vs_burn_causal() {
        use burn::prelude::*;
        use burn::tensor::{Int, Tensor as BurnTensor};
        type MyBackend = burn::backend::Metal;

        let device = <MyBackend as Backend>::Device::default();
        let batch: usize = 1;
        let seq_q: usize = 64;
        let seq_k: usize = 1024;
        let dim: usize = 64;
        let q_num = batch * seq_q * dim;
        let kv_num = batch * seq_k * dim;

        let q_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(q_num as i64), &device)
            .float()
            .reshape(burn::tensor::Shape::from([batch, seq_q, dim]));
        let k_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(kv_num as i64), &device)
            .float()
            .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));
        let v_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(
            (kv_num as i64)..(2 * kv_num as i64),
            &device,
        )
        .float()
        .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));

        let q_data_tensor = q_burn_input.to_data();
        let q_data = q_data_tensor.as_slice::<f32>().unwrap().to_vec();
        let k_data_tensor = k_burn_input.to_data();
        let k_data = k_data_tensor.as_slice::<f32>().unwrap().to_vec();
        let v_data_tensor = v_burn_input.to_data();
        let v_data = v_data_tensor.as_slice::<f32>().unwrap().to_vec();

        let burn_out = crate::sdpa_burn::scaled_dot_product_attention_burn(
            q_burn_input,
            k_burn_input,
            v_burn_input,
            None,
            true,
        );
        let burn_data = burn_out.to_data();
        let burn_slice = burn_data.as_slice::<f32>().unwrap();

        // Metallic
        use crate::metallic::{Context, Tensor};
        let mut ctx = Context::new().unwrap();
        let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
        let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
        let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &ctx).unwrap();
        let metal_out = ctx
            .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)
            .unwrap();
        let metal_slice = metal_out.as_slice();

        // Validate with tolerance due to FP reductions
        let rtol = 1e-4f64;
        let atol = 1e-6f64;
        for (i, (metal_val, burn_val)) in metal_slice.iter().zip(burn_slice.iter()).enumerate() {
            let diff = ((*metal_val) as f64 - (*burn_val) as f64).abs();
            let rel_err = if burn_val.abs() > 1e-8 {
                diff / ((*burn_val).abs() as f64)
            } else {
                diff
            };
            assert!(
                diff <= atol || rel_err <= rtol,
                "Mismatch at index {}: metal={:.6}, burn={:.6}, diff={:.2e}, rel={:.2e}",
                i, metal_val, burn_val, diff, rel_err
            );
        }
    }

    #[test]
    fn large_sdpa_ours_vs_burn_noncausal() {
        use burn::prelude::*;
        use burn::tensor::{Int, Tensor as BurnTensor};
        type MyBackend = burn::backend::Metal;

        let device = <MyBackend as Backend>::Device::default();
        let batch: usize = 1;
        let seq_q: usize = 64;
        let seq_k: usize = 1024;
        let dim: usize = 64;
        let q_num = batch * seq_q * dim;
        let kv_num = batch * seq_k * dim;

        let q_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(q_num as i64), &device)
            .float()
            .reshape(burn::tensor::Shape::from([batch, seq_q, dim]));
        let k_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(kv_num as i64), &device)
            .float()
            .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));
        let v_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(
            (kv_num as i64)..(2 * kv_num as i64),
            &device,
        )
        .float()
        .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));

        let q_data_tensor = q_burn_input.to_data();
        let q_data = q_data_tensor.as_slice::<f32>().unwrap().to_vec();
        let k_data_tensor = k_burn_input.to_data();
        let k_data = k_data_tensor.as_slice::<f32>().unwrap().to_vec();
        let v_data_tensor = v_burn_input.to_data();
        let v_data = v_data_tensor.as_slice::<f32>().unwrap().to_vec();

        let burn_out = crate::sdpa_burn::scaled_dot_product_attention_burn(
            q_burn_input,
            k_burn_input,
            v_burn_input,
            None,
            false,
        );
        let burn_data = burn_out.to_data();
        let burn_slice = burn_data.as_slice::<f32>().unwrap();

        // Metallic
        use crate::metallic::{Context, Tensor};
        let mut ctx = Context::new().unwrap();
        let q_tensor = Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
        let k_tensor = Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
        let v_tensor = Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &ctx).unwrap();
        let metal_out = ctx
            .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)
            .unwrap();
        let metal_slice = metal_out.as_slice();

        // Validate with tolerance due to FP reductions
        let rtol = 1e-4f64;
        let atol = 1e-6f64;
        for (i, (metal_val, burn_val)) in metal_slice.iter().zip(burn_slice.iter()).enumerate() {
            let diff = ((*metal_val) as f64 - (*burn_val) as f64).abs();
            let rel_err = if burn_val.abs() > 1e-8 {
                diff / ((*burn_val).abs() as f64)
            } else {
                diff
            };
            assert!(
                diff <= atol || rel_err <= rtol,
                "Mismatch at index {}: metal={:.6}, burn={:.6}, diff={:.2e}, rel={:.2e}",
                i, metal_val, burn_val, diff, rel_err
            );
        }
    }
}
