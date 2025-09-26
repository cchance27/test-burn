use super::{Context, MetalError, Tensor};

impl Context {
    pub fn scaled_dot_product_attention(&mut self, q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor, MetalError> {
        // Validate dimensions
        if q.dims().len() != 3 || k.dims().len() != 3 || v.dims().len() != 3 {
            return Err(MetalError::InvalidShape("SDPA requires 3D tensors".to_string()));
        }

        let b = q.dims()[0];
        let s_q = q.dims()[1];
        let s_k = k.dims()[1];
        let d = q.dims()[2];

        // Check batch dimension compatibility
        if b != k.dims()[0] || b != v.dims()[0] {
            return Err(MetalError::DimensionMismatch {
                expected: b,
                actual: k.dims()[0].max(v.dims()[0]),
            });
        }

        // Check feature dimension compatibility
        if d != k.dims()[2] {
            return Err(MetalError::DimensionMismatch {
                expected: d,
                actual: k.dims()[2],
            });
        }

        // Check value tensor compatibility
        if s_k != v.dims()[1] || d != v.dims()[2] {
            return Err(MetalError::DimensionMismatch {
                expected: s_k * d,
                actual: v.dims()[1] * v.dims()[2],
            });
        }

        // Calculate scale factor
        let scale = 1.0 / (d as f32).sqrt();

        // Create output tensors
        let out = Tensor::create_tensor_pooled(vec![b, s_q, d], self)?;
        let attn = Tensor::create_tensor_pooled(vec![b, s_q, s_k], self)?;

        // Process each batch separately to work with 2D matmul operations
        for i in 0..b {
            // Get batch slices for each tensor
            let q_i = q.get_batch(i)?; // [s_q, d]
            let k_i = k.get_batch(i)?; // [s_k, d]
            let v_i = v.get_batch(i)?; // [s_k, d]
            let mut out_i = out.get_batch(i)?; // [s_q, d]
            let mut attn_i = attn.get_batch(i)?; // [s_q, s_k]

            // Q x K^T -> attn (for this batch)
            // Get attention slice to write to: [s_q, s_k]
            // Note: k.get_batch(i) gives [s_k, d], need to transpose to [d, s_k]
            let k_i_t = k_i.permute(&[1, 0], self)?; // [d, s_k]

            // Perform matmul: [s_q, d] @ [d, s_k] = [s_q, s_k]
            // The result should be stored in attn_i
            // Since matmul creates a new tensor, we need a different approach
            let qk_result = self.matmul(&q_i, &k_i_t, false, false)?; // [s_q, s_k]

            // Apply scale using element-wise multiplication to the attention matrix
            // Create a scale tensor with the same shape as attention [s_q, s_k]
            let scale_values = vec![scale; s_q * s_k]; // Fill with scale value for each element
            let scale_tensor = Tensor::create_tensor_from_slice(&scale_values, vec![s_q, s_k], self)?;
            // TODO: replace per-call scale tensor with a matmul alpha or cached buffer.
            let scaled_attn = self.call::<crate::metallic::kernels::elemwise_mul::ElemwiseMulOp>((qk_result, scale_tensor))?;

            // Apply softmax to the scaled attention
            let softmax_result =
                self.call::<crate::metallic::kernels::softmax::SoftmaxOp>((scaled_attn, s_q as u32, s_k as u32, causal as u32))?;

            // attn x V -> out (for this batch)
            // [s_q, s_k] @ [s_k, d] = [s_q, d]
            let final_result = self.matmul(&softmax_result, &v_i, false, false)?; // [s_q, d]

            // Ensure GPU work is complete before touching host-visible buffers.
            self.synchronize();

            // TODO: replace this host copy with a matmul variant that writes directly into `out_i`.
            let final_slice = final_result.as_slice();
            let out_slice = out_i.as_mut_slice();
            out_slice.copy_from_slice(final_slice);

            // Keep attention buffer warm for future in-place pipeline once we avoid the copy path.
            attn_i.as_mut_slice().copy_from_slice(softmax_result.as_slice());
        }

        self.synchronize();
        Ok(out)
    }
}
