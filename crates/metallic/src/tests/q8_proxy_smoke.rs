#[cfg(test)]
mod tests {
    use half::f16;

    use crate::{
        Context, MetalError, Tensor, TensorStorage, kernels::{elemwise_add::BroadcastElemwiseAddInplaceOp, matmul_gemv::MatmulGemvOp}, tensor::{
            Q8_0_BLOCK_SIZE_BYTES, Q8_0_WEIGHTS_PER_BLOCK, QuantizedTensor, TensorType, quantized::{Q8_0_SCALE_BYTES_PER_BLOCK, QuantizedQ8_0Tensor, swizzle_q8_0_blocks_nk}
        }
    };

    fn pack_q8_blocks(rows: usize, cols: usize, weights: &[i8]) -> Vec<u8> {
        assert_eq!(weights.len(), rows * cols);
        let blocks_per_row = cols.div_ceil(32);
        let mut out = Vec::with_capacity(rows * blocks_per_row * Q8_0_BLOCK_SIZE_BYTES);

        for row in 0..rows {
            for block_idx in 0..blocks_per_row {
                let start_col = block_idx * 32;
                let mut qs = [0i8; 32];
                for i in 0..32 {
                    let col = start_col + i;
                    qs[i] = if col < cols { weights[row * cols + col] } else { 0 };
                }

                let scale = f16::from_f32(1.0);
                let mut blk = [0u8; 34];
                let sc = scale.to_bits().to_le_bytes();
                blk[0] = sc[0];
                blk[1] = sc[1];
                for i in 0..32 {
                    blk[2 + i] = qs[i] as u8;
                }

                out.extend_from_slice(&blk);
            }
        }
        out
    }

    fn split_blocks(raw: &[u8]) -> (Vec<u8>, Vec<u8>) {
        assert!(raw.len().is_multiple_of(Q8_0_BLOCK_SIZE_BYTES));
        let blocks = raw.len() / Q8_0_BLOCK_SIZE_BYTES;
        let mut data = Vec::with_capacity(blocks * Q8_0_WEIGHTS_PER_BLOCK);
        let mut scales = Vec::with_capacity(blocks * Q8_0_SCALE_BYTES_PER_BLOCK);
        for chunk in raw.chunks_exact(Q8_0_BLOCK_SIZE_BYTES) {
            scales.extend_from_slice(&chunk[0..Q8_0_SCALE_BYTES_PER_BLOCK]);
            data.extend_from_slice(&chunk[Q8_0_SCALE_BYTES_PER_BLOCK..Q8_0_BLOCK_SIZE_BYTES]);
        }
        (data, scales)
    }

    fn transpose_weights(weights_nk: &[i8], n: usize, k: usize) -> Vec<i8> {
        let mut transposed = Vec::with_capacity(weights_nk.len());
        for row in 0..k {
            for col in 0..n {
                transposed.push(weights_nk[col * k + row]);
            }
        }
        transposed
    }

    fn assert_proxy_bias_matches_fp16(
        ctx: &mut Context<crate::tensor::F16>,
        input_dim: usize,
        output_dim: usize,
        weights: &[i8],
        x_f32: &[f32],
        bias_f32: &[f32],
    ) -> Result<(), MetalError> {
        assert_eq!(weights.len(), input_dim * output_dim);
        assert_eq!(x_f32.len(), input_dim);
        assert_eq!(bias_f32.len(), output_dim);

        let weights_nk = transpose_weights(weights, input_dim, output_dim);
        let bytes = pack_q8_blocks(output_dim, input_dim, &weights_nk);
        let swizzled = swizzle_q8_0_blocks_nk(output_dim, input_dim, &bytes).expect("nk swizzle in test");
        let (data_bytes, scale_bytes) = split_blocks(&swizzled);
        let q8 = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![output_dim, input_dim], &data_bytes, &scale_bytes, ctx)?;

        let x_tensor = Tensor::<crate::tensor::F16>::from_f32_slice(vec![1, input_dim], TensorStorage::Pooled(ctx), x_f32)?;
        let bias = Tensor::<crate::tensor::F16>::from_f32_slice(vec![output_dim], TensorStorage::Pooled(ctx), bias_f32)?;

        let q8_y = ctx.call::<MatmulGemvOp>((&x_tensor.clone(), TensorType::Quant(QuantizedTensor::Q8_0(&q8)), None))?;
        let q8_y = ctx.call::<BroadcastElemwiseAddInplaceOp>((q8_y, bias.clone()))?;

        let weights_f32: Vec<f32> = weights.iter().map(|&v| v as f32).collect();
        let weight_tensor =
            Tensor::<crate::tensor::F16>::from_f32_slice(vec![input_dim, output_dim], TensorStorage::Pooled(ctx), &weights_f32)?;
        let fp16_y = ctx.call::<MatmulGemvOp>((&x_tensor, TensorType::Dense(&weight_tensor), None))?;
        let fp16_y = ctx.call::<BroadcastElemwiseAddInplaceOp>((fp16_y, bias))?;

        let q8_slice = q8_y.as_slice();
        let fp16_slice = fp16_y.as_slice();
        assert_eq!(q8_slice.len(), fp16_slice.len());

        for (idx, (&q8_val, &fp16_val)) in q8_slice.iter().zip(fp16_slice.iter()).enumerate() {
            let q8_float = q8_val.to_f32();
            let fp16_float = fp16_val.to_f32();
            assert!(
                (q8_float - fp16_float).abs() < 1e-2,
                "Quantized proxy + bias differs from FP16 matmul at {}; {} vs {}",
                idx,
                q8_float,
                fp16_float
            );
        }

        Ok(())
    }

    fn cpu_gemv(n: usize, k: usize, weights: &[i8], x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for row in 0..n {
            let mut acc = 0.0f32;
            for col in 0..k {
                acc += (weights[row * k + col] as f32) * x[col];
            }
            out[row] = acc;
        }
        out
    }

    #[test]
    fn q8_proxy_smoke() -> Result<(), MetalError> {
        let n = 8;
        let k = 16;
        let mut ctx = Context::<crate::tensor::F16>::new()?;

        let mut weights = Vec::with_capacity(n * k);
        for r in 0..n {
            for c in 0..k {
                weights.push(((r * 3 + c) % 7) as i8 - 3);
            }
        }

        let bytes = pack_q8_blocks(n, k, &weights);
        let swizzled = swizzle_q8_0_blocks_nk(n, k, &bytes).expect("nk swizzle in smoke test");
        let (data_bytes, scale_bytes) = split_blocks(&swizzled);
        let q8 = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![n, k], &data_bytes, &scale_bytes, &ctx)?;

        let x_f32: Vec<f32> = (0..k).map(|i| ((i % 5) as f32) - 1.5).collect();
        let x = Tensor::<crate::tensor::F16>::from_f32_slice(vec![1, k], TensorStorage::Pooled(&mut ctx), &x_f32)?;

        let q8_result = ctx.call::<MatmulGemvOp>((&x, TensorType::Quant(QuantizedTensor::Q8_0(&q8)), None))?;

        let y_ref = cpu_gemv(n, k, &weights, &x_f32);
        for (i, &v) in q8_result.as_slice().iter().enumerate() {
            let val = v.to_f32();
            let expected = y_ref[i];
            assert!(
                (val - expected).abs() < 1e-3,
                "Proxy GEMV mismatch at {}: {} vs {}",
                i,
                val,
                expected
            );
        }

        Ok(())
    }

    #[test]
    fn q8_proxy_bias_matches_fp16_matmul() -> Result<(), MetalError> {
        let mut ctx = Context::<crate::tensor::F16>::new()?;
        let input_dim = 63;
        let output_dim = 47;
        let mut weights = Vec::with_capacity(input_dim * output_dim);
        for row in 0..input_dim {
            for col in 0..output_dim {
                weights.push(((row * 7 + col * 3) % 9) as i8 - 4);
            }
        }

        let x_f32: Vec<f32> = (0..input_dim).map(|i| ((i * 13) % 17) as f32 * 0.08 - 0.65).collect();
        let bias_f32: Vec<f32> = (0..output_dim).map(|i| ((i * 5) % 11) as f32 * 0.04 - 0.3).collect();

        assert_proxy_bias_matches_fp16(&mut ctx, input_dim, output_dim, &weights, &x_f32, &bias_f32)
    }

    #[test]
    fn q8_proxy_k_bias_matches_fp16_matmul() -> Result<(), MetalError> {
        let mut ctx = Context::<crate::tensor::F16>::new()?;

        let input_dim = 64;
        let output_dim = 40;
        let mut weights = Vec::with_capacity(input_dim * output_dim);
        for row in 0..input_dim {
            for col in 0..output_dim {
                weights.push(((row * 5 + col * 7) % 13) as i8 - 6);
            }
        }
        let x_f32: Vec<f32> = (0..input_dim).map(|i| ((i * 9) % 13) as f32 * 0.05 - 0.4).collect();
        let bias_f32: Vec<f32> = (0..output_dim).map(|i| ((i * 11) % 19) as f32 * 0.03 - 0.2).collect();

        assert_proxy_bias_matches_fp16(&mut ctx, input_dim, output_dim, &weights, &x_f32, &bias_f32)
    }

    #[test]
    fn q8_proxy_v_bias_matches_fp16_matmul() -> Result<(), MetalError> {
        let mut ctx = Context::<crate::tensor::F16>::new()?;

        let input_dim = 64;
        let output_dim = 48;
        let mut weights = Vec::with_capacity(input_dim * output_dim);
        for row in 0..input_dim {
            for col in 0..output_dim {
                weights.push(((row * 4 + col * 6) % 15) as i8 - 7);
            }
        }
        let x_f32: Vec<f32> = (0..input_dim).map(|i| ((i * 4) % 17) as f32 * 0.07 - 0.5).collect();
        let bias_f32: Vec<f32> = (0..output_dim).map(|i| ((i * 7) % 23) as f32 * 0.02 - 0.35).collect();

        assert_proxy_bias_matches_fp16(&mut ctx, input_dim, output_dim, &weights, &x_f32, &bias_f32)
    }

    #[test]
    fn q8_proxy_bias_matches_fp16_matmul_nk_layout() -> Result<(), MetalError> {
        // Explicitly pack [N,K] to exercise the NK accessor path
        let mut ctx = Context::<crate::tensor::F16>::new()?;
        let input_dim = 33; // K
        let output_dim = 21; // N

        // Build weights in NK layout directly
        let mut weights_nk = Vec::with_capacity(output_dim * input_dim);
        for n in 0..output_dim {
            for k in 0..input_dim {
                weights_nk.push(((n * 3 + k * 5) % 17) as i8 - 8);
            }
        }

        // Pack NK rows (rows = N, cols = K) so proxy should pick NK path
        let bytes_nk = pack_q8_blocks(output_dim, input_dim, &weights_nk);
        let swizzled = swizzle_q8_0_blocks_nk(output_dim, input_dim, &bytes_nk).expect("swizzle nk layout");
        let (data_bytes, scale_bytes) = split_blocks(&swizzled);
        let q8 = QuantizedQ8_0Tensor::from_split_bytes_in_context(vec![output_dim, input_dim], &data_bytes, &scale_bytes, &ctx)?;

        // x in [1, K]
        let x_f32: Vec<f32> = (0..input_dim).map(|i| ((i * 7) % 19) as f32 * 0.06 - 0.5).collect();
        let bias_f32: Vec<f32> = (0..output_dim).map(|i| ((i * 11) % 23) as f32 * 0.03 - 0.25).collect();

        let x = Tensor::<crate::tensor::F16>::from_f32_slice(vec![1, input_dim], TensorStorage::Pooled(&mut ctx), &x_f32)?;
        let bias = Tensor::<crate::tensor::F16>::from_f32_slice(vec![output_dim], TensorStorage::Pooled(&mut ctx), &bias_f32)?;

        // Proxy result using NK layout should match FP16 GEMV with A = transpose(weights_nk) to [K,N]
        let q8_y = ctx.call::<MatmulGemvOp>((&x.clone(), TensorType::Quant(QuantizedTensor::Q8_0(&q8)), None))?;
        let q8_y = ctx.call::<BroadcastElemwiseAddInplaceOp>((q8_y, bias.clone()))?;

        // Construct FP16 [K,N] tensor from NK weights
        let mut weights_kn = Vec::with_capacity(input_dim * output_dim);
        for k in 0..input_dim {
            for n in 0..output_dim {
                weights_kn.push(weights_nk[n * input_dim + k] as f32);
            }
        }
        let a_fp16 =
            Tensor::<crate::tensor::F16>::from_f32_slice(vec![input_dim, output_dim], TensorStorage::Pooled(&mut ctx), &weights_kn)?;
        let fp16_y = ctx.call::<MatmulGemvOp>((&x, TensorType::Dense(&a_fp16), None))?;
        let fp16_y = ctx.call::<BroadcastElemwiseAddInplaceOp>((fp16_y, bias))?;

        let q8_slice = q8_y.as_slice();
        let fp16_slice = fp16_y.as_slice();
        for (idx, (&q8_val, &fp16_val)) in q8_slice.iter().zip(fp16_slice.iter()).enumerate() {
            assert!((q8_val.to_f32() - fp16_val.to_f32()).abs() < 1e-2, "NK layout mismatch at {}", idx);
        }

        Ok(())
    }
}
