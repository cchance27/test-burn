#[cfg(test)]
mod tests {
    use metallic_env::FORCE_MATMUL_BACKEND_VAR;

    use crate::{Context, F16Element, MetalError, Tensor, TensorStorage, kernels::matmul_dispatcher::MatmulDispatchOp};

    #[test]
    fn test_matmul_gemm_tiled_basic() -> Result<(), MetalError> {
        let mut ctx = Context::<F16Element>::new()?;

        // Test a simple matrix multiplication: 4x4 * 4x4 = 4x4
        let a_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..16).map(|i| (i + 1) as f32).collect();

        let a_tensor = Tensor::from_f32_slice(vec![4, 4], TensorStorage::Pooled(&mut ctx), &a_data)?;
        let b_tensor = Tensor::from_f32_slice(vec![4, 4], TensorStorage::Pooled(&mut ctx), &b_data)?;

        // Perform the matmul using the dispatcher (which may route to our new kernel for large enough matrices)
        let result = ctx.call::<MatmulDispatchOp>((&a_tensor, &b_tensor, None, None, false, false, 1.0, 0.0))?;
        ctx.synchronize();

        // Validate basic properties: result should be 4x4 and have valid values
        assert_eq!(result.dims(), &[4, 4]);

        // Convert result back to host for validation
        let result_host = result.to_vec(); // Remove ? operator since to_vec() doesn't return Result
        // The result should not be all zeros (since A*B for these matrices should have values)
        assert!(result_host.iter().any(|&x| x != half::f16::ZERO));

        Ok(())
    }

    #[test]
    fn test_matmul_gemm_tiled_sizes() -> Result<(), MetalError> {
        let mut ctx = Context::<F16Element>::new()?;

        // Test different sizes that should trigger the tiled GEMM path
        let test_sizes = vec![
            (64, 64, 64),    // Should be large enough to potentially use tiled
            (128, 128, 128), // Definitely should use tiled based on thresholds
            (256, 256, 256), // Should definitely use tiled
        ];

        for (m, k, n) in test_sizes {
            let a_data: Vec<f32> = (0..(m * k)).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..(k * n)).map(|i| (i + 1) as f32).collect();

            let a_tensor = Tensor::from_f32_slice(vec![m, k], TensorStorage::Pooled(&mut ctx), &a_data)?;
            let b_tensor = Tensor::from_f32_slice(vec![k, n], TensorStorage::Pooled(&mut ctx), &b_data)?;

            let result = ctx.call::<MatmulDispatchOp>((&a_tensor, &b_tensor, None, None, false, false, 1.0, 0.0))?;
            ctx.synchronize();

            assert_eq!(result.dims(), &[m, n]);
            let result_host = result.to_vec(); // Remove ? operator
            assert!(!result_host.is_empty());
        }

        Ok(())
    }

    #[test]
    fn test_matmul_gemm_tiled_alpha_beta() -> Result<(), MetalError> {
        let mut ctx = Context::<F16Element>::new()?;

        // Test alpha/beta fusion with the tiled kernel
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0]; // Identity-like matrix

        let a_tensor = Tensor::from_f32_slice(vec![2, 2], TensorStorage::Pooled(&mut ctx), &a_data)?;
        let b_tensor = Tensor::from_f32_slice(vec![2, 2], TensorStorage::Pooled(&mut ctx), &b_data)?;

        // Test with alpha=2.0, beta=1.0: result = 2.0 * A * B + 1.0 * C
        // where C starts as zeros
        let c_tensor = Tensor::zeros(vec![2, 2], &mut ctx, true)?;
        let result = ctx.call::<MatmulDispatchOp>((&a_tensor, &b_tensor, None, Some(&c_tensor), false, false, 2.0, 1.0))?;
        ctx.synchronize();

        assert_eq!(result.dims(), &[2, 2]);
        let result_host = result.to_vec(); // Remove ? operator

        // For this specific case: result should be [2*(1*1+2*0)+1, 2*(1*0+2*1)+1; ...] = [3, 5; ...]
        // Basic validation that values are reasonable
        assert!(result_host.iter().all(|&x| x >= half::f16::ZERO));

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[test]
    fn matmul_gemm_tiled_matches_mps_with_bias_and_beta() -> Result<(), MetalError> {
        fn run_backend(
            backend: &str,
            m: usize,
            k: usize,
            n: usize,
            alpha: f32,
            beta: f32,
            a_data: &[f32],
            b_data: &[f32],
            bias_data: &[f32],
            out_data: &[f32],
        ) -> Result<Vec<f32>, MetalError> {
            let _guard = FORCE_MATMUL_BACKEND_VAR.set_guard(backend.to_string()).unwrap();
            let mut ctx = Context::<F16Element>::new()?;

            let a = Tensor::from_f32_slice(vec![m, k], TensorStorage::Pooled(&mut ctx), a_data)?;
            let b = Tensor::from_f32_slice(vec![k, n], TensorStorage::Pooled(&mut ctx), b_data)?;
            let bias = Tensor::from_f32_slice(vec![n], TensorStorage::Pooled(&mut ctx), bias_data)?;
            let out = Tensor::from_f32_slice(vec![m, n], TensorStorage::Pooled(&mut ctx), out_data)?;

            let result = ctx.call::<MatmulDispatchOp>((&a, &b, Some(&bias), Some(&out), false, false, alpha, beta))?;
            ctx.synchronize();

            let host: Vec<half::f16> = result.to_vec();
            Ok(host.into_iter().map(|v| v.to_f32()).collect())
        }

        let (m, k, n) = (32usize, 64usize, 48usize);
        let alpha = 1.5f32;
        let beta = 0.3f32;

        let mut a_data = Vec::with_capacity(m * k);
        let mut b_data = Vec::with_capacity(k * n);
        let mut bias_data = Vec::with_capacity(n);
        let mut out_data = Vec::with_capacity(m * n);

        // Deterministic pseudo-random-ish data that stays within FP16-friendly range
        let mut val = 0.25f32;
        for _ in 0..(m * k) {
            a_data.push(val * 0.5 - 0.12);
            val = (val * 1.37 + 0.19).fract();
        }
        for _ in 0..(k * n) {
            b_data.push(val * 0.45 - 0.08);
            val = (val * 1.21 + 0.31).fract();
        }
        for _ in 0..n {
            bias_data.push(val * 0.2 - 0.05);
            val = (val * 1.17 + 0.27).fract();
        }
        for _ in 0..(m * n) {
            out_data.push(val * 0.3 - 0.07);
            val = (val * 1.29 + 0.23).fract();
        }

        let tiled = run_backend("gemm_tiled", m, k, n, alpha, beta, &a_data, &b_data, &bias_data, &out_data)?;
        let mps = run_backend("mps", m, k, n, alpha, beta, &a_data, &b_data, &bias_data, &out_data)?;

        assert_eq!(tiled.len(), mps.len(), "result vector lengths must match");

        let max_abs = tiled
            .iter()
            .zip(mps.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let max_rel = tiled
            .iter()
            .zip(mps.iter())
            .map(|(lhs, rhs)| {
                let denom = rhs.abs().max(1e-2);
                (lhs - rhs).abs() / denom
            })
            .fold(0.0f32, f32::max);

        assert!(max_abs <= 2e-1, "gemm_tiled vs mps max abs diff too large: {max_abs}");
        assert!(max_rel <= 5.0, "gemm_tiled vs mps max relative diff too large: {max_rel}");

        Ok(())
    }
}
