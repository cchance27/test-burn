use anyhow::Result;
use half::f16;

use super::*;
use crate::{Context, F16Element, Tensor, TensorInit, TensorStorage};

fn matmul_cpu(a: &[f16], b: &[f16], m: usize, k: usize, n: usize) -> Vec<f16> {
    let mut c = vec![f16::from_f32(0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                let val_a = a[i * k + l];
                let val_b = b[l * n + j];
                let prod = (val_a * val_b).to_f32();
                sum += prod;
            }
            c[i * n + j] = f16::from_f32(sum);
        }
    }
    c
}

#[test]
fn test_matmul_gemv_small_n8_correctness() -> Result<()> {
    let mut ctx = Context::<F16Element>::new()?;
    let m = 4;
    let k = 16;
    let n = 8;

    let a_base: Vec<f16> = vec![0.5, 1.0, 2.0, 4.0].into_iter().map(f16::from_f32).collect();
    let b_base: Vec<f16> = vec![0.5, 1.0, 2.0, 4.0, 0.125, 0.25, 0.5, 1.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let a_data: Vec<f16> = a_base.iter().cycle().take(m * k).cloned().collect();
    let b_data: Vec<f16> = b_base.iter().cycle().take(k * n).cloned().collect();

    let a = Tensor::new(vec![m, k], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&a_data))?;
    let b = Tensor::new(vec![k, n], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&b_data))?;

    let out = ctx.call::<MatmulGemvSmallN8Op>((&a, &b))?;
    ctx.synchronize();

    let expected_c = matmul_cpu(&a_data, &b_data, m, k, n);
    let result_c = out.as_slice();

    assert_eq!(expected_c.len(), result_c.len());
    for (i, (expected, actual)) in expected_c.iter().zip(result_c.iter()).enumerate() {
        assert!(
            (expected - actual).to_f32().abs() < 1e-2,
            "mismatch at index {}: expected: {}, actual: {}",
            i,
            expected,
            actual
        );
    }

    Ok(())
}
