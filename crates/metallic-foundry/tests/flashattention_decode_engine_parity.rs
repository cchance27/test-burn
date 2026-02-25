use half::f16;
use metallic_env::{EnvVarGuard, FoundryEnvVar};
use metallic_foundry::{
    Foundry, MetalError, metals::flashattention::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant, step::run_flash_decode_with_variant}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

fn max_abs_diff(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn lcg_next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let mantissa = *state >> 9;
    (mantissa as f32) * (1.0 / ((1u32 << 23) as f32))
}

fn fill_f16(state: &mut u32, len: usize, scale: f32) -> Vec<f16> {
    (0..len)
        .map(|_| {
            let r01 = lcg_next_f32(state);
            let r = (r01 * 2.0 - 1.0) * scale;
            f16::from_f32(r)
        })
        .collect()
}

#[test]
#[serial_test::serial]
#[ignore]
fn flashattention_decode_engine_scalar_vs_mma_half2_parity() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 4;
    let head_dim: u32 = 64;
    let kv_len: u32 = 32;
    let d_model = (n_heads * head_dim) as usize;

    let mut rng = 13u32;
    let q_host = fill_f16(&mut rng, d_model, 0.2);
    let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.2);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.2);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_host))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&k_host),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&v_host),
    )?;
    let out_scalar = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;
    let out_mma = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

    let variant = FlashDecodeVariant {
        warps: 4,
        keys_per_warp: 8,
        scalar: FlashDecodeScalar::Half2,
        tg_out: FlashDecodeTgOut::Float,
    };

    let _scalar_engine = EnvVarGuard::set(FoundryEnvVar::FaDecodeEngine, "scalar");
    let _scalar_dtype = EnvVarGuard::set(FoundryEnvVar::FaDecodeScalar, "half2");
    run_flash_decode_with_variant(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_scalar),
        n_heads,
        head_dim,
        kv_len,
        1,
        true,
        variant,
    )?;
    foundry.synchronize()?;
    let scalar_host: Vec<f16> = out_scalar.to_vec(&foundry);

    let _mma_engine = EnvVarGuard::set(FoundryEnvVar::FaDecodeEngine, "mma");
    let _mma_dtype = EnvVarGuard::set(FoundryEnvVar::FaDecodeScalar, "half2");
    run_flash_decode_with_variant(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_mma),
        n_heads,
        head_dim,
        kv_len,
        1,
        true,
        variant,
    )?;
    foundry.synchronize()?;
    let mma_host: Vec<f16> = out_mma.to_vec(&foundry);

    let diff = max_abs_diff(&scalar_host, &mma_host);
    assert!(diff < 6e-2, "decode scalar vs mma parity diff too high: {diff}");
    Ok(())
}
