use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::flashattention::step::run_flash_decode, storage::Pooled, tensor::{F16, F32, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

#[test]
fn flash_decode_rejects_non_f16_qkv() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 2;
    let head_dim: u32 = 64;
    let kv_len: u32 = 8;
    let d_model = (n_heads * head_dim) as usize;

    let q_f32 = vec![0.0f32; d_model];
    let k_f16 = vec![f16::from_f32(0.0); (n_heads as usize) * (kv_len as usize) * (head_dim as usize)];
    let v_f16 = vec![f16::from_f32(0.0); (n_heads as usize) * (kv_len as usize) * (head_dim as usize)];
    let out_f16 = vec![f16::from_f32(0.0); d_model];

    let q = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_f32))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&k_f16),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&v_f16),
    )?;
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&out_f16))?;

    let err = run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&output),
        n_heads,
        head_dim,
        kv_len,
        1,
        true,
    )
    .expect_err("expected fail-fast for mixed-policy qkv");

    let msg = format!("{err}");
    assert!(
        msg.contains("FlashAttention mixed-policy is unsupported"),
        "unexpected error: {msg}"
    );
    Ok(())
}

#[test]
fn flash_decode_accepts_uniform_f32_qkv() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 2;
    let head_dim: u32 = 64;
    let kv_len: u32 = 8;
    let d_model = (n_heads * head_dim) as usize;

    let q_f32 = vec![0.0f32; d_model];
    let kv_f32 = vec![0.0f32; (n_heads as usize) * (kv_len as usize) * (head_dim as usize)];
    let out_f32 = vec![0.0f32; d_model];

    let q = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_f32))?;
    let k = FoundryTensor::<F32, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_f32),
    )?;
    let v = FoundryTensor::<F32, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_f32),
    )?;
    let output = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&out_f32))?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&output),
        n_heads,
        head_dim,
        kv_len,
        1,
        true,
    )?;
    Ok(())
}

#[test]
fn flash_prefill_accepts_uniform_f32_qkv() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 2;
    let head_dim: u32 = 64;
    let q_len: u32 = 8;
    let kv_len: u32 = 8;
    let d_model = (n_heads * head_dim) as usize;

    let q_f32 = vec![0.0f32; (q_len as usize) * d_model];
    let kv_f32 = vec![0.0f32; (n_heads as usize) * (kv_len as usize) * (head_dim as usize)];
    let out_f32 = vec![0.0f32; (q_len as usize) * d_model];

    let q = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![q_len as usize, d_model], TensorInit::CopyFrom(&q_f32))?;
    let k = FoundryTensor::<F32, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_f32),
    )?;
    let v = FoundryTensor::<F32, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_f32),
    )?;
    let output = FoundryTensor::<F32, Pooled>::new(&mut foundry, vec![q_len as usize, d_model], TensorInit::CopyFrom(&out_f32))?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&output),
        n_heads,
        head_dim,
        kv_len,
        q_len,
        true,
    )?;
    Ok(())
}

#[test]
fn flash_decode_rejects_kv_head_major_false() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 2;
    let head_dim: u32 = 64;
    let kv_len: u32 = 8;
    let d_model = (n_heads * head_dim) as usize;

    let q_f16 = vec![f16::from_f32(0.0); d_model];
    let kv_f16 = vec![f16::from_f32(0.0); (n_heads as usize) * (kv_len as usize) * (head_dim as usize)];
    let out_f16 = vec![f16::from_f32(0.0); d_model];

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_f16))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_f16),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_len as usize, head_dim as usize],
        TensorInit::CopyFrom(&kv_f16),
    )?;
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&out_f16))?;

    let err = run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&output),
        n_heads,
        head_dim,
        kv_len,
        1,
        false,
    )
    .expect_err("expected fail-fast for kv_head_major=false decode path");

    let msg = format!("{err}");
    assert!(
        msg.contains("Flash decode only supports kv_head_major=true"),
        "unexpected error: {msg}"
    );
    Ok(())
}
