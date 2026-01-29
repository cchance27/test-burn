use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::{flashattention::step::run_flash_decode, sdpa::step::SdpaMaterializedStep}, spec::{DynamicValue, Step, TensorBindings}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};

fn max_abs_diff(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn lcg_next_f32(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let mantissa = (*state >> 9) as u32; // 23 bits
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
fn flashattention_decode_matches_materialized_on_fixed_foundry_buffers() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    // Mirrors Qwen2.5 Foundry decode shapes:
    // - q_rot: [1, 32, d_model] (fixed capacity), but only row 0 is used for m=1 decode.
    // - attn_out: [32, d_model]
    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let kv_len: u32 = 128;
    let capacity_tokens: usize = 32;
    let d_model = (n_heads * head_dim) as usize;

    let mut rng = 42u32;

    // Fill only the first row of q_rot (packed [d_model]).
    let q_row0 = fill_f16(&mut rng, d_model, 0.25);
    let mut q_host = vec![f16::from_f32(0.0); capacity_tokens * d_model];
    q_host[..d_model].copy_from_slice(&q_row0);

    // K/V cache: [n_heads, capacity, head_dim] head-major.
    let kv_capacity = kv_len as usize;
    let k_host = fill_f16(&mut rng, (n_heads as usize) * kv_capacity * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * kv_capacity * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, capacity_tokens, d_model], TensorInit::CopyFrom(&q_host))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_capacity, head_dim as usize],
        TensorInit::CopyFrom(&k_host),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, kv_capacity, head_dim as usize],
        TensorInit::CopyFrom(&v_host),
    )?;

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![capacity_tokens, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![capacity_tokens, d_model], TensorInit::Uninitialized)?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_flash),
        n_heads,
        head_dim,
        kv_len,
        1,
        true,
    )?;

    let mut bindings = TensorBindings::new();
    bindings.insert("q".to_string(), TensorArg::from_tensor(&q));
    bindings.insert("k".to_string(), TensorArg::from_tensor(&k));
    bindings.insert("v".to_string(), TensorArg::from_tensor(&v));
    bindings.insert("o".to_string(), TensorArg::from_tensor(&out_mat));

    let step = SdpaMaterializedStep {
        q: "q".into(),
        k: "k".into(),
        v: "v".into(),
        output: "o".into(),
        causal: true,
        query_offset: DynamicValue::Literal(kv_len - 1),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(1),
        kv_head_major: true,
    };
    step.execute(&mut foundry, &mut bindings)?;

    foundry.synchronize()?;

    let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);

    let diff = max_abs_diff(&flash_out[..d_model], &mat_out[..d_model]);
    assert!(diff < 6e-2, "max abs diff too high: {diff}");
    Ok(())
}
