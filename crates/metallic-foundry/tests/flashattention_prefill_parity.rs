use half::f16;
use metallic_env::{EnvVarGuard, FoundryEnvVar};
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
    // Deterministic PRNG (LCG). Good enough for tests; avoids adding a rand dependency.
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    let mantissa = *state >> 9; // 23 bits
    (mantissa as f32) * (1.0 / ((1u32 << 23) as f32))
}

fn fill_f16(state: &mut u32, len: usize, scale: f32) -> Vec<f16> {
    (0..len)
        .map(|_| {
            let r01 = lcg_next_f32(state);
            let r = (r01 * 2.0 - 1.0) * scale; // [-scale, +scale]
            f16::from_f32(r)
        })
        .collect()
}

fn with_prefill_warps<R>(warps: u32, f: impl FnOnce() -> R) -> R {
    let warps_v = warps.to_string();
    let _warps_guard = EnvVarGuard::set(FoundryEnvVar::FaPrefillWarps, &warps_v);
    f()
}

fn with_prefill_warps_and_split_k<R>(warps: u32, split_k: u32, f: impl FnOnce() -> R) -> R {
    let warps_v = warps.to_string();
    let split_k_v = split_k.to_string();
    let _warps_guard = EnvVarGuard::set(FoundryEnvVar::FaPrefillWarps, &warps_v);
    let _split_k_guard = EnvVarGuard::set(FoundryEnvVar::FaPrefillSplitK, &split_k_v);
    f()
}

fn run_prefill_case(n_heads: u32, head_dim: u32, m: u32, kv_len: u32, query_offset: u32) -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 123u32 ^ (m.wrapping_mul(31) ^ kv_len.wrapping_mul(131));

    let q_host = fill_f16(&mut rng, (m as usize) * d_model, 0.25);
    let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::CopyFrom(&q_host))?;
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

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_flash),
        n_heads,
        head_dim,
        kv_len,
        m,
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
        query_offset: DynamicValue::Literal(query_offset),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(m),
        kv_head_major: true,
    };
    step.execute(&mut foundry, &mut bindings)?;
    foundry.synchronize()?;

    let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    let diff = max_abs_diff(&flash_out, &mat_out);
    assert!(
        diff < 8e-2,
        "prefill parity failed: n_heads={n_heads} head_dim={head_dim} m={m} kv_len={kv_len} query_offset={query_offset} max_abs_diff={diff}"
    );
    Ok(())
}

#[test]
fn flashattention_prefill_matches_materialized_warps4_d64() -> Result<(), MetalError> {
    with_prefill_warps(4, || {
        // m spans <16, ==16, >16 to cover both partial and full tiles for WARPS=4 (TileM=16).
        run_prefill_case(14, 64, 7, 7, 0)?;
        run_prefill_case(14, 64, 16, 16, 0)?;
        run_prefill_case(14, 64, 17, 17, 0)?;
        Ok(())
    })
}

#[test]
fn flashattention_prefill_matches_materialized_warps4_d128() -> Result<(), MetalError> {
    with_prefill_warps(4, || {
        run_prefill_case(8, 128, 7, 7, 0)?;
        run_prefill_case(8, 128, 16, 16, 0)?;
        run_prefill_case(8, 128, 17, 17, 0)?;
        Ok(())
    })
}

#[test]
fn flashattention_prefill_splitk_matches_materialized_d64() -> Result<(), MetalError> {
    // Force Split-K even for smaller KV so we can validate correctness deterministically.
    with_prefill_warps_and_split_k(8, 4, || {
        run_prefill_case(14, 64, 32, 256, 224)?;
        run_prefill_case(14, 64, 17, 256, 239)?;
        run_prefill_case(14, 64, 32, 1024, 992)?;
        Ok(())
    })
}

#[test]
fn flashattention_prefill_splitk_matches_materialized_d128() -> Result<(), MetalError> {
    with_prefill_warps_and_split_k(8, 4, || {
        run_prefill_case(8, 128, 32, 256, 224)?;
        run_prefill_case(8, 128, 17, 256, 239)?;
        run_prefill_case(8, 128, 32, 1024, 992)?;
        Ok(())
    })
}

#[test]
fn flashattention_prefill_matches_materialized_no_prefix() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let m: u32 = 32;
    let kv_len: u32 = m;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 11u32;

    // Token-major packed Q: [M, d_model]
    let q_host = fill_f16(&mut rng, (m as usize) * d_model, 0.25);
    // KV cache head-major: [H, kv_len, D]
    let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::CopyFrom(&q_host))?;
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

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_flash),
        n_heads,
        head_dim,
        kv_len,
        m,
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
        query_offset: DynamicValue::Literal(0),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(m),
        kv_head_major: true,
    };
    step.execute(&mut foundry, &mut bindings)?;

    foundry.synchronize()?;

    let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    let diff = max_abs_diff(&flash_out, &mat_out);

    assert!(diff < 8e-2, "max abs diff too high: {diff}");
    Ok(())
}

#[test]
fn flashattention_prefill_matches_materialized_with_prefix() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let m: u32 = 32;
    let kv_len: u32 = 96;
    let query_offset: u32 = kv_len - m;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 19u32;

    let q_host = fill_f16(&mut rng, (m as usize) * d_model, 0.25);
    let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::CopyFrom(&q_host))?;
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

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_flash),
        n_heads,
        head_dim,
        kv_len,
        m,
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
        query_offset: DynamicValue::Literal(query_offset),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(m),
        kv_head_major: true,
    };
    step.execute(&mut foundry, &mut bindings)?;

    foundry.synchronize()?;

    let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    let diff = max_abs_diff(&flash_out, &mat_out);

    assert!(diff < 8e-2, "max abs diff too high: {diff}");
    Ok(())
}

#[test]
fn flashattention_prefill_matches_materialized_partial_tile() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let m: u32 = 26; // not a multiple of the tile size (32)
    let kv_len: u32 = m;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 29u32;

    let q_host = fill_f16(&mut rng, (m as usize) * d_model, 0.25);
    let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::CopyFrom(&q_host))?;
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

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m as usize, d_model], TensorInit::Uninitialized)?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_flash),
        n_heads,
        head_dim,
        kv_len,
        m,
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
        query_offset: DynamicValue::Literal(0),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(m),
        kv_head_major: true,
    };
    step.execute(&mut foundry, &mut bindings)?;

    foundry.synchronize()?;

    let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    let diff = max_abs_diff(&flash_out, &mat_out);

    assert!(diff < 8e-2, "max abs diff too high: {diff}");
    Ok(())
}

#[test]
fn flashattention_prefill_matches_materialized_q_token_meta_head_major_contents() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let m: u32 = 26;
    let m_cap: u32 = 32;
    let kv_len: u32 = m;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 41u32;

    // Mimic Foundry behavior:
    // - Q tensor is allocated as fixed-capacity token-major metadata: [1, M_cap, d_model]
    // - but fused KV-prep writes q_rot in head-major order over the *true* m:
    //   flatten [n_heads, m, head_dim] into the buffer starting at element 0.
    let mut q_host = vec![f16::from_f32(0.0); (m_cap as usize) * d_model];
    let packed_len = (n_heads as usize) * (m as usize) * (head_dim as usize);
    let packed = fill_f16(&mut rng, packed_len, 0.25);
    q_host[..packed_len].copy_from_slice(&packed);

    let k_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * (kv_len as usize) * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, m_cap as usize, d_model], TensorInit::CopyFrom(&q_host))?;
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

    // Mirror the fixed-capacity output shape seen in inference: [M_cap, d_model].
    let out_flash = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![m_cap as usize, d_model],
        TensorInit::CopyFrom(&vec![f16::from_f32(0.0); (m_cap as usize) * d_model]),
    )?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![m_cap as usize, d_model],
        TensorInit::CopyFrom(&vec![f16::from_f32(0.0); (m_cap as usize) * d_model]),
    )?;

    run_flash_decode(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&out_flash),
        n_heads,
        head_dim,
        kv_len,
        m,
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
        query_offset: DynamicValue::Literal(0),
        n_heads: DynamicValue::Literal(n_heads),
        head_dim: DynamicValue::Literal(head_dim),
        kv_seq_len: DynamicValue::Literal(kv_len),
        m: DynamicValue::Literal(m),
        kv_head_major: true,
    };
    step.execute(&mut foundry, &mut bindings)?;

    foundry.synchronize()?;

    let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);

    let compare_elems = (m as usize) * d_model;
    let diff = max_abs_diff(&flash_out[..compare_elems], &mat_out[..compare_elems]);
    assert!(diff < 8e-2, "max abs diff too high: {diff}");
    Ok(())
}

#[test]
fn flashattention_prefill_matches_materialized_various_m() -> Result<(), MetalError> {
    let n_heads: u32 = 14;
    let head_dim: u32 = 64;

    // Cover under/over tile and odd sizes.
    for m in [1u32, 2, 7, 17, 31, 32, 33].into_iter() {
        // No prefix
        run_prefill_case(n_heads, head_dim, m, m, 0)?;

        // With prefix (causal invariant: query_offset + m == kv_len)
        let kv_len = m + 64;
        let query_offset = kv_len - m;
        run_prefill_case(n_heads, head_dim, m, kv_len, query_offset)?;
    }
    Ok(())
}

#[test]
fn flashattention_prefill_matches_materialized_d128() -> Result<(), MetalError> {
    let n_heads: u32 = 8;
    let head_dim: u32 = 128;

    for m in [1u32, 2, 17, 31, 32, 33].into_iter() {
        run_prefill_case(n_heads, head_dim, m, m, 0)?;

        let kv_len = m + 64;
        let query_offset = kv_len - m;
        run_prefill_case(n_heads, head_dim, m, kv_len, query_offset)?;
    }
    Ok(())
}
