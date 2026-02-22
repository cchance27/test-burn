use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::{
        flashattention::{FlashDecodeScalar, FlashDecodeTgOut, FlashDecodeVariant, step::run_flash_decode_with_variant}, sdpa::step::SdpaMaterializedStep
    }, spec::{DynamicValue, Step, TensorBindings}, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
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

fn decode_variants_for_head_dim(head_dim: u32) -> Vec<FlashDecodeVariant> {
    match head_dim {
        64 => vec![
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half2,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 16,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 32,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
            FlashDecodeVariant {
                warps: 8,
                keys_per_warp: 16,
                scalar: FlashDecodeScalar::Half4,
                tg_out: FlashDecodeTgOut::Float,
            },
        ],
        128 => {
            // Smaller blocks can win for shorter KV; keep both to ensure we don't regress either regime.
            vec![
                FlashDecodeVariant {
                    warps: 8,
                    keys_per_warp: 16,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out: FlashDecodeTgOut::Float,
                },
                FlashDecodeVariant {
                    warps: 8,
                    keys_per_warp: 8,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out: FlashDecodeTgOut::Float,
                },
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 16,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out: FlashDecodeTgOut::Float,
                },
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 8,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out: FlashDecodeTgOut::Float,
                },
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 16,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out: FlashDecodeTgOut::Half,
                },
                FlashDecodeVariant {
                    warps: 16,
                    keys_per_warp: 8,
                    scalar: FlashDecodeScalar::Half4,
                    tg_out: FlashDecodeTgOut::Half,
                },
            ]
        }
        _ => vec![],
    }
}

#[test]
fn flashattention_decode_matches_materialized_small() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let kv_len: u32 = 128;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 1u32;
    let q_host = fill_f16(&mut rng, d_model, 0.25);

    // K/V cache: [n_heads, capacity, head_dim] head-major.
    let capacity = kv_len as usize;
    let k_host = fill_f16(&mut rng, (n_heads as usize) * capacity * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * capacity * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_host))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, capacity, head_dim as usize],
        TensorInit::CopyFrom(&k_host),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, capacity, head_dim as usize],
        TensorInit::CopyFrom(&v_host),
    )?;

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

    // Run materialized via the Step impl to keep bindings consistent.
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

    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    for variant in decode_variants_for_head_dim(head_dim) {
        run_flash_decode_with_variant(
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
            variant,
        )?;
        foundry.synchronize()?;
        let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
        let diff = max_abs_diff(&flash_out, &mat_out);
        // Loose tolerance: different accumulation orders + `fast::exp` differences.
        assert!(diff < 6e-2, "variant={variant:?} max abs diff too high: {diff}");
    }
    Ok(())
}

#[test]
fn flashattention_decode_matches_materialized_odd_kv_len() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 14;
    let head_dim: u32 = 64;
    let kv_len: u32 = 27; // intentionally not a multiple of the tile size

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 7u32;
    let q_host = fill_f16(&mut rng, d_model, 0.25);

    // Allocate a cache-like capacity bigger than kv_len to mimic real KV caches.
    let capacity: usize = 64;
    let k_host = fill_f16(&mut rng, (n_heads as usize) * capacity * (head_dim as usize), 0.25);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * capacity * (head_dim as usize), 0.25);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_host))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, capacity, head_dim as usize],
        TensorInit::CopyFrom(&k_host),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, capacity, head_dim as usize],
        TensorInit::CopyFrom(&v_host),
    )?;

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

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

    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    for variant in decode_variants_for_head_dim(head_dim) {
        run_flash_decode_with_variant(
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
            variant,
        )?;
        foundry.synchronize()?;
        let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
        let diff = max_abs_diff(&flash_out, &mat_out);
        assert!(diff < 6e-2, "variant={variant:?} max abs diff too high: {diff}");
    }
    Ok(())
}
#[test]
fn flashattention_decode_matches_materialized_d128() -> Result<(), MetalError> {
    let mut foundry = Foundry::new()?;

    let n_heads: u32 = 8;
    let head_dim: u32 = 128; // D=128 test case
    let kv_len: u32 = 128;

    let d_model = (n_heads * head_dim) as usize;
    let mut rng = 42u32;
    // Lower scale to avoid overflow in D=128 accumulation if not careful
    let q_host = fill_f16(&mut rng, d_model, 0.1);

    // K/V cache: [n_heads, capacity, head_dim] head-major.
    let capacity = kv_len as usize;
    let k_host = fill_f16(&mut rng, (n_heads as usize) * capacity * (head_dim as usize), 0.1);
    let v_host = fill_f16(&mut rng, (n_heads as usize) * capacity * (head_dim as usize), 0.1);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::CopyFrom(&q_host))?;
    let k = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, capacity, head_dim as usize],
        TensorInit::CopyFrom(&k_host),
    )?;
    let v = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads as usize, capacity, head_dim as usize],
        TensorInit::CopyFrom(&v_host),
    )?;

    let out_flash = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;
    let out_mat = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![1, d_model], TensorInit::Uninitialized)?;

    // Run materialized via the Step impl to keep bindings consistent.
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

    let mat_out: Vec<f16> = out_mat.to_vec(&foundry);
    for variant in decode_variants_for_head_dim(head_dim) {
        run_flash_decode_with_variant(
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
            variant,
        )?;
        foundry.synchronize()?;
        let flash_out: Vec<f16> = out_flash.to_vec(&foundry);
        let diff = max_abs_diff(&flash_out, &mat_out);
        assert!(diff < 6e-2, "variant={variant:?} max abs diff too high: {diff}");
    }
    Ok(())
}
