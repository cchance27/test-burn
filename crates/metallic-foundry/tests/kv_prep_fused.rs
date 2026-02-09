use half::f16;
use metallic_foundry::{
    F16, Foundry, metals::{
        kv_cache_write::{KvCacheWriteRepeatKvHeads, KvCacheWriteRepeatKvHeadsParamsResolved}, kv_prep::{KvPrepFused, KvPrepFusedParamsResolved}, kv_rearrange::{KvRearrange, KvRearrangeParamsResolved}, rope::{Rope, RopeParamsResolved}
    }, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-4;

fn build_rope_tables(max_seq: usize, dim: usize) -> (Vec<f16>, Vec<f16>) {
    let half_dim = dim / 2;
    let mut cos = Vec::with_capacity(max_seq * half_dim);
    let mut sin = Vec::with_capacity(max_seq * half_dim);
    for pos in 0..max_seq {
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / dim as f32;
            let inv_freq = 1.0f32 / 10000.0f32.powf(exponent);
            let angle = pos as f32 * inv_freq;
            cos.push(f16::from_f32(angle.cos()));
            sin.push(f16::from_f32(angle.sin()));
        }
    }
    (cos, sin)
}

#[allow(clippy::too_many_arguments)]
fn run_fused_parity(
    seq_len: usize,
    position_offset: usize,
    max_seq_len: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let mut rng = rng();

    let d_model = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    let group_size = n_heads / n_kv_heads;

    let total_q = n_heads * seq_len * head_dim;
    let total_kv = n_kv_heads * seq_len * head_dim;
    let cache_elems = n_heads * max_seq_len * head_dim;

    let q_data: Vec<f16> = (0..seq_len * d_model).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let k_data: Vec<f16> = (0..seq_len * kv_dim).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();
    let v_data: Vec<f16> = (0..seq_len * kv_dim).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();

    let (cos_data, sin_data) = build_rope_tables(max_seq_len, head_dim);

    let q = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![seq_len, d_model], TensorInit::CopyFrom(&q_data))?;
    let k = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![seq_len, kv_dim], TensorInit::CopyFrom(&k_data))?;
    let v = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![seq_len, kv_dim], TensorInit::CopyFrom(&v_data))?;
    let cos = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![max_seq_len, head_dim / 2], TensorInit::CopyFrom(&cos_data))?;
    let sin = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![max_seq_len, head_dim / 2], TensorInit::CopyFrom(&sin_data))?;

    // Reference chain outputs
    let q_heads = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_q], TensorInit::Uninitialized)?;
    let k_heads = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_kv], TensorInit::Uninitialized)?;
    let v_heads = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_kv], TensorInit::Uninitialized)?;
    let q_rot_ref = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_q], TensorInit::Uninitialized)?;
    let k_rot = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_kv], TensorInit::Uninitialized)?;

    let cache_zero = vec![f16::from_f32(0.0); cache_elems];
    let k_cache_ref = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads, max_seq_len, head_dim],
        TensorInit::CopyFrom(&cache_zero),
    )?;
    let v_cache_ref = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads, max_seq_len, head_dim],
        TensorInit::CopyFrom(&cache_zero),
    )?;

    let q_rearrange = KvRearrange {
        input: TensorArg::from_tensor(&q),
        output: TensorArg::from_tensor(&q_heads),
        params: KvRearrangeParamsResolved {
            kv_dim: d_model as u32,
            row_stride: d_model as u32,
            kv_head_dim: head_dim as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_heads as u32,
            head_dim: head_dim as u32,
            seq: seq_len as u32,
            total_elements: total_q as u32,
        },
    };
    foundry.run(&q_rearrange)?;

    let k_rearrange = KvRearrange {
        input: TensorArg::from_tensor(&k),
        output: TensorArg::from_tensor(&k_heads),
        params: KvRearrangeParamsResolved {
            kv_dim: kv_dim as u32,
            row_stride: kv_dim as u32,
            kv_head_dim: head_dim as u32,
            n_heads: n_kv_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            seq: seq_len as u32,
            total_elements: total_kv as u32,
        },
    };
    foundry.run(&k_rearrange)?;

    let v_rearrange = KvRearrange {
        input: TensorArg::from_tensor(&v),
        output: TensorArg::from_tensor(&v_heads),
        params: KvRearrangeParamsResolved {
            kv_dim: kv_dim as u32,
            row_stride: kv_dim as u32,
            kv_head_dim: head_dim as u32,
            n_heads: n_kv_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            seq: seq_len as u32,
            total_elements: total_kv as u32,
        },
    };
    foundry.run(&v_rearrange)?;

    let rope_q = Rope::new(
        &TensorArg::from_tensor(&q_heads),
        &TensorArg::from_tensor(&q_rot_ref),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        RopeParamsResolved {
            dim: head_dim as u32,
            seq_len: seq_len as u32,
            position_offset: position_offset as u32,
            total_elements: total_q as u32,
        },
    );
    foundry.run(&rope_q)?;

    let rope_k = Rope::new(
        &TensorArg::from_tensor(&k_heads),
        &TensorArg::from_tensor(&k_rot),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        RopeParamsResolved {
            dim: head_dim as u32,
            seq_len: seq_len as u32,
            position_offset: position_offset as u32,
            total_elements: total_kv as u32,
        },
    );
    foundry.run(&rope_k)?;

    let write_k = KvCacheWriteRepeatKvHeads {
        input: TensorArg::from_tensor(&k_rot),
        cache: TensorArg::from_tensor(&k_cache_ref),
        params: KvCacheWriteRepeatKvHeadsParamsResolved {
            n_kv_heads: n_kv_heads as u32,
            n_heads: n_heads as u32,
            group_size: group_size as u32,
            head_dim: head_dim as u32,
            input_seq_len: seq_len as u32,
            position_offset: position_offset as u32,
            max_seq_len: max_seq_len as u32,
            total_elements: total_kv as u32,
            layer_idx: 0,
        },
    };
    foundry.run(&write_k)?;

    let write_v = KvCacheWriteRepeatKvHeads {
        input: TensorArg::from_tensor(&v_heads),
        cache: TensorArg::from_tensor(&v_cache_ref),
        params: KvCacheWriteRepeatKvHeadsParamsResolved {
            n_kv_heads: n_kv_heads as u32,
            n_heads: n_heads as u32,
            group_size: group_size as u32,
            head_dim: head_dim as u32,
            input_seq_len: seq_len as u32,
            position_offset: position_offset as u32,
            max_seq_len: max_seq_len as u32,
            total_elements: total_kv as u32,
            layer_idx: 0,
        },
    };
    foundry.run(&write_v)?;

    // Fused outputs
    let q_rot = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![total_q], TensorInit::Uninitialized)?;
    let k_cache = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads, max_seq_len, head_dim],
        TensorInit::CopyFrom(&cache_zero),
    )?;
    let v_cache = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_heads, max_seq_len, head_dim],
        TensorInit::CopyFrom(&cache_zero),
    )?;

    let fused = KvPrepFused {
        q: TensorArg::from_tensor(&q),
        k: TensorArg::from_tensor(&k),
        v: TensorArg::from_tensor(&v),
        q_rot: TensorArg::from_tensor(&q_rot),
        k_cache: TensorArg::from_tensor(&k_cache),
        v_cache: TensorArg::from_tensor(&v_cache),
        cos: TensorArg::from_tensor(&cos),
        sin: TensorArg::from_tensor(&sin),
        params: KvPrepFusedParamsResolved {
            d_model: d_model as u32,
            kv_dim: kv_dim as u32,
            head_dim: head_dim as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            group_size: group_size as u32,
            seq_len: seq_len as u32,
            position_offset: position_offset as u32,
            max_seq_len: max_seq_len as u32,
            total_elements: total_q as u32,
            rope_mode: 0,
        },
    };
    foundry.run(&fused)?;
    foundry.synchronize()?;

    let q_ref = q_rot_ref.to_vec(&foundry);
    let q_fused = q_rot.to_vec(&foundry);
    let k_ref = k_cache_ref.to_vec(&foundry);
    let k_fused = k_cache.to_vec(&foundry);
    let v_ref = v_cache_ref.to_vec(&foundry);
    let v_fused = v_cache.to_vec(&foundry);

    let mut max_diff = 0.0f32;
    for i in 0..q_ref.len() {
        max_diff = max_diff.max((q_ref[i].to_f32() - q_fused[i].to_f32()).abs());
    }
    assert!(max_diff <= TOLERANCE, "q_rot mismatch: max diff {}", max_diff);

    for i in 0..k_ref.len() {
        assert_eq!(k_ref[i].to_bits(), k_fused[i].to_bits(), "k_cache mismatch at {}", i);
    }
    for i in 0..v_ref.len() {
        assert_eq!(v_ref[i].to_bits(), v_fused[i].to_bits(), "v_cache mismatch at {}", i);
    }

    Ok(())
}

#[test]
#[serial]
fn test_kv_prep_fused_decode() -> Result<(), Box<dyn std::error::Error>> {
    run_fused_parity(1, 5, 16, 14, 2, 64)
}

#[test]
#[serial]
fn test_kv_prep_fused_prefill() -> Result<(), Box<dyn std::error::Error>> {
    run_fused_parity(4, 0, 16, 14, 2, 64)
}
