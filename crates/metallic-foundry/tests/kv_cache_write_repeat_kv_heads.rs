use half::f16;
use metallic_foundry::{
    F16, Foundry, metals::kv_cache_write::{KvCacheWriteRepeatKvHeads, KvCacheWriteRepeatKvHeadsParamsResolved}, storage::Pooled, tensor::{Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, SeedableRng, rngs::StdRng};
#[test]
fn test_kv_cache_write_repeat_kv_heads_m2() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let mut rng = StdRng::seed_from_u64(42);

    let n_kv_heads: usize = 2;
    let group_size: usize = 7;
    let n_heads: usize = n_kv_heads * group_size;
    let head_dim: usize = 64;
    let input_seq_len: usize = 2;
    let max_seq_len: usize = 32;
    let position_offset: usize = 5;

    let total_in = n_kv_heads * input_seq_len * head_dim;

    // Input: [n_kv_heads, input_seq_len, head_dim]
    let input_data: Vec<f16> = (0..total_in).map(|_| f16::from_f32(rng.random_range(-1.0f32..1.0f32))).collect();

    let input = FoundryTensor::<F16, Pooled>::new(
        &mut foundry,
        vec![n_kv_heads, input_seq_len, head_dim],
        TensorInit::CopyFrom(&input_data),
    )?;

    // Cache: [n_heads, max_seq_len, head_dim]
    let cache = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n_heads, max_seq_len, head_dim], TensorInit::Uninitialized)?;

    let params = KvCacheWriteRepeatKvHeadsParamsResolved {
        n_kv_heads: n_kv_heads as u32,
        n_heads: n_heads as u32,
        group_size: group_size as u32,
        head_dim: head_dim as u32,
        input_seq_len: input_seq_len as u32,
        position_offset: position_offset as u32,
        max_seq_len: max_seq_len as u32,
        total_elements: total_in as u32,
        layer_idx: 0,
    };

    let kernel = KvCacheWriteRepeatKvHeads {
        input: TensorArg::from_tensor(&input),
        cache: TensorArg::from_tensor(&cache),
        params,
    };

    foundry.run(&kernel)?;
    foundry.synchronize()?;

    // CPU reference: for each kv head, write into repeated heads.
    let cache_gpu: Vec<f16> = cache.to_vec(&foundry);

    // Compare only the written region for strictness.
    for kv_h in 0..n_kv_heads {
        for t in 0..input_seq_len {
            for d in 0..head_dim {
                let expected = input_data[kv_h * input_seq_len * head_dim + t * head_dim + d];
                for r in 0..group_size {
                    let out_h = kv_h * group_size + r;
                    let idx = out_h * max_seq_len * head_dim + (position_offset + t) * head_dim + d;
                    let got = cache_gpu[idx];
                    assert_eq!(got.to_bits(), expected.to_bits(), "mismatch at head={}, t={}, d={}", out_h, t, d);
                }
            }
        }
    }

    // Sanity: cache is the expected size.
    assert_eq!(cache_gpu.len(), n_heads * max_seq_len * head_dim);

    Ok(())
}
