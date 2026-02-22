//! Parity tests for Foundry KV cache write/read kernels.
//!
//! We compare Foundry kernels against CPU references.

use half::f16;
use metallic_foundry::{
    Foundry, MetalError, metals::kv_cache_write::{KvCacheRead, KvCacheReadParamsResolved, KvCacheWrite, KvCacheWriteParamsResolved}, storage::Pooled, tensor::{F16 as FoundryF16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

const TOLERANCE: f32 = 1e-3;

fn cpu_kv_cache_write(
    cache: &mut [f16],
    input: &[f16],
    n_kv_heads: usize,
    head_dim: usize,
    input_seq_len: usize,
    position_offset: usize,
    max_seq_len: usize,
) {
    for head in 0..n_kv_heads {
        for s in 0..input_seq_len {
            for d in 0..head_dim {
                let src_idx = ((head * input_seq_len) + s) * head_dim + d;
                let dst_idx = ((head * max_seq_len) + position_offset + s) * head_dim + d;
                cache[dst_idx] = input[src_idx];
            }
        }
    }
}

fn cpu_kv_cache_read(cache: &[f16], n_kv_heads: usize, head_dim: usize, seq_len: usize, max_seq_len: usize) -> Vec<f16> {
    let mut out = vec![f16::from_f32(0.0); n_kv_heads * seq_len * head_dim];
    for head in 0..n_kv_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                let src_idx = ((head * max_seq_len) + s) * head_dim + d;
                let dst_idx = ((head * seq_len) + s) * head_dim + d;
                out[dst_idx] = cache[src_idx];
            }
        }
    }
    out
}

fn max_diff(a: &[f16], b: &[f16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

#[test]
#[serial]
fn test_kv_cache_write_parity_single_step() -> Result<(), MetalError> {
    let n_kv_heads = 2usize;
    let head_dim = 8usize;
    let max_seq_len = 16usize;
    let position_offset = 5usize;
    let input_seq_len = 1usize;

    let mut rng = rng();
    let input_len = n_kv_heads * input_seq_len * head_dim;
    let input_data: Vec<f16> = (0..input_len).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();

    let mut cpu_cache = vec![f16::from_f32(0.0); n_kv_heads * max_seq_len * head_dim];
    cpu_kv_cache_write(
        &mut cpu_cache,
        &input_data,
        n_kv_heads,
        head_dim,
        input_seq_len,
        position_offset,
        max_seq_len,
    );

    // Foundry kernel
    let mut foundry = Foundry::new()?;
    let zero_cache = vec![f16::from_f32(0.0); n_kv_heads * max_seq_len * head_dim];
    let cache_foundry = FoundryTensor::<FoundryF16, Pooled>::new(
        &mut foundry,
        vec![n_kv_heads, max_seq_len, head_dim],
        TensorInit::CopyFrom(&zero_cache),
    )?;
    let input_foundry = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![input_len], TensorInit::CopyFrom(&input_data))?;

    let params = KvCacheWriteParamsResolved {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        input_seq_len: input_seq_len as u32,
        position_offset: position_offset as u32,
        max_seq_len: max_seq_len as u32,
        total_elements: input_len as u32,
        layer_idx: 0,
    };
    let kernel = KvCacheWrite {
        input: TensorArg::from_tensor(&input_foundry),
        cache: TensorArg::from_tensor(&cache_foundry),
        params,
    };
    foundry.run(&kernel)?;
    let foundry_cache = cache_foundry.to_vec(&foundry);

    let diff_foundry_cpu = max_diff(&foundry_cache, &cpu_cache);

    assert!(diff_foundry_cpu <= TOLERANCE, "Foundry vs CPU max diff {}", diff_foundry_cpu);

    Ok(())
}

#[test]
#[serial]
fn test_kv_cache_read_parity_multi_step() -> Result<(), MetalError> {
    let n_kv_heads = 2usize;
    let head_dim = 8usize;
    let max_seq_len = 16usize;
    let steps = 6usize;

    let mut rng = rng();
    let input_len = n_kv_heads * head_dim;
    let mut cpu_cache = vec![f16::from_f32(0.0); n_kv_heads * max_seq_len * head_dim];

    let mut foundry = Foundry::new()?;
    let cache_foundry = FoundryTensor::<FoundryF16, Pooled>::new(
        &mut foundry,
        vec![n_kv_heads, max_seq_len, head_dim],
        TensorInit::CopyFrom(&cpu_cache),
    )?;

    for pos in 0..steps {
        let step_data: Vec<f16> = (0..input_len).map(|_| f16::from_f32(rng.random_range(-1.0..1.0))).collect();

        cpu_kv_cache_write(&mut cpu_cache, &step_data, n_kv_heads, head_dim, 1, pos, max_seq_len);

        let input_foundry = FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![input_len], TensorInit::CopyFrom(&step_data))?;
        let params = KvCacheWriteParamsResolved {
            n_kv_heads: n_kv_heads as u32,
            head_dim: head_dim as u32,
            input_seq_len: 1,
            position_offset: pos as u32,
            max_seq_len: max_seq_len as u32,
            total_elements: input_len as u32,
            layer_idx: 0,
        };
        let kernel = KvCacheWrite {
            input: TensorArg::from_tensor(&input_foundry),
            cache: TensorArg::from_tensor(&cache_foundry),
            params,
        };
        foundry.run(&kernel)?;
    }

    let output_foundry =
        FoundryTensor::<FoundryF16, Pooled>::new(&mut foundry, vec![n_kv_heads * steps * head_dim], TensorInit::Uninitialized)?;
    let read_params = KvCacheReadParamsResolved {
        n_kv_heads: n_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len: steps as u32,
        max_seq_len: max_seq_len as u32,
        total_elements: (n_kv_heads * steps * head_dim) as u32,
    };
    let read_kernel = KvCacheRead {
        cache: TensorArg::from_tensor(&cache_foundry),
        output: TensorArg::from_tensor(&output_foundry),
        params: read_params,
    };
    foundry.run(&read_kernel)?;
    let foundry_read = output_foundry.to_vec(&foundry);

    let cpu_read = cpu_kv_cache_read(&cpu_cache, n_kv_heads, head_dim, steps, max_seq_len);

    let diff_foundry_cpu = max_diff(&foundry_read, &cpu_read);

    assert!(diff_foundry_cpu <= TOLERANCE, "Foundry vs CPU max diff {}", diff_foundry_cpu);

    Ok(())
}
