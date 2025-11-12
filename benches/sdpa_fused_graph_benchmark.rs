use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use metallic::{Context, Tensor, kernels::sdpa_mps_graph::SdpaMpsGraphOp, tensor::F16};

// Compare SDPA-only path vs fused SDPA+projection path
fn benchmark_sdpa_vs_fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa_fused_graph_benchmark");

    // Test parameters
    let batch = 1;
    let seq_q = 128;
    let seq_k = 128;
    let dim = 64;
    let output_dim = 64;

    group.bench_function("sdpa_only", |b| {
        b.iter(|| {
            let mut ctx = Context::<F16>::new().unwrap();

            // Create sample tensors using f32 data (which gets converted to F16)
            let q_data = vec![0.1f32; batch * seq_q * dim];
            let k_data = vec![0.2f32; batch * seq_k * dim];
            let v_data = vec![0.3f32; batch * seq_k * dim];

            let q_tensor = Tensor::from_f32_slice(vec![batch, seq_q, dim], metallic::TensorStorage::Pooled(&mut ctx), &q_data).unwrap();

            let k_tensor = Tensor::from_f32_slice(vec![batch, seq_k, dim], metallic::TensorStorage::Pooled(&mut ctx), &k_data).unwrap();

            let v_tensor = Tensor::from_f32_slice(vec![batch, seq_k, dim], metallic::TensorStorage::Pooled(&mut ctx), &v_data).unwrap();

            // Run SDPA operation
            let _result = black_box(
                ctx.call::<SdpaMpsGraphOp>((&q_tensor, &k_tensor, &v_tensor, true, 0), None)
                    .unwrap(),
            );
        })
    });

    group.bench_function("fused_sdpa_projection", |b| {
        b.iter(|| {
            // For now, this is a placeholder - in a real implementation we would have a fused kernel
            // that runs SDPA + projection in a single graph operation
            // Here we're just showing the structure

            let mut ctx = Context::<F16>::new().unwrap();

            // Create sample tensors using f32 data (which gets converted to F16)
            let q_data = vec![0.1f32; batch * seq_q * dim];
            let k_data = vec![0.2f32; batch * seq_k * dim];
            let v_data = vec![0.3f32; batch * seq_k * dim];
            let proj_data = vec![0.4f32; dim * output_dim];

            let q_tensor = Tensor::from_f32_slice(vec![batch, seq_q, dim], metallic::TensorStorage::Pooled(&mut ctx), &q_data).unwrap();

            let k_tensor = Tensor::from_f32_slice(vec![batch, seq_k, dim], metallic::TensorStorage::Pooled(&mut ctx), &k_data).unwrap();

            let v_tensor = Tensor::from_f32_slice(vec![batch, seq_k, dim], metallic::TensorStorage::Pooled(&mut ctx), &v_data).unwrap();

            let _proj_weight =
                Tensor::from_f32_slice(vec![dim, output_dim], metallic::TensorStorage::Pooled(&mut ctx), &proj_data).unwrap();

            // In a real implementation, this would call a fused kernel
            // that performs both SDPA and projection in a single graph
            // For now, simulate the performance characteristics
            let _result = black_box(
                ctx.call::<SdpaMpsGraphOp>((&q_tensor, &k_tensor, &v_tensor, true, 0), None)
                    .unwrap(),
            );
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_sdpa_vs_fused);
criterion_main!(benches);
