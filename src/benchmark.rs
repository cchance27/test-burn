use crate::metallic::{Context, Tensor};
use burn::prelude::*;
use burn::tensor::{Distribution, Float, Tensor as BurnTensor};
use std::ffi::c_void;
use std::time::Instant;
use test_burn::alternatives::sdpa_burn::scaled_dot_product_attention_burn;
use test_burn::alternatives::sdpa_metal::scaled_dot_product_attention_metal;

const ITERATIONS: usize = 100;

fn benchmark_sdpa(
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    dim: usize,
    iterations: usize,
    causal: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Benchmarking SDPA with batch={}, seq_q={}, seq_k={}, dim={}, iterations={}, causal={}",
        batch, seq_q, seq_k, dim, iterations, causal
    );

    // Create a context
    let mut context = Context::new()?;

    // Create test tensors
    let q_tensor = Tensor::arange(batch * seq_q * dim, vec![batch, seq_q, dim], &context)?;
    let k_tensor = Tensor::arange(batch * seq_k * dim, vec![batch, seq_k, dim], &context)?;
    let v_tensor = Tensor::arange(batch * seq_k * dim, vec![batch, seq_k, dim], &context)?;

    // Warm up
    println!("Warming up...");
    for _ in 0..5 {
        let _ = context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal)?;
    }

    // Actual benchmark
    println!("Running benchmark...");
    let start = Instant::now();
    for i in 0..iterations {
        let _output =
            context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal)?;
        if i % 10 == 0 {
            println!("Completed iteration {}", i);
        }
    }
    let duration = start.elapsed();

    println!("Time for {} iterations: {:.2?}", iterations, duration);
    println!(
        "Average time per iteration: {:.2?}",
        duration / iterations as u32
    );
    println!(
        "Iterations per second: {:.2}",
        iterations as f64 / duration.as_secs_f64()
    );

    Ok(())
}

pub fn run_benchmarks() -> Result<(), Box<dyn std::error::Error>> {
    // Run benchmarks with different configurations
    println!("=== SDPA Metal Performance Benchmark ===\n");

    // Small benchmark
    benchmark_sdpa(4, 128, 128, 64, 50, false)?;
    println!("\n----------------------------------------\n");

    // Medium benchmark
    benchmark_sdpa(4, 512, 512, 64, 20, false)?;
    println!("\n----------------------------------------\n");

    // Large benchmark
    benchmark_sdpa(4, 1024, 1024, 64, 10, false)?;
    println!("\n----------------------------------------\n");

    // Causal benchmark
    benchmark_sdpa(4, 512, 512, 64, 20, true)?;
    println!("\n----------------------------------------\n");

    println!("Benchmarking completed!");
    Ok(())
}

pub fn benchmark_burn<MyBackend: Backend>(device: &<MyBackend as Backend>::Device) {
    let batch = 32;
    let seq = 1024;
    let dim = 64;
    let query = BurnTensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let key = BurnTensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let value = BurnTensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    MyBackend::sync(device);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _output = scaled_dot_product_attention_burn(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            true,
        );
        MyBackend::sync(device);
    }
    let duration = start.elapsed();
    println!("Burn time for {ITERATIONS} iterations: {duration:?}");
}

pub fn benchmark_metal() {
    let batch = 32;
    let seq = 1024;
    let dim = 64;
    let query: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let key: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let value: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let q_ptr =
        std::ptr::NonNull::new(query.as_ptr() as *mut c_void).expect("query pointer is null");
    let k_ptr = std::ptr::NonNull::new(key.as_ptr() as *mut c_void).expect("key pointer is null");
    let v_ptr =
        std::ptr::NonNull::new(value.as_ptr() as *mut c_void).expect("value pointer is null");

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _output = scaled_dot_product_attention_metal(q_ptr, k_ptr, v_ptr, batch, seq, seq, dim);
    }
    let duration = start.elapsed();
    println!("Metal (MPS) time for {ITERATIONS} iterations: {duration:?}");
}

pub fn benchmark_metallic(causal: bool) {
    use crate::metallic::{Context, Tensor};

    let batch = 32;
    let seq = 1024;
    let dim = 64;

    let mut context = Context::new().unwrap();

    // Use Tensor::from_vec to populate tensors from random CPU data
    let query: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let key: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let value: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();

    let q_tensor = Tensor::from_vec(query, vec![batch, seq, dim], &context).unwrap();
    let k_tensor = Tensor::from_vec(key, vec![batch, seq, dim], &context).unwrap();
    let v_tensor = Tensor::from_vec(value, vec![batch, seq, dim], &context).unwrap();

    let start: Instant = Instant::now();
    for _ in 0..ITERATIONS {
        let _output = context
            .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal)
            .unwrap();
    }
    let duration = start.elapsed();
    println!("Metal Opt (MPS) time for {ITERATIONS} iterations: {duration:?}");
}
