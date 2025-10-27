use std::time::Instant;

#[cfg(test)]
use crate::{Context, MetalError, SamplerBuffers, Tensor, TensorInit, TensorStorage, kernels::elemwise_add::ElemwiseAddOp};
use crate::{
    F16Element, generation::{gpu_sample_top_k_top_p, sample_top_k_top_p}
};

#[cfg(test)]
fn create_test_logits<T: crate::TensorElement>(ctx: &mut Context<T>, size: usize) -> Result<Tensor<T>, MetalError> {
    // Create test logits with known values for reproducible testing
    let mut logits_data = Vec::with_capacity(size);
    for i in 0..size {
        // Create logits with decreasing values to have a clear top choice
        let val = (size - i) as f32 * 0.1; // Values from 0.1*size down to 0.1
        logits_data.push(T::from_f32(val));
    }

    Tensor::new(vec![size], TensorStorage::Pooled(ctx), TensorInit::CopyFrom(&logits_data))
}

#[test]
fn benchmark_cpu_vs_gpu_sampling() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;

    let vocab_size = 100000; // Similar to Qwen2.5 vocab size

    // Set realistic sampling parameters
    let k = 50;
    let top_p = 0.9;
    let temperature = 0.8;

    // Benchmark CPU sampling WITHOUT GPU operations (CPU-only baseline)
    let cpu_only_iterations = 1000;
    let mut cpu_only_times = Vec::new();

    println!("Starting CPU-only sampling benchmark (no GPU operations)...");
    // Create logits once in memory for CPU-only benchmark
    let mut logits_data = Vec::<half::f16>::with_capacity(vocab_size);
    for i in 0..vocab_size {
        let val = (vocab_size - i) as f32 * 0.1; // Values from 0.1*vocab_size down to 0.1
        logits_data.push(half::f16::from_f32(val));
    }

    for _ in 0..cpu_only_iterations {
        let start = Instant::now();
        let mut sampler_buffers = SamplerBuffers::default();
        let _cpu_token = sample_top_k_top_p::<F16Element>(&logits_data, k, top_p, temperature, &mut sampler_buffers) as u32;
        let duration = start.elapsed();
        cpu_only_times.push(duration);
    }

    let avg_cpu_only_time: std::time::Duration = cpu_only_times.iter().sum::<std::time::Duration>() / cpu_only_iterations as u32;

    // Create the test tensor once, do a GPU operation, then reuse it to better measure sync overhead
    let test_logits = create_test_logits(&mut ctx, vocab_size)?;

    // Perform a dummy GPU operation to ensure the tensor is truly GPU-resident
    // This simulates a real-world scenario where logits come from a GPU operation
    let ones = Tensor::ones(vec![vocab_size], &mut ctx)?;

    // Perform a dummy add operation using the kernels
    let new = ctx.call::<ElemwiseAddOp>((test_logits.clone(), ones.clone()))?;
    ctx.synchronize(); // Ensure the add operation completes before measuring

    // Benchmark the sync operation separately to understand overhead
    let sync_iterations = 1000;
    let mut sync_times = Vec::new();

    println!("Measuring GPU->CPU sync times...");
    for _ in 0..sync_iterations {
        // Perform another GPU operation to make sure tensor is GPU-resident again
        let new = ctx.call::<ElemwiseAddOp>((new.clone(), ones.clone()))?;
        ctx.synchronize(); // Complete the operation

        let sync_start = Instant::now();
        let _logits_slice = new.as_slice(); // This is where the sync happens
        let duration = sync_start.elapsed();
        sync_times.push(duration);
    }

    let avg_sync_time: std::time::Duration = sync_times.iter().sum::<std::time::Duration>() / sync_iterations as u32;

    // Benchmark CPU sampling WITH GPU->CPU sync overhead (real-world scenario)
    let cpu_iterations = 1000;
    let mut cpu_total_times = Vec::new();
    let ones = Tensor::ones(vec![vocab_size], &mut ctx)?;

    let test_logits = create_test_logits(&mut ctx, vocab_size)?;

    println!("Starting CPU sampling benchmark (including GPU->CPU sync)...");
    for i in 0..cpu_iterations {
        // Perform a GPU operation to make sure tensor is GPU-resident
        let new = ctx.call::<ElemwiseAddOp>((test_logits.clone(), ones.clone()))?;
        ctx.synchronize(); // Complete the operation

        // Time the complete operation: GPU->CPU sync + CPU sampling
        let start = Instant::now();

        // This triggers GPU->CPU sync due to as_slice() call
        let logits_slice = new.as_slice();
        let vocab_logits = &logits_slice[..vocab_size];

        let mut sampler_buffers = SamplerBuffers::default();
        let _cpu_token = sample_top_k_top_p::<F16Element>(vocab_logits, k, top_p, temperature, &mut sampler_buffers) as u32;

        let duration = start.elapsed();
        cpu_total_times.push(duration);

        // Periodic resource cleanup to prevent GPU memory accumulation
        if i % 25 == 0 {
            // Every 25 iterations
            ctx.reset_pool(); // Reset tensor pool to free temporary GPU memory
        }
    }

    let avg_cpu_total_time: std::time::Duration = cpu_total_times.iter().sum::<std::time::Duration>() / cpu_iterations as u32;

    // Benchmark GPU sampling (no GPU->CPU sync needed beyond result token)
    println!("Starting GPU sampling benchmark (no GPU->CPU sync needed for logits)...");
    let gpu_iterations = 1000;
    let mut gpu_times = Vec::new();

    let ones = Tensor::ones(vec![vocab_size], &mut ctx)?;

    let test_logits = create_test_logits(&mut ctx, vocab_size)?;

    for i in 0..gpu_iterations {
        // Perform a GPU operation to make sure tensor is GPU-resident
        let new = ctx.call::<ElemwiseAddOp>((test_logits.clone(), ones.clone()))?;
        ctx.synchronize(); // Complete the operation

        let start = Instant::now();
        let _gpu_token = gpu_sample_top_k_top_p::<F16Element>(&new, vocab_size, k, top_p, temperature, &mut ctx)?;
        let duration = start.elapsed();
        gpu_times.push(duration);

        // Periodic resource cleanup to prevent GPU memory accumulation
        if i % 25 == 0 {
            // Every 25 iterations
            ctx.reset_pool(); // Reset tensor pool to free temporary GPU memory
        }
    }

    let avg_gpu_time: std::time::Duration = gpu_times.iter().sum::<std::time::Duration>() / gpu_iterations as u32;

    // Ensure GPU operations complete and cleanup resources
    ctx.synchronize();
    ctx.reset_pool();

    println!("CPU Sampling (CPU-only baseline) - Average time: {:?}", avg_cpu_only_time);
    println!("GPU->CPU sync only - Average time: {:?}", avg_sync_time);
    println!("CPU Sampling (with GPU->CPU sync) - Average time: {:?}", avg_cpu_total_time);
    println!("GPU Sampling (no GPU->CPU sync for logits) - Average time: {:?}", avg_gpu_time);

    // Calculate the time breakdown
    let cpu_sync_ns = avg_sync_time.as_nanos() as f64;
    let cpu_total_with_sync_ns = avg_cpu_total_time.as_nanos() as f64;
    let cpu_only_ns = avg_cpu_only_time.as_nanos() as f64;
    let gpu_ns = avg_gpu_time.as_nanos() as f64;

    println!("Estimated GPU->CPU sync overhead: {:.2}µs", cpu_sync_ns / 1000.0);

    if gpu_ns > 0.0 {
        let cpu_vs_gpu_speedup = cpu_total_with_sync_ns / gpu_ns;
        println!(
            "GPU sampling is {:.2}x faster than CPU sampling (including sync overhead)",
            cpu_vs_gpu_speedup
        );

        let cpu_only_vs_gpu_speedup = cpu_only_ns / gpu_ns;
        println!(
            "GPU sampling is {:.2}x faster than CPU sampling (CPU-only baseline)",
            cpu_only_vs_gpu_speedup
        );
    }

    Ok(())
}

#[test]
fn test_logits_download_overhead_simulation() {
    use std::time::Duration;

    let vocab_size = 150000;
    let element_size = std::mem::size_of::<f32>(); // 4 bytes for f32
    let tensor_size_bytes = vocab_size * element_size; // ~128KB
    let gbps_bandwidth = 50.0; // Typical GPU memory bandwidth in GB/s

    // Calculate rough sync time based on memory transfer
    let sync_time_estimate = (tensor_size_bytes as f64) / (gbps_bandwidth * 1_000_000_000.0);
    let sync_time = Duration::from_secs_f64(sync_time_estimate);

    println!(
        "Logits tensor size: {} bytes (~{:.1} KB)",
        tensor_size_bytes,
        tensor_size_bytes as f64 / 1024.0
    );
    println!("Estimated GPU->CPU sync time: {:?}", sync_time);
    println!("This overhead is eliminated with GPU-based sampling!");
}

#[test]
fn test_gpu_kernel_correctness_multiple_runs() -> Result<(), MetalError> {
    let mut ctx = Context::new()?;

    let vocab_size = 100;
    let test_logits = create_test_logits(&mut ctx, vocab_size)?;

    // Set fixed parameters
    let k = 10;
    let top_p = 0.9;
    let temperature = 1.0;

    // Generate multiple tokens to ensure the kernel works reliably
    for i in 0..5 {
        let token = gpu_sample_top_k_top_p::<F16Element>(&test_logits, vocab_size, k, top_p, temperature, &mut ctx)?;
        assert!(token < vocab_size as u32, "Token {} is out of range", token);
        println!("Run {}: Generated token {}", i + 1, token);
    }

    Ok(())
}
