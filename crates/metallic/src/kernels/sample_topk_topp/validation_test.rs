use std::time::Instant;

use crate::generation::{gpu_sample_top_k_top_p, sample_top_k_top_p};
#[cfg(test)]
use crate::{Context, F32Element, MetalError, SamplerBuffers, Tensor, TensorElement, TensorInit, TensorStorage};

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
fn benchmark_cpu_vs_gpu_with_sync_overhead() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    let vocab_size = 32000; // Similar to Qwen2.5 vocab size
    let test_logits = create_test_logits(&mut ctx, vocab_size)?;

    // Set realistic sampling parameters
    let k = 50;
    let top_p = 0.9;
    let temperature = 0.8;

    // Benchmark CPU sampling WITH sync time (the real bottleneck)
    let cpu_iterations = 5;
    let mut cpu_times = Vec::new();

    for _ in 0..cpu_iterations {
        let start = Instant::now();

        // This simulates the real CPU path: sync + sampling
        let logits_slice = test_logits.as_slice(); // This triggers the sync!
        let vocab_logits = &logits_slice[..vocab_size];
        let mut sampler_buffers = SamplerBuffers::default();
        let _cpu_token = sample_top_k_top_p::<F32Element>(vocab_logits, k, top_p, temperature, &mut sampler_buffers) as u32;

        let duration = start.elapsed();
        cpu_times.push(duration);
    }

    let avg_cpu_with_sync_time: std::time::Duration = cpu_times.iter().sum::<std::time::Duration>() / cpu_iterations as u32;

    // Benchmark GPU sampling (no sync needed for main computation, only to read result token)
    let gpu_iterations = 5;
    let mut gpu_times = Vec::new();

    for _ in 0..gpu_iterations {
        let start = Instant::now();

        // GPU sampling - no sync for main computation, only to read single token
        let _gpu_token = gpu_sample_top_k_top_p::<F32Element>(&test_logits, vocab_size, k, top_p, temperature, &mut ctx)?;

        let duration = start.elapsed();
        gpu_times.push(duration);
    }

    let avg_gpu_time: std::time::Duration = gpu_times.iter().sum::<std::time::Duration>() / gpu_iterations as u32;

    println!("CPU Sampling (with GPU->CPU sync): {:?}", avg_cpu_with_sync_time);
    println!("GPU Sampling (no major sync):     {:?}", avg_gpu_time);

    // Calculate the real performance benefit
    if avg_cpu_with_sync_time > avg_gpu_time {
        let speedup = avg_cpu_with_sync_time.as_millis() as f64 / avg_gpu_time.as_millis() as f64;
        println!("GPU is {:.2}x faster when considering sync overhead!", speedup);
    } else {
        let slowdown = avg_gpu_time.as_millis() as f64 / avg_cpu_with_sync_time.as_millis() as f64;
        println!("CPU is {:.2}x faster, but GPU avoids sync for subsequent calls!", slowdown);
    }

    Ok(())
}

#[test]
fn benchmark_sync_time_simulation() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Test different vocab sizes to show how sync time scales
    let vocab_sizes = vec![1000, 5000, 10000, 32000];

    for &vocab_size in &vocab_sizes {
        let test_logits = create_test_logits(&mut ctx, vocab_size)?;

        // Time the sync operation alone (this is the bottleneck)
        let start = Instant::now();
        let _logits_slice = test_logits.as_slice(); // This triggers sync
        let sync_time = start.elapsed();

        let tensor_size_bytes = vocab_size * std::mem::size_of::<f32>();
        println!(
            "Vocab size: {:>6}, Tensor size: {:>6} KB, Sync time: {:?}",
            vocab_size,
            tensor_size_bytes / 1024,
            sync_time
        );
    }

    Ok(())
}

#[test]
fn test_deterministic_sampling_comparison() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Use simpler, more deterministic data for better comparison
    let vocab_size = 100;
    let test_logits = create_test_logits(&mut ctx, vocab_size)?;

    // Use temperature = 0 for deterministic behavior (always pick highest logit)
    let k = 5;
    let top_p = 1.0; // No top-p filtering
    let temperature = 0.001; // Near 0 for more deterministic behavior

    // Run multiple times and compare distribution patterns
    let num_trials = 100;
    let mut cpu_tokens = Vec::new();
    let mut gpu_tokens = Vec::new();

    for _trial in 0..num_trials {
        // CPU path
        let logits_slice = test_logits.as_slice();
        let vocab_logits = &logits_slice[..vocab_size];
        let mut sampler_buffers = SamplerBuffers::default();
        let cpu_token = sample_top_k_top_p::<F32Element>(vocab_logits, k, top_p, temperature, &mut sampler_buffers) as u32;
        cpu_tokens.push(cpu_token);

        // GPU path
        let gpu_token = gpu_sample_top_k_top_p::<F32Element>(&test_logits, vocab_size, k, top_p, temperature, &mut ctx)?;
        gpu_tokens.push(gpu_token);
    }

    // Count occurrences of top choices for both methods
    use std::collections::HashMap;
    let mut cpu_counts = HashMap::new();
    let mut gpu_counts = HashMap::new();

    for &token in &cpu_tokens {
        *cpu_counts.entry(token).or_insert(0) += 1;
    }

    for &token in &gpu_tokens {
        *gpu_counts.entry(token).or_insert(0) += 1;
    }

    // Display top tokens for each method
    println!("CPU Token Distribution (top 5):");
    let mut cpu_sorted: Vec<_> = cpu_counts.into_iter().collect();
    cpu_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, (token, count)) in cpu_sorted.iter().take(5).enumerate() {
        println!("  {}: token {} appeared {} times", i + 1, token, count);
    }

    println!("\nGPU Token Distribution (top 5):");
    let mut gpu_sorted: Vec<_> = gpu_counts.into_iter().collect();
    gpu_sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (i, (token, count)) in gpu_sorted.iter().take(5).enumerate() {
        println!("  {}: token {} appeared {} times", i + 1, token, count);
    }

    Ok(())
}

// Test to debug the issue with GPU sampler returning invalid tokens
#[test]
fn debug_gpu_sampler_issue() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Test with a smaller vocab to debug
    let vocab_size = 1000;
    let test_logits = create_test_logits(&mut ctx, vocab_size)?;
    
    // Use parameters similar to the failing case
    let k = 40;  // Default top-k
    let top_p = 0.95;  // Default top-p
    let temperature = 1.0;  // Default temperature

    println!("Testing GPU sampling with vocab_size={}, k={}, top_p={}, temperature={}", 
             vocab_size, k, top_p, temperature);
    
    // Multiple test runs to check for consistency
    for i in 0..10 {
        let gpu_token = gpu_sample_top_k_top_p::<F32Element>(&test_logits, vocab_size, k, top_p, temperature, &mut ctx)?;
        println!("Run {}: GPU token = {}, token as char = {:?}", i, gpu_token, char::from_u32(gpu_token));
        
        // Validate the token is within expected range
        assert!(gpu_token < vocab_size as u32, "GPU token {} is out of range for vocab size {}", gpu_token, vocab_size);
    }

    Ok(())
}

// Test to compare GPU vs CPU sampling results
#[test]
fn compare_gpu_vs_cpu_sampling() -> Result<(), MetalError> {
    let mut ctx = Context::<F32Element>::new()?;

    // Test with a smaller vocab to debug
    let vocab_size = 100;
    let test_logits = create_test_logits(&mut ctx, vocab_size)?;
    
    // Use parameters similar to the failing case
    let k = 10;  // Smaller k for easier verification
    let top_p = 0.9;  // Smaller top_p for focused sampling
    let temperature = 1.0;  // Default temperature

    println!("Comparing GPU vs CPU sampling with vocab_size={}, k={}, top_p={}, temperature={}", 
             vocab_size, k, top_p, temperature);
    
    // CPU sampling
    let logits_slice = test_logits.as_slice();
    let vocab_logits = &logits_slice[..vocab_size];
    let mut sampler_buffers = SamplerBuffers::default();
    let cpu_token = sample_top_k_top_p::<F32Element>(vocab_logits, k, top_p, temperature, &mut sampler_buffers) as u32;
    
    // GPU sampling (multiple times to check consistency)
    for i in 0..5 {
        let gpu_token = gpu_sample_top_k_top_p::<F32Element>(&test_logits, vocab_size, k, top_p, temperature, &mut ctx)?;
        println!("Run {}: CPU token = {}, GPU token = {}", i, cpu_token, gpu_token);
        
        // Both should be within range
        assert!(cpu_token < vocab_size as u32, "CPU token {} is out of range for vocab size {}", cpu_token, vocab_size);
        assert!(gpu_token < vocab_size as u32, "GPU token {} is out of range for vocab size {}", gpu_token, vocab_size);
    }

    Ok(())
}
