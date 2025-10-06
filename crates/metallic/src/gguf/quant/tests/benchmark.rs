#![cfg(test)]
use super::*;

/// Benchmark the Q8 dequantization performance
#[test]
pub fn benchmark_q8_dequantization() {
    println!("Starting Q8 dequantization benchmark...");

    // Get initial memory usage
    let initial_memory = get_memory_usage();
    println!("Initial memory usage: {} bytes", initial_memory);

    // Create a synthetic Q8_0 block for testing
    // Each block is 34 bytes: 2 bytes scale + 32 bytes weights
    let mut data = Vec::new();
    let num_blocks = 1000; // Test with 1000 blocks

    // Create synthetic data
    for i in 0..num_blocks {
        // Scale (2 bytes as f16)
        let scale_bytes = (i as u16).to_le_bytes();
        data.extend_from_slice(&scale_bytes);

        // 32 weights (32 bytes)
        for j in 0..32 {
            data.push((i * 32 + j) as u8);
        }
    }

    // Get memory usage before dequantization
    let memory_before = get_memory_usage();
    println!("Memory before dequantization: {} bytes", memory_before);

    // Benchmark the regular dequantization
    let start = Instant::now();
    let result = q8::dequantize_q8_to_f32(&data, GGUFDataType::Q8_0);
    let duration = start.elapsed();

    // Get memory usage after dequantization
    let memory_after = get_memory_usage();
    println!("Memory after dequantization: {} bytes", memory_after);

    match result {
        Ok(f32_data) => {
            let peak_memory = memory_after.saturating_sub(memory_before);
            println!("Q8_0 regular dequantization benchmark:");
            println!("  - Blocks processed: {}", num_blocks);
            println!("  - Weights processed: {}", f32_data.len());
            println!("  - Time taken: {:?}", duration);
            println!(
                "  - Throughput: {:.2} weights/second",
                f32_data.len() as f64 / duration.as_secs_f64()
            );
            println!("  - Memory usage during dequantization: {} bytes", peak_memory);
            println!("  - Memory per weight: {:.2} bytes", peak_memory as f64 / f32_data.len() as f64);

            // Print first few values for verification
            println!("  - First 10 values: {:?}", &f32_data[..10.min(f32_data.len())]);
        }
        Err(e) => {
            println!("Error during benchmark: {}", e);
        }
    }

    // Benchmark SIMD version if available
    #[cfg(target_arch = "aarch64")]
    {
        let memory_before = get_memory_usage();
        let start = Instant::now();
        let result = q8_simd::dequantize_q8_to_f32_simd(&data, GGUFDataType::Q8_0);
        let duration = start.elapsed();
        let memory_after = get_memory_usage();

        match result {
            Ok(f32_data) => {
                let peak_memory = memory_after.saturating_sub(memory_before);
                println!("Q8_0 SIMD dequantization benchmark:");
                println!("  - Blocks processed: {}", num_blocks);
                println!("  - Weights processed: {}", f32_data.len());
                println!("  - Time taken: {:?}", duration);
                println!(
                    "  - Throughput: {:.2} weights/second",
                    f32_data.len() as f64 / duration.as_secs_f64()
                );
                println!("  - Memory usage during dequantization: {} bytes", peak_memory);
                println!("  - Memory per weight: {:.2} bytes", peak_memory as f64 / f32_data.len() as f64);

                // Print first few values for verification
                println!("  - First 10 values: {:?}", &f32_data[..10.min(f32_data.len())]);
            }
            Err(e) => {
                println!("Error during SIMD benchmark: {}", e);
            }
        }
    }

    // Also test Q8_1
    data.clear();
    for i in 0..num_blocks {
        // Scale (2 bytes as f16)
        let scale_bytes = (i as u16).to_le_bytes();
        data.extend_from_slice(&scale_bytes);

        // Delta (2 bytes as f16)
        let delta_bytes = ((i + 1000) as u16).to_le_bytes();
        data.extend_from_slice(&delta_bytes);

        // 32 weights (32 bytes)
        for j in 0..32 {
            data.push((i * 32 + j) as u8);
        }
    }

    // Get memory usage before dequantization
    let memory_before = get_memory_usage();

    let start = Instant::now();
    let result = q8::dequantize_q8_to_f32(&data, GGUFDataType::Q8_1);
    let duration = start.elapsed();

    // Get memory usage after dequantization
    let memory_after = get_memory_usage();

    match result {
        Ok(f32_data) => {
            let peak_memory = memory_after.saturating_sub(memory_before);
            println!("Q8_1 regular dequantization benchmark:");
            println!("  - Blocks processed: {}", num_blocks);
            println!("  - Weights processed: {}", f32_data.len());
            println!("  - Time taken: {:?}", duration);
            println!(
                "  - Throughput: {:.2} weights/second",
                f32_data.len() as f64 / duration.as_secs_f64()
            );
            println!("  - Memory usage during dequantization: {} bytes", peak_memory);
            println!("  - Memory per weight: {:.2} bytes", peak_memory as f64 / f32_data.len() as f64);

            // Print first few values for verification
            println!("  - First 10 values: {:?}", &f32_data[..10.min(f32_data.len())]);
        }
        Err(e) => {
            println!("Error during benchmark: {}", e);
        }
    }

    // Benchmark SIMD version if available
    #[cfg(target_arch = "aarch64")]
    {
        let memory_before = get_memory_usage();
        let start = Instant::now();
        let result = q8_simd::dequantize_q8_to_f32_simd(&data, GGUFDataType::Q8_1);
        let duration = start.elapsed();
        let memory_after = get_memory_usage();

        match result {
            Ok(f32_data) => {
                let peak_memory = memory_after.saturating_sub(memory_before);
                println!("Q8_1 SIMD dequantization benchmark:");
                println!("  - Blocks processed: {}", num_blocks);
                println!("  - Weights processed: {}", f32_data.len());
                println!("  - Time taken: {:?}", duration);
                println!(
                    "  - Throughput: {:.2} weights/second",
                    f32_data.len() as f64 / duration.as_secs_f64()
                );
                println!("  - Memory usage during dequantization: {} bytes", peak_memory);
                println!("  - Memory per weight: {:.2} bytes", peak_memory as f64 / f32_data.len() as f64);

                // Print first few values for verification
                println!("  - First 10 values: {:?}", &f32_data[..10.min(f32_data.len())]);
            }
            Err(e) => {
                println!("Error during SIMD benchmark: {}", e);
            }
        }
    }

    let final_memory = get_memory_usage();
    println!("Final memory usage: {} bytes", final_memory);
    println!(
        "Memory difference from start: {} bytes",
        final_memory.saturating_sub(initial_memory)
    );
}
