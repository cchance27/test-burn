use super::*;

/// Benchmark the Q8 dequantization performance with large data
#[test]
pub fn benchmark_q8_large_dequantization() {
    println!("Starting Q8 large dequantization benchmark...");

    // Create a synthetic Q8_0 block for testing with larger data
    // Each block is 34 bytes: 2 bytes scale + 32 bytes weights
    let mut data = Vec::new();
    let num_blocks = 100000; // Test with 100,000 blocks (3.2M weights)

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

    println!("Created {} blocks with {} total weights", num_blocks, num_blocks * 32);

    // Benchmark the regular dequantization
    let start = Instant::now();
    let result = q8::dequantize_q8_to_f32(&data, GGUFDataType::Q8_0);
    let duration = start.elapsed();

    match result {
        Ok(f32_data) => {
            println!("Q8_0 regular dequantization benchmark:");
            println!("  - Blocks processed: {}", num_blocks);
            println!("  - Weights processed: {}", f32_data.len());
            println!("  - Time taken: {:?}", duration);
            println!(
                "  - Throughput: {:.2} weights/second",
                f32_data.len() as f64 / duration.as_secs_f64()
            );
        }
        Err(e) => {
            println!("Error during benchmark: {}", e);
        }
    }

    // Benchmark SIMD version if available
    #[cfg(target_arch = "aarch64")]
    {
        let start = Instant::now();
        let result = q8_simd::dequantize_q8_to_f32_simd(&data, GGUFDataType::Q8_0);
        let duration = start.elapsed();

        match result {
            Ok(f32_data) => {
                println!("Q8_0 SIMD dequantization benchmark:");
                println!("  - Blocks processed: {}", num_blocks);
                println!("  - Weights processed: {}", f32_data.len());
                println!("  - Time taken: {:?}", duration);
                println!(
                    "  - Throughput: {:.2} weights/second",
                    f32_data.len() as f64 / duration.as_secs_f64()
                );
            }
            Err(e) => {
                println!("Error during SIMD benchmark: {}", e);
            }
        }
    }
}
