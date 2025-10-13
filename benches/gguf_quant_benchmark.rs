//! Benchmark for the GGUF quantization implementations
//!
//! We run multiple iterations of each operation within each benchmark measurement
//! to reduce the impact of benchmark framework overhead and get more stable measurements.
//! Dequantization operations are very fast, so running multiple iterations helps
//! get more accurate performance measurements.

use criterion::{Criterion, criterion_group, criterion_main};
use metallic::gguf::GGUFDataType;
use metallic::gguf::quant::{q8, q8_simd};

/// Number of iterations to run within each benchmark measurement
/// This helps reduce measurement noise for very fast operations like dequantization
const ITERATIONS: usize = 100;

fn benchmark_q8_dequantization(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("q8_dequantization");

    // Benchmark the regular dequantization
    group.bench_function("q8_0_regular", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                let _result = q8::dequantize_q8_to_f32(&data, GGUFDataType::Q8_0);
            }
        })
    });

    // Benchmark SIMD version if available
    #[cfg(target_arch = "aarch64")]
    group.bench_function("q8_0_simd", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                let _result = q8_simd::dequantize_q8_to_f32_simd(&data, GGUFDataType::Q8_0);
            }
        })
    });

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

    // Benchmark the regular dequantization for Q8_1
    group.bench_function("q8_1_regular", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                let _result = q8::dequantize_q8_to_f32(&data, GGUFDataType::Q8_1);
            }
        })
    });

    // Benchmark SIMD version if available for Q8_1
    #[cfg(target_arch = "aarch64")]
    group.bench_function("q8_1_simd", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                let _result = q8_simd::dequantize_q8_to_f32_simd(&data, GGUFDataType::Q8_1);
            }
        })
    });

    group.finish();
}

fn benchmark_q8_large_dequantization(c: &mut Criterion) {
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

    let mut group = c.benchmark_group("q8_large_dequantization");

    // Benchmark the regular dequantization
    group.bench_function("q8_0_regular_large", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                let _result = q8::dequantize_q8_to_f32(&data, GGUFDataType::Q8_0);
            }
        })
    });

    // Benchmark SIMD version if available
    #[cfg(target_arch = "aarch64")]
    group.bench_function("q8_0_simd_large", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                let _result = q8_simd::dequantize_q8_to_f32_simd(&data, GGUFDataType::Q8_0);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, benchmark_q8_dequantization, benchmark_q8_large_dequantization);
criterion_main!(benches);
