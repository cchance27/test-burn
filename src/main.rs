use crate::{
    benchmark::{benchmark_burn, benchmark_metal, benchmark_metallic},
    gguf::{GGUFFile, GGUFValue},
};

mod benchmark;
mod gguf;
mod metallic;
mod sdpa_burn;
mod sdpa_metal;

fn benchmark() {
    println!("\nRunning Metal Opt (MPS) implementation (causal)...");
    benchmark_metallic(true);
    benchmark_metallic(true);
    benchmark_metallic(true);
    benchmark_metallic(true);
    benchmark_metallic(true);

    println!("\nRunning Metal Opt (MPS) implementation (non-causal)...");
    benchmark_metallic(false);
    benchmark_metallic(false);
    benchmark_metallic(false);
    benchmark_metallic(false);
    benchmark_metallic(false);

    println!("\nRunning Metal (MPS) implementation...");
    benchmark_metal();
    benchmark_metal();
    benchmark_metal();
    benchmark_metal();
    benchmark_metal();

    type MyBackend = burn::backend::Metal;
    let device = <MyBackend as burn::prelude::Backend>::Device::default();
    println!("\nRunning Burn implementation...");
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);

    // Run our custom benchmarks
    println!("\nRunning SDPA Custom Benchmarks...");
    if let Err(e) = benchmark::run_benchmarks() {
        eprintln!("Benchmark error: {}", e);
    }
}

fn test_load_gguf_file() {
    let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load(path) {
        Ok(gguf) => {
            println!("Successfully loaded GGUF file:");
            println!(
                "Magic: {:?}",
                std::str::from_utf8(&gguf.header.magic).unwrap_or("Invalid UTF-8")
            );
            println!("Version: {}", gguf.header.version);
            println!("Tensor count: {}", gguf.header.tensor_count);
            println!("Metadata count: {}", gguf.header.metadata_count);

            println!("Metadata entries: {}", gguf.metadata.entries.len());
            for (key, value) in &gguf.metadata.entries {
                if key == "tokenizer.ggml.tokens" {
                    // For the tokenizer, just print the count to avoid huge output
                    if let GGUFValue::Array(tokens) = value {
                        println!("  {}: Array with {} elements", key, tokens.len());
                    } else {
                        println!("  {}: {:?}", key, value);
                    }
                } else {
                    // Print other metadata normally
                    println!("  {}: {:?}", key, value);
                }
            }

            // Print tensor count and first few tensor names
            println!("First 10 tensors:");
            let mut count = 0;
            for tensor in &gguf.tensors {
                if count < 10 {
                    println!(
                        "  {}: {:?} ({:?})",
                        tensor.name, tensor.dimensions, tensor.data_type
                    );
                    count += 1;
                } else {
                    break;
                }
            }
            println!(
                "... and {} more tensors",
                gguf.tensors.len().saturating_sub(10)
            );
        }
        Err(e) => {
            panic!("Failed to load GGUF file: {:?}", e);
        }
    }
}

fn main() {
    // Test GGUF loading
    test_load_gguf_file();
}
