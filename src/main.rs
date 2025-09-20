use crate::benchmark::{benchmark_burn, benchmark_metal, benchmark_metallic};

mod benchmark;
mod metallic;
mod sdpa_burn;
mod sdpa_metal;

fn main() {
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
