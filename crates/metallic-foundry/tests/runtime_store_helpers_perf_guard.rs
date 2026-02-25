use std::time::Instant;

use metallic_foundry::{
    Foundry, Includes, Kernel, KernelSource, MetalError, compound, fusion, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::{self, ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};
use metallic_macros::KernelArgs;

#[derive(Debug, KernelArgs)]
struct StoreBenchArgs {
    #[arg(buffer = 0, output, metal_type = "device OutputStorageT*")]
    output: TensorArg,
    #[arg(buffer = 1)]
    n: u32,
}

#[derive(Debug)]
struct StoreScalarKernel {
    args: StoreBenchArgs,
}

impl Kernel for StoreScalarKernel {
    type Args = StoreBenchArgs;

    fn function_name(&self) -> &str {
        "runtime_store_helper_scalar_bench"
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void runtime_store_helper_scalar_bench(
    device OutputStorageT* output [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint t = gid.x;
    const uint base = t * 4u;
    if (base + 3u >= n) return;

    metallic_store_output(output, base + 0u, (AccumT)1.0f);
    metallic_store_output(output, base + 1u, (AccumT)2.0f);
    metallic_store_output(output, base + 2u, (AccumT)3.0f);
    metallic_store_output(output, base + 3u, (AccumT)4.0f);
}
"#
            .to_string(),
        )
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.args.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        let lanes = ((self.args.n as usize) / 4).max(1);
        DispatchConfig {
            grid: GridSize::d1(lanes),
            group: ThreadgroupSize::d1(256),
        }
    }
}

#[derive(Debug)]
struct StoreVectorKernel {
    args: StoreBenchArgs,
}

impl Kernel for StoreVectorKernel {
    type Args = StoreBenchArgs;

    fn function_name(&self) -> &str {
        "runtime_store_helper_vector_bench"
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void runtime_store_helper_vector_bench(
    device OutputStorageT* output [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint t = gid.x;
    const uint base = t * 4u;
    if (base + 3u >= n) return;
    // Indexed contiguous path should auto-route to contiguous helper lanes.
    metallic_store_output4(output, base + 0u, base + 1u, base + 2u, base + 3u, float4(1.0f, 2.0f, 3.0f, 4.0f));
}
"#
            .to_string(),
        )
    }

    fn includes(&self) -> Includes {
        Includes(vec![])
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.args.bind_args(encoder);
    }

    fn dispatch_config(&self) -> DispatchConfig {
        let lanes = ((self.args.n as usize) / 4).max(1);
        DispatchConfig {
            grid: GridSize::d1(lanes),
            group: ThreadgroupSize::d1(256),
        }
    }
}

fn parse_env_usize(key: &'static str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}

fn measure_us(foundry: &mut Foundry, kernel: &impl Kernel, warmup: usize, iters: usize) -> Result<f64, MetalError> {
    for _ in 0..warmup {
        foundry.run(kernel)?;
    }
    foundry.synchronize()?;

    let start = Instant::now();
    foundry.start_capture()?;
    for _ in 0..iters {
        foundry.run(kernel)?;
    }
    let cb = foundry.end_capture()?;
    cb.wait_until_completed();
    let elapsed = start.elapsed().as_micros() as f64;
    Ok(elapsed / iters as f64)
}

#[test]
#[ignore]
fn runtime_store_helpers_perf_non_regression() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = match Foundry::new() {
        Ok(v) => v,
        Err(MetalError::DeviceNotFound) => {
            eprintln!("Skipping runtime_store_helpers_perf_non_regression: Metal device unavailable");
            return Ok(());
        }
        Err(e) => return Err(Box::new(e)),
    };

    let n = parse_env_usize("METALLIC_STORE_HELPER_BENCH_LEN", 262_144);
    let warmup = parse_env_usize("METALLIC_STORE_HELPER_BENCH_WARMUP", 10);
    let iters = parse_env_usize("METALLIC_STORE_HELPER_BENCH_ITERS", 100);
    let max_regress_pct = std::env::var("METALLIC_STORE_HELPER_MAX_REGRESS_PCT")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .unwrap_or(20.0);

    println!("runtime_store_helper perf guard: n={n}, warmup={warmup}, iters={iters}, max_regress_pct={max_regress_pct:.2}");

    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![n], TensorInit::Uninitialized)?;
    let scalar = StoreScalarKernel {
        args: StoreBenchArgs {
            output: TensorArg::from_tensor(&output),
            n: n as u32,
        },
    };
    let vector = StoreVectorKernel {
        args: StoreBenchArgs {
            output: TensorArg::from_tensor(&output),
            n: n as u32,
        },
    };

    let scalar_us = measure_us(&mut foundry, &scalar, warmup, iters)?;
    let vector_us = measure_us(&mut foundry, &vector, warmup, iters)?;
    let regress_pct = ((vector_us - scalar_us) / scalar_us) * 100.0;

    println!("  scalar={scalar_us:.2} us, vector(auto-contig)={vector_us:.2} us, regress={regress_pct:+.2}%");

    assert!(
        regress_pct <= max_regress_pct,
        "vector helper regression too high: {regress_pct:+.2}% > +{max_regress_pct:.2}% (scalar={scalar_us:.2} us, vector={vector_us:.2} us)"
    );

    Ok(())
}
