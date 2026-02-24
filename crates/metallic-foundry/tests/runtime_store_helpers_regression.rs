use half::f16;
use metallic_foundry::{
    Foundry, Includes, Kernel, KernelSource, compound, fusion, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::{self, ComputeCommandEncoder, DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};
use metallic_macros::KernelArgs;
use serial_test::serial;

#[derive(Debug, KernelArgs)]
struct RuntimeStoreHelpersArgs {
    #[arg(buffer = 0, output, metal_type = "device OutputStorageT*")]
    output: TensorArg,
}

#[derive(Debug)]
struct RuntimeStoreHelpersKernel {
    args: RuntimeStoreHelpersArgs,
}

impl Kernel for RuntimeStoreHelpersKernel {
    type Args = RuntimeStoreHelpersArgs;

    fn function_name(&self) -> &str {
        "runtime_store_helpers_regression"
    }

    fn source(&self) -> KernelSource {
        KernelSource::String(
            r#"
#include <metal_stdlib>
using namespace metal;

kernel void runtime_store_helpers_regression(
    device OutputStorageT* output [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x != 0) return;

    for (uint i = 0; i < 16; ++i) {
        metallic_store_output(output, i, (AccumT)0.0f);
    }

    // Strided writes must respect explicit indices.
    metallic_store_output2(output, 0u, 4u, float2(1.0f, 2.0f));
    metallic_store_output4(output, 1u, 3u, 5u, 7u, float4(3.0f, 4.0f, 5.0f, 6.0f));

    // Contiguous writes should use packed fast paths when available.
    metallic_store_output2_contig(output, 8u, float2(7.0f, 8.0f));
    metallic_store_output(output, 12u, (AccumT)9.0f);
    metallic_store_output(output, 13u, (AccumT)10.0f);
    metallic_store_output(output, 14u, (AccumT)11.0f);
    metallic_store_output(output, 15u, (AccumT)12.0f);
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
        DispatchConfig {
            grid: GridSize::d1(1),
            group: ThreadgroupSize::d1(1),
        }
    }
}

#[test]
#[serial]
fn runtime_store_helpers_respect_strided_and_contiguous_indices() -> Result<(), Box<dyn std::error::Error>> {
    let mut foundry = Foundry::new()?;
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![16], TensorInit::Uninitialized)?;

    let kernel = RuntimeStoreHelpersKernel {
        args: RuntimeStoreHelpersArgs {
            output: TensorArg::from_tensor(&output),
        },
    };
    foundry.run(&kernel)?;
    foundry.synchronize()?;

    let got: Vec<f16> = output.to_vec(&foundry);
    let got_f32: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
    let expected: [f32; 16] = [1.0, 3.0, 0.0, 4.0, 2.0, 5.0, 0.0, 6.0, 7.0, 8.0, 0.0, 0.0, 9.0, 10.0, 11.0, 12.0];

    for (i, (&g, &e)) in got_f32.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(diff <= 1e-3, "mismatch at idx={i}: got={g}, expected={e}, diff={diff}");
    }

    Ok(())
}
