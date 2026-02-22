use std::sync::Arc;

use half::f16;
use metallic_foundry::{
    Foundry, compound::CompoundKernel, metals::swiglu::{SwigluArgs, SwigluParamsResolved, stages::SwigluStage}, storage::Pooled, tensor::{F16, Tensor, TensorInit}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};
use rand::Rng;
use serial_test::serial;

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn run_cpu_swiglu(gate: &[f16], up: &[f16], gate_bias: &[f16], up_bias: &[f16]) -> Vec<f16> {
    gate.iter()
        .zip(up.iter())
        .enumerate()
        .map(|(i, (g, u))| {
            let col = i % gate_bias.len();
            let g_val = g.to_f32() + gate_bias[col].to_f32();
            let u_val = u.to_f32() + up_bias[col].to_f32();
            f16::from_f32(silu(g_val) * u_val)
        })
        .collect()
}

#[test]
#[serial]
fn test_swiglu_parity() {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rand::rng();

    let seq_len = 128;
    let hidden_dim = 1024;
    let total_elements = seq_len * hidden_dim;

    let mut gate_data = vec![f16::from_f32(0.0); total_elements];
    let mut up_data = vec![f16::from_f32(0.0); total_elements];
    let mut gate_bias_data = vec![f16::from_f32(0.0); hidden_dim];
    let mut up_bias_data = vec![f16::from_f32(0.0); hidden_dim];

    for i in 0..total_elements {
        gate_data[i] = f16::from_f32(rng.random_range(-2.0..2.0));
        up_data[i] = f16::from_f32(rng.random_range(-2.0..2.0));
    }
    for i in 0..hidden_dim {
        gate_bias_data[i] = f16::from_f32(rng.random_range(-0.5..0.5));
        up_bias_data[i] = f16::from_f32(rng.random_range(-0.5..0.5));
    }

    let cpu_result = run_cpu_swiglu(&gate_data, &up_data, &gate_bias_data, &up_bias_data);

    let gate_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![seq_len, hidden_dim], TensorInit::CopyFrom(&gate_data)).unwrap();
    let up_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![seq_len, hidden_dim], TensorInit::CopyFrom(&up_data)).unwrap();
    let gate_bias_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![hidden_dim], TensorInit::CopyFrom(&gate_bias_data)).unwrap();
    let up_bias_tensor = Tensor::<F16, Pooled>::new(&mut foundry, vec![hidden_dim], TensorInit::CopyFrom(&up_bias_data)).unwrap();

    let gate_arg = TensorArg::from_tensor(&gate_tensor);
    let up_arg = TensorArg::from_tensor(&up_tensor);
    let gate_bias_arg = TensorArg::from_tensor(&gate_bias_tensor);
    let up_bias_arg = TensorArg::from_tensor(&up_bias_tensor);

    let params = SwigluParamsResolved {
        total_elements: total_elements as u32,
        bias_len: hidden_dim as u32,
        vector_width: 4,
        gate_leading_stride: hidden_dim as u32,
        up_leading_stride: hidden_dim as u32,
    };

    let args = SwigluArgs {
        gate: gate_arg,
        up_inout: up_arg,
        gate_bias: gate_bias_arg,
        up_bias: up_bias_arg,
        params,
    };

    // Construct Fused SwiGLU Kernel
    let kernel = Arc::new(
        CompoundKernel::new("fused_swiglu")
            .main(SwigluStage::new(params))
            .with_manual_output(true)
            .compile(),
    );

    let threads_per_group = 256;
    let vectorized_threads = total_elements / 4;
    let num_groups = vectorized_threads.div_ceil(threads_per_group);

    let dispatch = DispatchConfig {
        grid: GridSize::d1(num_groups),
        group: ThreadgroupSize::d1(threads_per_group),
    };

    foundry.run(&kernel.bind_arc(args, dispatch)).unwrap();

    let gpu_result: Vec<f16> = up_tensor.to_vec(&foundry);

    let mut max_diff = 0.0f32;
    for i in 0..total_elements {
        let diff = (gpu_result[i].to_f32() - cpu_result[i].to_f32()).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!("Max diff: {}", max_diff);
    assert!(max_diff < 0.01, "SwiGLU Parity failed! Max diff: {}", max_diff);
}
