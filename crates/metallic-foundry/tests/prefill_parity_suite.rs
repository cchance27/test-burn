use half::f16;
use metallic_foundry::{
    Foundry, compound::stages::Quantization, metals::{
        gemm::step::{GemmParams, GemmV2Args, gemm_dispatch_config, get_gemm_kernel}, mma::stages::TileConfig
    }, storage::Pooled, tensor::{F16, Tensor as FoundryTensor, TensorInit}, types::TensorArg
};
use rand::{Rng, rng};
use serial_test::serial;

fn run_parity_check(m: usize, n: usize, k: usize, tile_config: TileConfig) {
    let mut foundry = Foundry::new().unwrap();
    let mut rng = rng();

    // Init data
    let a_data: Vec<f16> = (0..m * k).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();
    let b_data: Vec<f16> = (0..k * n).map(|_| f16::from_f32(rng.random_range(-0.5..0.5))).collect();

    // CPU Reference
    let mut cpu_out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[i * k + l].to_f32() * b_data[l * n + j].to_f32();
            }
            cpu_out[i * n + j] = sum;
        }
    }

    // GPU Run
    let a = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, k], TensorInit::CopyFrom(&a_data)).unwrap();
    let b = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![k, n], TensorInit::CopyFrom(&b_data)).unwrap();
    let output = FoundryTensor::<F16, Pooled>::new(&mut foundry, vec![m, n], TensorInit::Uninitialized).unwrap();

    let params = GemmParams::simple(m as i32, n as i32, k as i32, false, false, tile_config);
    let dispatch = gemm_dispatch_config(&params, tile_config);
    let kernel = get_gemm_kernel(Quantization::F16, Quantization::F16, false, false, tile_config, false, false);

    let args = GemmV2Args {
        a: TensorArg::from_tensor(&a),
        b: TensorArg::from_tensor(&b),
        d: TensorArg::from_tensor(&output),
        c: TensorArg::from_tensor(&output),
        bias: TensorArg::from_tensor(&output),
        b_scales: TensorArg::from_tensor(&output),
        weights_per_block: 32,
        alpha: 1.0,
        beta: 0.0,
        params,
    };

    foundry.run(&kernel.bind(args, dispatch)).unwrap();

    let gpu_out = FoundryTensor::to_vec(&output, &foundry);

    // Compare
    let mut max_err = 0.0f32;
    for i in 0..m * n {
        let diff = (gpu_out[i].to_f32() - cpu_out[i]).abs();
        if diff > max_err {
            max_err = diff;
        }
    }

    println!("Test M={} N={} K={} Config={:?} | Max Err: {:.5}", m, n, k, tile_config, max_err);
    assert!(max_err < 0.1, "Max error too high: {}", max_err);
}

#[test]
#[serial]
fn test_prefill_m1() {
    run_parity_check(1, 128, 128, TileConfig::Default);
}

#[test]
#[serial]
fn test_prefill_m4_default() {
    run_parity_check(4, 128, 128, TileConfig::Default);
}

#[test]
#[serial]
fn test_prefill_m4_skinnym() {
    run_parity_check(4, 128, 128, TileConfig::SkinnyM);
}

#[test]
#[serial]
fn test_prefill_m7_unaligned() {
    run_parity_check(7, 129, 130, TileConfig::Default);
}

#[test]
#[serial]
fn test_prefill_m32_default() {
    run_parity_check(32, 128, 128, TileConfig::Default);
}
