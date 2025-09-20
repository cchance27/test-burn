const PYTORCH_ARANGE_NONCAUSAL: [f32; 256] = [
    112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0,
    125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
    122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
    119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0,
    116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0,
    113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0,
    126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
    123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
    120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 112.0, 113.0, 114.0, 115.0, 116.0,
    117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 240.0, 241.0,
    242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0,
    255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0,
    252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0,
    249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0,
    246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0,
    243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0,
    240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0,
    253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0, 248.0, 249.0,
    250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0,
    247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0,
];

const DIMENSIONS: [usize; 3] = [2, 8, 16];
const NUM_ELEMENTS: i64 = 2 * 8 * 16;

#[test]
fn arange_sdpa_burn_vs_pytorch_causal() {
    use burn::prelude::*;
    type MyBackend = burn::backend::Metal;
    let pytorch_arange_causal = (0..256).map(|x| x as f32).collect::<Vec<_>>();

    let device = <MyBackend as Backend>::Device::default();
    let query = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
        .float()
        .reshape(DIMENSIONS);
    let key = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
        .float()
        .reshape(DIMENSIONS);
    let value = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
        .float()
        .reshape(DIMENSIONS);
    let output = crate::sdpa_burn::scaled_dot_product_attention_burn(query, key, value, None, true);
    assert_eq!(output.dims(), DIMENSIONS);
    assert_eq!(
        output.to_data().as_slice::<f32>().unwrap(),
        &pytorch_arange_causal
    );
}

#[test]
fn arange_sdpa_ours_vs_pytorch_causal() {
    use crate::metallic::{Context, Tensor};
    let pytorch_arange_causal = (0..256).map(|x| x as f32).collect::<Vec<_>>();

    let mut context = Context::new().unwrap();
    let query: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
    let key: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
    let value: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();

    // Clone the device so the created tensors don't hold an immutable borrow on `context`
    let q_tensor = Tensor::create_tensor_from_slice(&query, vec![2, 8, 16], &context).unwrap();
    let k_tensor = Tensor::create_tensor_from_slice(&key, vec![2, 8, 16], &context).unwrap();
    let v_tensor = Tensor::create_tensor_from_slice(&value, vec![2, 8, 16], &context).unwrap();

    let output = context
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)
        .unwrap();
    assert_eq!(output.dims(), DIMENSIONS);
    assert_eq!(output.as_slice(), &pytorch_arange_causal);
}

#[test]
fn arange_sdpa_burn_vs_pytorch_noncausal() {
    use burn::prelude::*;
    type MyBackend = burn::backend::Metal;

    let device = <MyBackend as Backend>::Device::default();
    let query = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
        .float()
        .reshape(DIMENSIONS);
    let key = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
        .float()
        .reshape(DIMENSIONS);
    let value = Tensor::<MyBackend, 1, Int>::arange(0..NUM_ELEMENTS, &device)
        .float()
        .reshape(DIMENSIONS);
    let output =
        crate::sdpa_burn::scaled_dot_product_attention_burn(query, key, value, None, false);
    assert_eq!(output.dims(), DIMENSIONS);
    assert_eq!(
        output.to_data().as_slice::<f32>().unwrap(),
        &PYTORCH_ARANGE_NONCAUSAL
    );
}

#[test]
fn arange_sdpa_ours_vs_pytorch_noncausal() {
    use crate::metallic::{Context, Tensor};

    let mut context = Context::new().unwrap();
    let query: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
    let key: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();
    let value: Vec<f32> = (0..NUM_ELEMENTS).map(|x| x as f32).collect();

    // Clone the device so the created tensors don't hold an immutable borrow on `context`
    let q_tensor = Tensor::create_tensor_from_slice(&query, vec![2, 8, 16], &context).unwrap();
    let k_tensor = Tensor::create_tensor_from_slice(&key, vec![2, 8, 16], &context).unwrap();
    let v_tensor = Tensor::create_tensor_from_slice(&value, vec![2, 8, 16], &context).unwrap();

    let output = context
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)
        .unwrap();
    assert_eq!(output.dims(), DIMENSIONS);
    assert_eq!(output.as_slice(), &PYTORCH_ARANGE_NONCAUSAL);
}

#[test]
fn large_sdpa_ours_vs_burn_causal() {
    use burn::prelude::*;
    use burn::tensor::{Int, Tensor as BurnTensor};
    type MyBackend = burn::backend::Metal;

    let device = <MyBackend as Backend>::Device::default();
    let batch: usize = 1;
    let seq_q: usize = 64;
    let seq_k: usize = 1024;
    let dim: usize = 64;
    let q_num = batch * seq_q * dim;
    let kv_num = batch * seq_k * dim;

    let q_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(q_num as i64), &device)
        .float()
        .reshape(burn::tensor::Shape::from([batch, seq_q, dim]));
    let k_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(kv_num as i64), &device)
        .float()
        .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));
    let v_burn_input =
        BurnTensor::<MyBackend, 1, Int>::arange((kv_num as i64)..(2 * kv_num as i64), &device)
            .float()
            .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));

    let q_data_tensor = q_burn_input.to_data();
    let q_data = q_data_tensor.as_slice::<f32>().unwrap().to_vec();
    let k_data_tensor = k_burn_input.to_data();
    let k_data = k_data_tensor.as_slice::<f32>().unwrap().to_vec();
    let v_data_tensor = v_burn_input.to_data();
    let v_data = v_data_tensor.as_slice::<f32>().unwrap().to_vec();

    let burn_out = crate::sdpa_burn::scaled_dot_product_attention_burn(
        q_burn_input,
        k_burn_input,
        v_burn_input,
        None,
        true,
    );
    let burn_data = burn_out.to_data();
    let burn_slice = burn_data.as_slice::<f32>().unwrap();

    // Metallic
    use crate::metallic::{Context, Tensor};
    let mut ctx = Context::new().unwrap();
    let q_tensor =
        Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
    let k_tensor =
        Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let v_tensor =
        Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let metal_out = ctx
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, true)
        .unwrap();
    let metal_slice = metal_out.as_slice();

    // Validate with tolerance due to FP reductions
    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for (i, (metal_val, burn_val)) in metal_slice.iter().zip(burn_slice.iter()).enumerate() {
        let diff = ((*metal_val) as f64 - (*burn_val) as f64).abs();
        let rel_err = if burn_val.abs() > 1e-8 {
            diff / ((*burn_val).abs() as f64)
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, burn={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            burn_val,
            diff,
            rel_err
        );
    }
}

#[test]
fn large_sdpa_ours_vs_burn_noncausal() {
    use burn::prelude::*;
    use burn::tensor::{Int, Tensor as BurnTensor};
    type MyBackend = burn::backend::Metal;

    let device = <MyBackend as Backend>::Device::default();
    let batch: usize = 1;
    let seq_q: usize = 64;
    let seq_k: usize = 1024;
    let dim: usize = 64;
    let q_num = batch * seq_q * dim;
    let kv_num = batch * seq_k * dim;

    let q_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(q_num as i64), &device)
        .float()
        .reshape(burn::tensor::Shape::from([batch, seq_q, dim]));
    let k_burn_input = BurnTensor::<MyBackend, 1, Int>::arange(0..(kv_num as i64), &device)
        .float()
        .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));
    let v_burn_input =
        BurnTensor::<MyBackend, 1, Int>::arange((kv_num as i64)..(2 * kv_num as i64), &device)
            .float()
            .reshape(burn::tensor::Shape::from([batch, seq_k, dim]));

    let q_data_tensor = q_burn_input.to_data();
    let q_data = q_data_tensor.as_slice::<f32>().unwrap().to_vec();
    let k_data_tensor = k_burn_input.to_data();
    let k_data = k_data_tensor.as_slice::<f32>().unwrap().to_vec();
    let v_data_tensor = v_burn_input.to_data();
    let v_data = v_data_tensor.as_slice::<f32>().unwrap().to_vec();

    let burn_out = crate::sdpa_burn::scaled_dot_product_attention_burn(
        q_burn_input,
        k_burn_input,
        v_burn_input,
        None,
        false,
    );
    let burn_data = burn_out.to_data();
    let burn_slice = burn_data.as_slice::<f32>().unwrap();

    // Metallic
    use crate::metallic::{Context, Tensor};
    let mut ctx = Context::new().unwrap();
    let q_tensor =
        Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
    let k_tensor =
        Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let v_tensor =
        Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let metal_out = ctx
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, false)
        .unwrap();
    let metal_slice = metal_out.as_slice();

    // Validate with tolerance due to FP reductions
    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for (i, (metal_val, burn_val)) in metal_slice.iter().zip(burn_slice.iter()).enumerate() {
        let diff = ((*metal_val) as f64 - (*burn_val) as f64).abs();
        let rel_err = if burn_val.abs() > 1e-8 {
            diff / ((*burn_val).abs() as f64)
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, burn={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            burn_val,
            diff,
            rel_err
        );
    }
}

// Helper function for SDPA tests using random data to compare Metallic against Burn.
fn run_sdpa_test(batch: usize, seq_q: usize, seq_k: usize, dim: usize, causal: bool) {
    use burn::prelude::*;
    use burn::tensor::{Distribution, Tensor as BurnTensor};
    type MyBackend = burn::backend::Metal;

    let device = <MyBackend as Backend>::Device::default();

    // Use random data for more robust tests
    let q_burn_input = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_q, dim],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let k_burn_input = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_k, dim],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );
    let v_burn_input = BurnTensor::<MyBackend, 3>::random(
        [batch, seq_k, dim],
        Distribution::Uniform(-1.0, 1.0),
        &device,
    );

    let q_data = q_burn_input
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let k_data = k_burn_input
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    let v_data = v_burn_input
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();

    let burn_out = crate::sdpa_burn::scaled_dot_product_attention_burn(
        q_burn_input,
        k_burn_input,
        v_burn_input,
        None,
        causal,
    );
    let burn_data = burn_out.into_data();
    let burn_slice = burn_data.as_slice::<f32>().unwrap();

    // Metallic
    use crate::metallic::{Context, Tensor};
    let mut ctx = Context::new().unwrap();
    let q_tensor =
        Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
    let k_tensor =
        Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let v_tensor =
        Tensor::create_tensor_from_slice(&v_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let metal_out = ctx
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal)
        .unwrap();
    let metal_slice = metal_out.as_slice();

    assert_eq!(
        metal_out.dims(),
        &[batch, seq_q, dim],
        "Output shape mismatch"
    );

    // Validate with tolerance
    let rtol = 1e-4f64;
    let atol = 1e-6f64;
    for (i, (metal_val, burn_val)) in metal_slice.iter().zip(burn_slice.iter()).enumerate() {
        let diff = ((*metal_val) as f64 - (*burn_val) as f64).abs();
        let rel_err = if burn_val.abs() > 1e-8 {
            diff / ((*burn_val).abs() as f64)
        } else {
            diff
        };
        assert!(
            diff <= atol || rel_err <= rtol,
            "Mismatch at index {}: metal={:.6}, burn={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            metal_val,
            burn_val,
            diff,
            rel_err
        );
    }
}

#[test]
fn sdpa_non_square_non_power_of_two() {
    run_sdpa_test(1, 7, 13, 5, false);
    run_sdpa_test(1, 7, 13, 5, true);
}

#[test]
fn sdpa_very_small_dims() {
    run_sdpa_test(1, 1, 1, 1, false);
    run_sdpa_test(1, 1, 1, 1, true);
}

#[test]
fn sdpa_large_odd_sizes() {
    run_sdpa_test(1, 31, 257, 63, false);
    run_sdpa_test(1, 31, 257, 63, true);
}

#[test]
fn sdpa_causality_correctness() {
    use crate::metallic::{Context, Tensor};

    let batch = 1;
    let seq_q = 3;
    let seq_k = 3;
    let dim = 2;

    // Create simple test data
    let q_data = vec![
        1.0, 0.0, // query 0
        0.0, 1.0, // query 1
        1.0, 1.0, // query 2
    ];

    let k_data = vec![
        1.0, 0.0, // key 0
        0.0, 1.0, // key 1
        1.0, 1.0, // key 2
    ];

    // Base V data
    let v_data_base = vec![
        1.0, 0.0, // value 0
        0.0, 1.0, // value 1
        0.0, 0.0, // value 2
    ];

    // Modified V data - change value at position 2 which should be masked for queries 0 and 1
    let v_data_modified = vec![
        1.0, 0.0, // value 0
        0.0, 1.0, // value 1
        9.0, 9.0, // value 2 (should be masked for queries 0 and 1)
    ];

    let mut ctx = Context::new().unwrap();

    // Create tensors
    let q_tensor =
        Tensor::create_tensor_from_slice(&q_data, vec![batch, seq_q, dim], &ctx).unwrap();
    let k_tensor =
        Tensor::create_tensor_from_slice(&k_data, vec![batch, seq_k, dim], &ctx).unwrap();
    let v_tensor_base =
        Tensor::create_tensor_from_slice(&v_data_base, vec![batch, seq_k, dim], &ctx).unwrap();
    let v_tensor_modified =
        Tensor::create_tensor_from_slice(&v_data_modified, vec![batch, seq_k, dim], &ctx).unwrap();

    // Run SDPA with causal=True and base V
    let metal_out_base = ctx
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor_base, true)
        .unwrap();
    let metal_slice_base = metal_out_base.as_slice().to_vec();

    // Run SDPA with causal=True and modified V
    let metal_out_modified = ctx
        .scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor_modified, true)
        .unwrap();
    let metal_slice_modified = metal_out_modified.as_slice().to_vec();

    // For causal attention:
    // Query 0 should only attend to Key/Value 0 (positions 1,2 are masked)
    // Query 1 should only attend to Key/Value 0,1 (position 2 is masked)
    // Query 2 should attend to all Key/Value positions 0,1,2
    // So outputs for queries 0 and 1 should be identical between base and modified
    // Query 2 output may differ because it attends to position 2

    let rtol = 1e-4f64;
    let atol = 1e-6f64;

    // Check query 0 output (indices 0-1) - should be identical
    for i in 0..2 {
        let base_val = metal_slice_base[i] as f64;
        let modified_val = metal_slice_modified[i] as f64;
        let diff = (base_val - modified_val).abs();
        let rel_err = if base_val.abs() > 1e-8 {
            diff / base_val.abs()
        } else {
            diff
        };

        assert!(
            diff <= atol || rel_err <= rtol,
            "Causality violation for query 0 at index {}: base={:.6}, modified={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            base_val,
            modified_val,
            diff,
            rel_err
        );
    }

    // Check query 1 output (indices 2-3) - should be identical
    for i in 2..4 {
        let base_val = metal_slice_base[i] as f64;
        let modified_val = metal_slice_modified[i] as f64;
        let diff = (base_val - modified_val).abs();
        let rel_err = if base_val.abs() > 1e-8 {
            diff / base_val.abs()
        } else {
            diff
        };

        assert!(
            diff <= atol || rel_err <= rtol,
            "Causality violation for query 1 at index {}: base={:.6}, modified={:.6}, diff={:.2e}, rel={:.2e}",
            i,
            base_val,
            modified_val,
            diff,
            rel_err
        );
    }

    // Note: We don't check query 2 (indices 4-5) because it should legitimately differ
    // since it attends to the modified position 2
}
