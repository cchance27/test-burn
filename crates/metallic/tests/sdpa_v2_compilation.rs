use metallic::{
    foundry::{Foundry, storage::Pooled, tensor::Tensor}, metals::{
        rope::RopeParamsResolved, v2::attention::{stages::SdpaParamsResolved, step::FusedMhaStep}
    }, tensor::{TensorInit, dtypes::F16}, types::TensorArg
};

#[test]
fn test_fused_mha_compilation() {
    let mut foundry = Foundry::new().unwrap();

    // Create Dummy Tensors
    let q = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, 1, 128], TensorInit::Uninitialized).unwrap();
    let k = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, 1, 128], TensorInit::Uninitialized).unwrap();
    let v = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, 1, 128], TensorInit::Uninitialized).unwrap();
    let output = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, 1, 128], TensorInit::Uninitialized).unwrap();
    let cos = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, 64], TensorInit::Uninitialized).unwrap();
    let sin = Tensor::<F16, Pooled>::new(&mut foundry, vec![1, 64], TensorInit::Uninitialized).unwrap();

    // Params
    let rope_params = RopeParamsResolved {
        dim: 128,
        seq_len: 1,
        position_offset: 0,
        total_elements: 128,
    };

    let sdpa_params = SdpaParamsResolved {
        kv_len: 1,
        head_dim: 128,
        scale: 0.1,
        stride_k_s: 128,
        stride_v_s: 128,
    };

    // Compile
    let _step = FusedMhaStep::compile(
        &mut foundry,
        &TensorArg::from_tensor(&q),
        &TensorArg::from_tensor(&k),
        &TensorArg::from_tensor(&v),
        &TensorArg::from_tensor(&cos),
        &TensorArg::from_tensor(&sin),
        &TensorArg::from_tensor(&output),
        rope_params,
        sdpa_params,
        1,          // batch
        1,          // heads
        128,        // head_dim
        (128, 128), // q_strides
        (128, 128), // k_strides
        (128, 128), // v_strides
        (128, 128), // out_strides
    )
    .expect("Failed to compile FusedMhaStep");

    println!("Successfully compiled FusedMhaStep!");
}
