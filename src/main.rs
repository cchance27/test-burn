use crate::sdpa_burn::scaled_dot_product_attention_burn;
use crate::sdpa_metal::scaled_dot_product_attention_metal;
use burn::prelude::*;
use burn::tensor::{Distribution, Float, Tensor as BurnTensor};
use std::ffi::c_void;
use std::time::Instant;

mod metallic;
mod sdpa_burn;
mod sdpa_metal;

const ITERATIONS: usize = 100;

fn benchmark_burn<MyBackend: Backend>(device: &<MyBackend as Backend>::Device) {
    let batch = 32;
    let seq = 1024;
    let dim = 64;
    let query = BurnTensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let key = BurnTensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let value = BurnTensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    MyBackend::sync(device);
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _output = scaled_dot_product_attention_burn(
            query.clone(),
            key.clone(),
            value.clone(),
            None,
            true,
        );
        MyBackend::sync(device);
    }
    let duration = start.elapsed();
    println!("Burn time for {ITERATIONS} iterations: {duration:?}");
}

fn benchmark_metal() {
    let batch = 32;
    let seq = 1024;
    let dim = 64;
    let query: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let key: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let value: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let q_ptr =
        std::ptr::NonNull::new(query.as_ptr() as *mut c_void).expect("query pointer is null");
    let k_ptr = std::ptr::NonNull::new(key.as_ptr() as *mut c_void).expect("key pointer is null");
    let v_ptr =
        std::ptr::NonNull::new(value.as_ptr() as *mut c_void).expect("value pointer is null");

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _output = scaled_dot_product_attention_metal(q_ptr, k_ptr, v_ptr, batch, seq, seq, dim);
    }
    let duration = start.elapsed();
    println!("Metal (MPS) time for {ITERATIONS} iterations: {duration:?}");
}

fn benchmark_metallic(causal: bool) {
    use crate::metallic::{Context, Tensor};

    let batch = 32;
    let seq = 1024;
    let dim = 64;

    let mut context = Context::new().unwrap();
    let query: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let key: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let value: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();

    // Clone the device so the created tensors don't hold an immutable borrow on `context`
    let q_tensor = Tensor::create_tensor_from_slice(&query, vec![batch, seq, dim], &context).unwrap();
    let k_tensor = Tensor::create_tensor_from_slice(&key, vec![batch, seq, dim], &context).unwrap();
    let v_tensor = Tensor::create_tensor_from_slice(&value, vec![batch, seq, dim], &context).unwrap();

    let start: Instant = Instant::now();
    for _ in 0..ITERATIONS {
        let _output =
            context.scaled_dot_product_attention(&q_tensor, &k_tensor, &v_tensor, causal).unwrap();
    }
    let duration = start.elapsed();
    println!("Metal Opt (MPS) time for {ITERATIONS} iterations: {duration:?}");
}

fn main() {
    type MyBackend = burn::backend::Metal;
    let device = <MyBackend as Backend>::Device::default();

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

    println!("\nRunning Burn implementation...");
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
}
