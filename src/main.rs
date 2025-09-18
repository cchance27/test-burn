use std::ffi::c_void;

use crate::sdpa::scaled_dot_product_attention;
use crate::sdpa_custom_metal::scaled_dot_product_attention_custom_metal;
use crate::sdpa_metal::scaled_dot_product_attention_metal;
use burn::prelude::*;
use burn::tensor::{Distribution, Float, Tensor};
mod sdpa;
mod sdpa_custom_metal;
mod sdpa_metal;

fn benchmark_burn<MyBackend: Backend>(device: &<MyBackend as Backend>::Device) {
    use std::time::Instant;
    let iterations = 5;
    let batch = 32;
    let seq = 1024;
    let dim = 64;
    let query = Tensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let key = Tensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let value = Tensor::<MyBackend, 3, Float>::random(
        [batch, seq, dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    MyBackend::sync(device);
    let start = Instant::now();
    for _ in 0..iterations {
        let _output =
            scaled_dot_product_attention(query.clone(), key.clone(), value.clone(), None, true);
    }
    MyBackend::sync(device);
    let duration = start.elapsed();
    println!("Burn time for {iterations} iterations: {duration:?}");
}

fn benchmark_metal() {
    use std::time::Instant;
    let iterations = 5;
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
    for _ in 0..iterations {
        let _output = scaled_dot_product_attention_metal(q_ptr, k_ptr, v_ptr, batch, seq, seq, dim);
    }
    let duration = start.elapsed();
    println!("Metal (MPS) time for {iterations} iterations: {duration:?}");
}

fn benchmark_custom_metal() {
    use std::time::Instant;
    let iterations = 5;
    let batch = 32;
    let seq = 1024;
    let dim = 64;
    let query: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let key: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let value: Vec<f32> = (0..(batch * seq * dim)).map(|_| rand::random()).collect();
    let start = Instant::now();
    for _ in 0..iterations {
        let _output =
            scaled_dot_product_attention_custom_metal(&query, &key, &value, batch, seq, seq, dim);
    }
    let duration = start.elapsed();
    println!("Metal (Custom Kernel) time for {iterations} iterations: {duration:?}");
}

fn main() {
    type MyBackend = burn::backend::Metal;
    let device = <MyBackend as Backend>::Device::default();
    println!("\nRunning Metal (Custom Kernel) implementation...");
    benchmark_custom_metal();
    benchmark_custom_metal();
    println!("\nRunning Metal (MPS) implementation...");
    benchmark_metal();
    benchmark_metal();
    println!("Running Burn implementation...");
    benchmark_burn::<MyBackend>(&device);
    benchmark_burn::<MyBackend>(&device);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arange_scaled_dot_product_attention_matches_pytorch() {
        type MyBackend = burn::backend::Metal;
        let pytorch_output = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0,
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0,
            44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
            58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0,
            72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0,
            86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0,
            100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
            112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0,
            124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0, 132.0, 133.0, 134.0, 135.0,
            136.0, 137.0, 138.0, 139.0, 140.0, 141.0, 142.0, 143.0, 144.0, 145.0, 146.0, 147.0,
            148.0, 149.0, 150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0,
            160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0, 170.0, 171.0,
            172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, 180.0, 181.0, 182.0, 183.0,
            184.0, 185.0, 186.0, 187.0, 188.0, 189.0, 190.0, 191.0, 192.0, 193.0, 194.0, 195.0,
            196.0, 197.0, 198.0, 199.0, 200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0,
            208.0, 209.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0, 219.0,
            220.0, 221.0, 222.0, 223.0, 224.0, 225.0, 226.0, 227.0, 228.0, 229.0, 230.0, 231.0,
            232.0, 233.0, 234.0, 235.0, 236.0, 237.0, 238.0, 239.0, 240.0, 241.0, 242.0, 243.0,
            244.0, 245.0, 246.0, 247.0, 248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0,
        ];
        let device = <MyBackend as Backend>::Device::default();
        let num_elements: i64 = 2 * 8 * 16;
        let query = Tensor::<MyBackend, 1, Int>::arange(0..num_elements, &device)
            .float()
            .reshape([2, 8, 16]);
        let key = Tensor::<MyBackend, 1, Int>::arange(0..num_elements, &device)
            .float()
            .reshape([2, 8, 16]);
        let value = Tensor::<MyBackend, 1, Int>::arange(0..num_elements, &device)
            .float()
            .reshape([2, 8, 16]);
        let output = scaled_dot_product_attention(query, key, value, None, true);
        assert_eq!(output.dims(), [2, 8, 16]);
        assert_eq!(output.to_data().as_slice::<f32>().unwrap(), &pytorch_output);
    }
}
