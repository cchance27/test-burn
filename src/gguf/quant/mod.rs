#![allow(dead_code)]

//! Tensor conversion utilities for GGUF format and Quantization
use crate::gguf::GGUFDataType;
use crate::gguf::{GGUFFile, GGUTensorInfo};
use crate::metallic::Tensor;
use crate::metallic::{TensorInit, TensorStorage};
use half::f16;
use std::convert::TryFrom;

// Quantization modules for GGUF format
pub mod q8;
pub mod q8_simd;

#[cfg(not(target_arch = "aarch64"))]
pub use q8::dequantize_q8_to_f32;

#[cfg(target_arch = "aarch64")]
pub use q8_simd::dequantize_q8_to_f32_simd;

/// Uniform wrapper that selects the appropriate dequantization implementation
/// depending on the target architecture. Call this from other code when you
/// want a stable, architecture-independent API.
pub fn dequantize_q8(data: &[u8], data_type: GGUFDataType) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    #[cfg(target_arch = "aarch64")]
    {
        // Use SIMD-ready implementation on AArch64
        q8_simd::dequantize_q8_to_f32_simd(data, data_type)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // Use portable implementation on other targets
        q8::dequantize_q8_to_f32(data, data_type)
    }
}

// Benchmark modules
#[cfg(test)]
mod tests;

// Converting between GGUFTensors and Metallic Tensors
impl TryFrom<(&GGUFFile, &GGUTensorInfo)> for Tensor {
    type Error = Box<dyn std::error::Error>;

    fn try_from(value: (&GGUFFile, &GGUTensorInfo)) -> Result<Self, Self::Error> {
        let (gguf_file, tensor_info) = value;

        // Validate tensor data before processing
        gguf_file.validate_tensor_data(tensor_info)?;

        // Get the tensor data from the GGUF file
        let data = gguf_file.get_tensor_data(tensor_info)?;

        // Convert dimensions from u64 to usize
        let dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();

        // Initialize Metallic context (this should ideally be passed in)
        let context = crate::metallic::Context::new().map_err(|e| format!("Failed to create Metallic context: {:?}", e))?;

        match tensor_info.data_type {
            crate::gguf::GGUFDataType::F32 => {
                // For F32 tensors, we can directly copy the data
                let float_data: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / std::mem::size_of::<f32>()) };

                Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(float_data)).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            crate::gguf::GGUFDataType::F16 => {
                // F16 (IEEE 754 half) stored as little-endian u16 per element
                let elem_count = data.len() / 2;
                let mut f32_data = Vec::with_capacity(elem_count);
                for i in 0..std::cmp::min(10, elem_count) {
                    let lo = data[2 * i] as u16;
                    let hi = data[2 * i + 1] as u16;
                    let bits = lo | (hi << 8);
                    let h = f16::from_bits(bits);
                    let f32_val = h.to_f32();
                    f32_data.push(f32_val);
                }
                // Add the rest of the data
                for i in 10..elem_count {
                    let lo = data[2 * i] as u16;
                    let hi = data[2 * i + 1] as u16;
                    let bits = lo | (hi << 8);
                    let h = f16::from_bits(bits);
                    f32_data.push(h.to_f32());
                }
                Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&f32_data)).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            crate::gguf::GGUFDataType::BF16 => {
                // BF16 (bfloat16) stored as little-endian u16 per element.
                // To convert to f32, left-shift 16 bits into a u32 and reinterpret.
                let elem_count = data.len() / 2;
                let mut f32_data = Vec::with_capacity(elem_count);
                for i in 0..elem_count {
                    let lo = data[2 * i] as u32;
                    let hi = data[2 * i + 1] as u32;
                    let bits16 = lo | (hi << 8);
                    let bits32 = bits16 << 16;
                    f32_data.push(f32::from_bits(bits32));
                }
                Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&f32_data)).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            crate::gguf::GGUFDataType::F64 => {
                // For F64 tensors, convert to F32
                let f64_data: &[f64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len() / std::mem::size_of::<f64>()) };

                // Convert F64 to F32
                let f32_data: Vec<f32> = f64_data.iter().map(|&x| x as f32).collect();

                Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&f32_data)).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            crate::gguf::GGUFDataType::Q8_0 | crate::gguf::GGUFDataType::Q8_1 => {
                // For Q8_0/Q8_1 tensors, we need to dequantize to F32
                #[cfg(target_arch = "aarch64")]
                let f32_data = {
                    // For now, always use the regular SIMD version as it's faster
                    // The parallel version has overhead that's not beneficial in our tests
                    dequantize_q8_to_f32_simd(data, tensor_info.data_type)?
                };

                #[cfg(not(target_arch = "aarch64"))]
                let f32_data = dequantize_q8_to_f32(data, tensor_info.data_type)?;

                Tensor::new(dims, TensorStorage::Dedicated(&context), TensorInit::CopyFrom(&f32_data)).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
            }
            _ => {
                // For other data types, we would need to implement specific dequantization
                Err(format!("Unsupported tensor data type: {:?}", tensor_info.data_type).into())
            }
        }
    }
}