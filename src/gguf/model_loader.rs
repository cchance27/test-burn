use super::{GGUFDataType, GGUFError, GGUFFile};
use crate::{
    gguf::{GGUFValue, GGUTensorInfo},
    metallic::{Context, Tensor, TensorInit, TensorStorage},
};
use half::f16;
use std::collections::HashMap;

fn convert_f16_bytes(raw: &[u8]) -> Result<Vec<f32>, GGUFError> {
    if !raw.len().is_multiple_of(2) {
        return Err(GGUFError::InvalidTensorData("F16 tensor byte length must be even".to_string()));
    }

    let elem_count = raw.len() / 2;
    let mut f32_data = Vec::with_capacity(elem_count);

    for chunk in raw.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        f32_data.push(f16::from_bits(bits).to_f32());
    }

    Ok(f32_data)
}

fn convert_bf16_bytes(raw: &[u8]) -> Result<Vec<f32>, GGUFError> {
    if !raw.len().is_multiple_of(2) {
        return Err(GGUFError::InvalidTensorData("BF16 tensor byte length must be even".to_string()));
    }

    let mut f32_data = Vec::with_capacity(raw.len() / 2);
    for chunk in raw.chunks_exact(2) {
        let bits16 = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        let bits32 = bits16 << 16;
        f32_data.push(f32::from_bits(bits32));
    }

    Ok(f32_data)
}

fn convert_f64_bytes(raw: &[u8]) -> Result<Vec<f32>, GGUFError> {
    if !raw.len().is_multiple_of(8) {
        return Err(GGUFError::InvalidTensorData(
            "F64 tensor byte length must be divisible by 8".to_string(),
        ));
    }

    let mut f32_data = Vec::with_capacity(raw.len() / 8);
    for chunk in raw.chunks_exact(8) {
        let bits = u64::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7]]);
        f32_data.push(f64::from_bits(bits) as f32);
    }

    Ok(f32_data)
}

fn tensor_from_slice(tensor_name: &str, dims: Vec<usize>, data: &[f32], context: &Context) -> Result<Tensor, GGUFError> {
    Tensor::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(data))
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to upload tensor '{}': {}", tensor_name, err)))
}

fn convert_f32_bytes(raw: &[u8]) -> Result<Vec<f32>, GGUFError> {
    if !raw.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(GGUFError::InvalidTensorData(
            "F32 tensor byte length must be divisible by 4".to_string(),
        ));
    }

    let mut data = Vec::with_capacity(raw.len() / std::mem::size_of::<f32>());
    for chunk in raw.chunks_exact(std::mem::size_of::<f32>()) {
        // SAFETY: `chunks_exact` guarantees chunk length matches the requested size.
        let bytes: [u8; std::mem::size_of::<f32>()] = chunk.try_into().unwrap();
        data.push(f32::from_le_bytes(bytes));
    }
    Ok(data)
}

fn tensor_from_bytes(tensor_name: &str, dims: Vec<usize>, bytes: &[u8], context: &Context) -> Result<Tensor, GGUFError> {
    let data = convert_f32_bytes(bytes)?;
    tensor_from_slice(tensor_name, dims, &data, context)
}

fn adjust_embedding_dims(name: &str, dims: &mut [usize]) {
    if name == "token_embd.weight" && dims.len() == 2 && dims[0] == 896 && dims[1] == 151936 {
        dims.swap(0, 1);
    }
}

/// A model loader that can construct a Metallic model from GGUF tensors
pub struct GGUFModelLoader {
    gguf_file: GGUFFile,
}

impl GGUFModelLoader {
    /// Create a new model loader from a GGUF file
    pub fn new(gguf_file: GGUFFile) -> Self {
        Self { gguf_file }
    }

    /// Return the size of the memory-mapped GGUF file in bytes so callers can
    /// reason about the resident host footprint of keeping the loader alive.
    pub fn mapped_len(&self) -> usize {
        self.gguf_file.mmap.len()
    }

    /// Load a model from the GGUF file
    pub fn load_model(&self, context: &Context) -> Result<GGUFModel, GGUFError> {
        let mut tensors = HashMap::new();

        for tensor_info in &self.gguf_file.tensors {
            match self.load_tensor(context, tensor_info) {
                Ok(tensor) => {
                    tensors.insert(tensor_info.name.clone(), tensor);
                }
                Err(err) => {
                    println!(
                        "Warning: Failed to load tensor '{}': {:?} (type={:?})",
                        tensor_info.name, err, tensor_info.data_type
                    );

                    if let Ok(fallback) = Tensor::try_from((&self.gguf_file, tensor_info)) {
                        tensors.insert(tensor_info.name.clone(), fallback);
                    }
                }
            }
        }

        Ok(GGUFModel {
            tensors,
            metadata: self.gguf_file.metadata.clone(),
        })
    }

    fn load_tensor(&self, context: &Context, tensor_info: &GGUTensorInfo) -> Result<Tensor, GGUFError> {
        let raw = self.gguf_file.get_tensor_data(tensor_info)?;
        let mut dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        adjust_embedding_dims(&tensor_info.name, &mut dims);
        let expected_elements: usize = dims.iter().product();

        match tensor_info.data_type {
            GGUFDataType::F32 => {
                if !raw.len().is_multiple_of(std::mem::size_of::<f32>()) {
                    return Err(GGUFError::InvalidTensorData(format!(
                        "F32 tensor '{}' byte length {} is not divisible by 4",
                        tensor_info.name,
                        raw.len()
                    )));
                }
                let actual = raw.len() / std::mem::size_of::<f32>();
                if actual != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual,
                    });
                }
                tensor_from_bytes(&tensor_info.name, dims, raw, context)
            }
            GGUFDataType::F16 => {
                let f32_data = convert_f16_bytes(raw)?;
                if f32_data.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: f32_data.len(),
                    });
                }
                tensor_from_slice(&tensor_info.name, dims, &f32_data, context)
            }
            GGUFDataType::BF16 => {
                let f32_data = convert_bf16_bytes(raw)?;
                if f32_data.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: f32_data.len(),
                    });
                }
                tensor_from_slice(&tensor_info.name, dims, &f32_data, context)
            }
            GGUFDataType::F64 => {
                let f32_data = convert_f64_bytes(raw)?;
                if f32_data.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: f32_data.len(),
                    });
                }
                tensor_from_slice(&tensor_info.name, dims, &f32_data, context)
            }
            GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                #[cfg(target_arch = "aarch64")]
                let dequant = crate::gguf::quant::dequantize_q8_to_f32_simd(raw, tensor_info.data_type)
                    .map_err(|err| GGUFError::DequantizationError(err.to_string()))?;
                #[cfg(not(target_arch = "aarch64"))]
                let dequant = crate::gguf::quant::dequantize_q8_to_f32(raw, tensor_info.data_type)
                    .map_err(|err| GGUFError::DequantizationError(err.to_string()))?;

                if dequant.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: dequant.len(),
                    });
                }
                tensor_from_slice(&tensor_info.name, dims, &dequant, context)
            }
            _ => Err(GGUFError::InvalidTensorData(format!(
                "Unsupported tensor data type: {:?}",
                tensor_info.data_type
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_f32_bytes_handles_misaligned_slice() {
        let values = [1.0f32, 2.5, -3.75];
        let mut storage = Vec::with_capacity(values.len() * std::mem::size_of::<f32>() + 2);
        storage.push(0); // introduce a one-byte offset for misalignment
        for value in values {
            storage.extend_from_slice(&value.to_le_bytes());
        }
        storage.push(0); // padding to avoid reading past the buffer in debug modes

        let start = 1;
        let end = start + values.len() * std::mem::size_of::<f32>();
        let misaligned = &storage[start..end];

        let converted = convert_f32_bytes(misaligned).expect("misaligned slice should convert");
        assert_eq!(converted, values);
    }
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    pub tensors: HashMap<String, Tensor>,
    pub metadata: super::GGUFMetadata,
}

macro_rules! create_metadata_getter {
    ($func_name:ident, $variant:path, $return_type:ty) => {
        pub fn $func_name(&self, names: &[&str], or: $return_type) -> $return_type {
            for name in names {
                if let Some($variant(v)) = self.metadata.entries.get(*name) {
                    return *v; // Return the found value
                }
            }
            or // Return the default value if nothing was found
        }
    };
}

impl GGUFModel {
    create_metadata_getter!(get_metadata_u32_or, GGUFValue::U32, u32);
    create_metadata_getter!(get_metadata_f32_or, GGUFValue::F32, f32);

    /// Get a tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    /// Get model metadata
    pub fn get_metadata(&self) -> &super::GGUFMetadata {
        &self.metadata
    }

    /// Get model architecture from metadata
    pub fn get_architecture(&self) -> Option<&str> {
        if let Some(super::GGUFValue::String(arch)) = self.metadata.entries.get("general.architecture") {
            Some(arch)
        } else {
            None
        }
    }

    /// Get context length from metadata
    pub fn get_context_length(&self) -> Option<u64> {
        if let Some(super::GGUFValue::U32(len)) = self.metadata.entries.get("qwen2.context_length") {
            Some(*len as u64)
        } else {
            None
        }
    }

    /// Instantiate a concrete Metallic model that implements `LoadableModel`.
    /// This allows callers to do:
    ///   let gguf_model = GGUFModelLoader::new(...).load_model(...)?
    ///   let qwen: Qwen25 = gguf_model.instantiate(&mut ctx)?;
    pub fn instantiate<T: crate::metallic::models::LoadableModel>(
        &self,
        ctx: &mut crate::metallic::Context,
    ) -> Result<T, super::GGUFError> {
        // Delegate to the metallic::model::Model::load helper. Map MetalError -> GGUFError::InvalidData with context.
        match crate::metallic::models::load::<T>(self, ctx) {
            Ok(v) => Ok(v),
            Err(_e) => Err(super::GGUFError::InvalidData),
        }
    }
}
