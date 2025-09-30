use super::{GGUFDataType, GGUFError, GGUFFile, GGUTensorInfo};
use crate::gguf::GGUFValue;
use crate::gguf::quant::{
    dequantize_q8_to_f32_into, dequantize_q8_to_f32_simd_into,
};
use half::{bf16, f16};
use std::collections::HashMap;

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

    /// Load a model from the GGUF file without materializing any tensors.
    /// Callers can later stream individual tensors directly from the mmap when
    /// instantiating a concrete model which keeps peak memory usage low.
    pub fn load_model(self) -> Result<GGUFModel, GGUFError> {
        let mut tensor_index = HashMap::new();
        for (idx, tensor_info) in self.gguf_file.tensors.iter().enumerate() {
            tensor_index.insert(tensor_info.name.clone(), idx);
        }

        Ok(GGUFModel {
            gguf_file: self.gguf_file,
            tensor_index,
        })
    }
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    gguf_file: GGUFFile,
    tensor_index: HashMap<String, usize>,
}

macro_rules! create_metadata_getter {
    ($func_name:ident, $variant:path, $return_type:ty) => {
        pub fn $func_name(&self, names: &[&str], or: $return_type) -> $return_type {
            for name in names {
                if let Some($variant(v)) = self.metadata().entries.get(*name) {
                    return *v;
                }
            }
            println!("Warning: metadata keys {:?} not found for {}, using default {:?}", names, stringify!($func_name), or);
            or
        }
    };
}

impl GGUFModel {
    create_metadata_getter!(get_metadata_u32_or, GGUFValue::U32, u32);
    create_metadata_getter!(get_metadata_f32_or, GGUFValue::F32, f32);

    pub fn metadata(&self) -> &super::GGUFMetadata {
        &self.gguf_file.metadata
    }

    pub fn tensor_infos(&self) -> impl Iterator<Item = &GGUTensorInfo> {
        self.gguf_file.tensors.iter()
    }

    pub fn tensor_info(&self, name: &str) -> Option<&GGUTensorInfo> {
        self.tensor_index
            .get(name)
            .and_then(|&idx| self.gguf_file.tensors.get(idx))
    }

    pub fn tensor_element_count(&self, tensor: &GGUTensorInfo) -> usize {
        tensor.dimensions.iter().product::<u64>() as usize
    }

    pub fn copy_tensor_to_f32(&self, tensor: &GGUTensorInfo, dst: &mut [f32]) -> Result<(), GGUFError> {
        let elem_count = self.tensor_element_count(tensor);
        if dst.len() != elem_count {
            return Err(GGUFError::DimensionMismatch { expected: elem_count, actual: dst.len() });
        }

        let data = self.gguf_file.get_tensor_data(tensor)?;
        match tensor.data_type {
            GGUFDataType::F32 => {
                let floats = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f32, elem_count)
                };
                dst.copy_from_slice(floats);
                Ok(())
            }
            GGUFDataType::F16 => {
                if data.len() != elem_count * 2 {
                    return Err(GGUFError::InvalidTensorData(format!("F16 tensor '{}' has {} bytes, expected {}", tensor.name, data.len(), elem_count * 2)));
                }
                for (chunk, out) in data.chunks_exact(2).zip(dst.iter_mut()) {
                    *out = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                }
                Ok(())
            }
            GGUFDataType::BF16 => {
                if data.len() != elem_count * 2 {
                    return Err(GGUFError::InvalidTensorData(format!("BF16 tensor '{}' has {} bytes, expected {}", tensor.name, data.len(), elem_count * 2)));
                }
                for (chunk, out) in data.chunks_exact(2).zip(dst.iter_mut()) {
                    *out = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
                }
                Ok(())
            }
            GGUFDataType::F64 => {
                if data.len() != elem_count * 8 {
                    return Err(GGUFError::InvalidTensorData(format!("F64 tensor '{}' has {} bytes, expected {}", tensor.name, data.len(), elem_count * 8)));
                }
                let doubles = unsafe {
                    std::slice::from_raw_parts(data.as_ptr() as *const f64, elem_count)
                };
                for (src, dst_elem) in doubles.iter().zip(dst.iter_mut()) {
                    *dst_elem = *src as f32;
                }
                Ok(())
            }
            GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                #[cfg(target_arch = "aarch64")]
                {
                    dequantize_q8_to_f32_simd_into(data, tensor.data_type, dst)
                        .map_err(|e| GGUFError::DequantizationError(e.to_string()))
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    dequantize_q8_to_f32_into(data, tensor.data_type, dst)
                        .map_err(|e| GGUFError::DequantizationError(e.to_string()))
                }
            }
            _ => Err(GGUFError::InvalidTensorData(format!("Unsupported tensor data type {:?} for '{}'", tensor.data_type, tensor.name))),
        }
    }

    pub fn instantiate<T: crate::metallic::models::LoadableModel>(
        &self,
        ctx: &mut crate::metallic::Context,
    ) -> Result<T, super::GGUFError> {
        crate::metallic::models::load::<T>(self, ctx).map_err(|_e| super::GGUFError::InvalidData)
    }
}
