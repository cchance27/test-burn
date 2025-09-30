use rustc_hash::FxHashMap;

use super::{GGUFDataType, GGUFError, GGUFFile};
use crate::gguf::file::GGUFMetadata;
use crate::gguf::tensor_info::{GGUFRawTensor, GGUTensorInfo};
use crate::metallic::tensor::TensorInit;
use crate::metallic::{Tensor, TensorElement};
use crate::{
    gguf::GGUFValue,
    metallic::{Context, Dtype, F16Element, F32Element, TensorStorage},
};
use std::cell::{Ref, RefCell};
use std::ops::Deref;

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

fn tensor_from_slice<T: TensorElement>(
    tensor_name: &str,
    dims: Vec<usize>,
    data: &[T::Scalar],
    context: &Context<T>,
) -> Result<Tensor<T>, GGUFError> {
    Tensor::<T>::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(data))
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to upload tensor '{}': {}", tensor_name, err)))
}

fn adjust_embedding_dims(name: &str, dims: &mut [usize]) {
    if name == "token_embd.weight" && dims.len() == 2 && dims[0] == 896 && dims[1] == 151936 {
        dims.swap(0, 1);
    }
}

/// A model loader that can construct a Metallic model from GGUF tensors
pub struct GGUFModelLoader {
    pub(crate) gguf_file: GGUFFile,
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
    pub fn load_model<T>(&self, context: &Context<T>) -> Result<GGUFModel, GGUFError>
    where
        T: TensorElement,
        GGUFTensor: From<Tensor<T>>,
    {
        let mut tensors: FxHashMap<String, GGUFTensor> = FxHashMap::default();

        for tensor_info in &self.gguf_file.tensor_metadata {
            match self.load_tensor(context, tensor_info) {
                Ok(tensor) => {
                    tensors.insert(tensor_info.name.clone(), tensor);
                }
                Err(err) => {
                    println!(
                        "Warning: Failed to load tensor '{}': {:?} (type={:?})",
                        tensor_info.name, err, tensor_info.data_type
                    );

                    // For fallback, we'll convert the raw data to the appropriate tensor type
                    // and put it into the right GGUFTensor variant based on the original data type
                    if let Ok(_tensor_as_slice) = self.gguf_file.get_tensor_data(tensor_info) {
                        // For now, skip this fallback to avoid the error
                        // The full implementation would involve creating a tensor of type T
                        // and then converting it to the appropriate GGUFTensor variant
                        unimplemented!("We don't support fallback tensor loading")
                    }
                }
            }
        }

        Ok(GGUFModel {
            tensors,
            metadata: self.gguf_file.metadata.clone(),
        })
    }

    fn load_tensor<T>(&self, context: &Context<T>, tensor_info: &GGUTensorInfo) -> Result<GGUFTensor, GGUFError>
    where
        T: TensorElement,
        GGUFTensor: From<Tensor<T>>,
    {
        let mut dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
        adjust_embedding_dims(&tensor_info.name, &mut dims);
        let expected_elements: usize = dims.iter().product();

        let view = tensor_info.into(&self.gguf_file)?;
        match view {
            GGUFRawTensor::F32(values) => {
                if values.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: values.len(),
                    });
                }
                // Convert F32 values to target type T
                let converted_values: Vec<T::Scalar> = values.iter().copied().map(T::from_f32).collect();
                let tensor = tensor_from_slice::<T>(&tensor_info.name, dims.clone(), &converted_values, context)?;
                Ok(GGUFTensor::from(tensor))
            }
            GGUFRawTensor::F16(values) => {
                if values.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: values.len(),
                    });
                }
                // Convert F16 values to target type T via f32
                let converted_values: Vec<T::Scalar> = values.iter().copied().map(half::f16::to_f32).map(T::from_f32).collect();
                let tensor = tensor_from_slice::<T>(&tensor_info.name, dims.clone(), &converted_values, context)?;
                Ok(GGUFTensor::from(tensor))
            }
            GGUFRawTensor::Bytes(raw, data_type) => match data_type {
                GGUFDataType::F64 => {
                    let f32_data = convert_f64_bytes(raw)?;
                    if f32_data.len() != expected_elements {
                        return Err(GGUFError::DimensionMismatch {
                            expected: expected_elements,
                            actual: f32_data.len(),
                        });
                    }
                    // Convert F32 values to target type T
                    let converted_values: Vec<T::Scalar> = f32_data.iter().copied().map(T::from_f32).collect();
                    let tensor = tensor_from_slice::<T>(&tensor_info.name, dims.clone(), &converted_values, context)?;
                    Ok(GGUFTensor::from(tensor))
                }
                GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                    #[cfg(target_arch = "aarch64")]
                    let dequant = crate::gguf::quant::dequantize_q8_to_f32_simd(raw, data_type)
                        .map_err(|err| GGUFError::DequantizationError(err.to_string()))?;
                    #[cfg(not(target_arch = "aarch64"))]
                    let dequant = crate::gguf::quant::dequantize_q8_to_f32(raw, data_type)
                        .map_err(|err| GGUFError::DequantizationError(err.to_string()))?;

                    if dequant.len() != expected_elements {
                        return Err(GGUFError::DimensionMismatch {
                            expected: expected_elements,
                            actual: dequant.len(),
                        });
                    }
                    // Convert F32 values to target type T
                    let converted_values: Vec<T::Scalar> = dequant.iter().copied().map(T::from_f32).collect();
                    let tensor = tensor_from_slice::<T>(&tensor_info.name, dims.clone(), &converted_values, context)?;
                    Ok(GGUFTensor::from(tensor))
                }
                _ => Err(GGUFError::InvalidTensorData(format!(
                    "Unsupported tensor data type: {:?}",
                    data_type
                ))),
            },
        }
    }
}

/// Heterogeneous tensor container for GGUF weights.
pub enum GGUFTensor {
    F32(Tensor<F32Element>),
    F16 {
        tensor: Tensor<F16Element>,
        cached_f32: RefCell<Option<Tensor<F32Element>>>,
    },
}

impl From<Tensor<F32Element>> for GGUFTensor {
    fn from(tensor: Tensor<F32Element>) -> Self {
        Self::F32(tensor)
    }
}

impl From<Tensor<F16Element>> for GGUFTensor {
    fn from(tensor: Tensor<F16Element>) -> Self {
        Self::F16 {
            tensor,
            cached_f32: RefCell::new(None),
        }
    }
}

/// Guard type that yields an `&Tensor<F32Element>` whether the backing storage is native f32 or lazily materialized.
pub enum F32TensorGuard<'a> {
    Borrowed(&'a Tensor<F32Element>),
    Cached(Ref<'a, Tensor<F32Element>>),
}

impl<'a> Deref for F32TensorGuard<'a> {
    type Target = Tensor<F32Element>;

    fn deref(&self) -> &Self::Target {
        match self {
            F32TensorGuard::Borrowed(tensor) => tensor,
            F32TensorGuard::Cached(tensor) => tensor,
        }
    }
}

impl GGUFTensor {
    pub fn dtype(&self) -> Dtype {
        match self {
            GGUFTensor::F32(_) => Dtype::F32,
            GGUFTensor::F16 { .. } => Dtype::F16,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            GGUFTensor::F32(tensor) => tensor.is_empty(),
            GGUFTensor::F16 { tensor, .. } => tensor.is_empty(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            GGUFTensor::F32(tensor) => tensor.len(),
            GGUFTensor::F16 { tensor, .. } => tensor.len(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            GGUFTensor::F32(tensor) => tensor.dims(),
            GGUFTensor::F16 { tensor, .. } => tensor.dims(),
        }
    }
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    pub tensors: FxHashMap<String, GGUFTensor>,
    pub metadata: GGUFMetadata,
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
    pub fn get_tensor(&self, name: &str) -> Option<&GGUFTensor> {
        self.tensors.get(name)
    }

    /// Get model metadata
    pub fn get_metadata(&self) -> &GGUFMetadata {
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
    pub fn instantiate<L: crate::metallic::models::LoadableModel<T>, T: TensorElement>(
        &self,
        ctx: &mut crate::metallic::Context<T>,
    ) -> Result<L, super::GGUFError> {
        // Delegate to the metallic::model::Model::load helper. Map MetalError -> GGUFError::InvalidData with context.
        match crate::metallic::models::load::<L, T>(self, ctx) {
            Ok(v) => Ok(v),
            Err(_e) => Err(super::GGUFError::InvalidData),
        }
    }
}
