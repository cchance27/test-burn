use super::{GGUFDataType, GGUFError, GGUFFile, GGUFRawTensor};
use crate::metallic::tensor::TensorInit;
use crate::{
    gguf::{GGUFValue, GGUTensorInfo},
    metallic::{
        BF16Element, Context, Dtype, F16Element, F32Element, GenericTensor, Tensor as LegacyTensor, TensorBF16, TensorElement, TensorF16,
        TensorF32, TensorStorage,
    },
};
use std::cell::{Ref, RefCell};
use std::collections::HashMap;
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
    context: &Context,
) -> Result<GenericTensor<T>, GGUFError> {
    GenericTensor::<T>::new(dims, TensorStorage::Dedicated(context), TensorInit::CopyFrom(data))
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to upload tensor '{}': {}", tensor_name, err)))
}

fn materialize_f32_from_tensor<T: TensorElement>(
    tensor_name: &str,
    tensor: &GenericTensor<T>,
    context: &Context,
) -> Result<TensorF32, GGUFError> {
    let host = tensor.to_f32_vec();
    TensorF32::from_f32_slice(tensor.dims.clone(), TensorStorage::Dedicated(context), &host)
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to materialize f32 tensor '{}': {}", tensor_name, err)))
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
        let mut tensors: HashMap<String, GGUFTensor> = HashMap::new();

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

                    if let Ok(fallback) = LegacyTensor::try_from((&self.gguf_file, tensor_info)) {
                        tensors.insert(tensor_info.name.clone(), GGUFTensor::from(fallback));
                    }
                }
            }
        }

        Ok(GGUFModel {
            tensors,
            metadata: self.gguf_file.metadata.clone(),
        })
    }

    fn load_tensor(&self, context: &Context, tensor_info: &GGUTensorInfo) -> Result<GGUFTensor, GGUFError> {
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
                let tensor = tensor_from_slice::<F32Element>(&tensor_info.name, dims.clone(), values, context)?;
                Ok(GGUFTensor::from(tensor))
            }
            GGUFRawTensor::F16(values) => {
                if values.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: values.len(),
                    });
                }
                let tensor = tensor_from_slice::<F16Element>(&tensor_info.name, dims.clone(), values, context)?;
                Ok(GGUFTensor::from(tensor))
            }
            GGUFRawTensor::BF16(values) => {
                if values.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: values.len(),
                    });
                }
                let tensor = tensor_from_slice::<BF16Element>(&tensor_info.name, dims.clone(), values, context)?;
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
                    let tensor = tensor_from_slice::<F32Element>(&tensor_info.name, dims.clone(), &f32_data, context)?;
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
                    let tensor = tensor_from_slice::<F32Element>(&tensor_info.name, dims.clone(), &dequant, context)?;
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
    F32(TensorF32),
    F16 {
        tensor: TensorF16,
        cached_f32: RefCell<Option<TensorF32>>,
    },
    BF16 {
        tensor: TensorBF16,
        cached_f32: RefCell<Option<TensorF32>>,
    },
}

impl From<TensorF32> for GGUFTensor {
    fn from(tensor: TensorF32) -> Self {
        Self::F32(tensor)
    }
}

impl From<TensorF16> for GGUFTensor {
    fn from(tensor: TensorF16) -> Self {
        Self::F16 {
            tensor,
            cached_f32: RefCell::new(None),
        }
    }
}

impl From<TensorBF16> for GGUFTensor {
    fn from(tensor: TensorBF16) -> Self {
        Self::BF16 {
            tensor,
            cached_f32: RefCell::new(None),
        }
    }
}

/// Guard type that yields an `&TensorF32` whether the backing storage is native f32 or lazily materialized.
pub enum F32TensorGuard<'a> {
    Borrowed(&'a TensorF32),
    Cached(Ref<'a, TensorF32>),
}

impl<'a> Deref for F32TensorGuard<'a> {
    type Target = TensorF32;

    fn deref(&self) -> &Self::Target {
        match self {
            F32TensorGuard::Borrowed(tensor) => tensor,
            F32TensorGuard::Cached(tensor) => &*tensor,
        }
    }
}

impl GGUFTensor {
    pub fn dtype(&self) -> Dtype {
        match self {
            GGUFTensor::F32(_) => Dtype::F32,
            GGUFTensor::F16 { .. } => Dtype::F16,
            GGUFTensor::BF16 { .. } => Dtype::BF16,
        }
    }

    pub fn ensure_f32<'a>(&'a self, name: &str, context: &Context) -> Result<F32TensorGuard<'a>, GGUFError> {
        match self {
            GGUFTensor::F32(tensor) => Ok(F32TensorGuard::Borrowed(tensor)),
            GGUFTensor::F16 { tensor, cached_f32 } => {
                if cached_f32.borrow().is_none() {
                    let converted = materialize_f32_from_tensor(name, tensor, context)?;
                    *cached_f32.borrow_mut() = Some(converted);
                }
                let guard = Ref::map(cached_f32.borrow(), |opt| opt.as_ref().expect("f32 cache populated"));
                Ok(F32TensorGuard::Cached(guard))
            }
            GGUFTensor::BF16 { tensor, cached_f32 } => {
                if cached_f32.borrow().is_none() {
                    let converted = materialize_f32_from_tensor(name, tensor, context)?;
                    *cached_f32.borrow_mut() = Some(converted);
                }
                let guard = Ref::map(cached_f32.borrow(), |opt| opt.as_ref().expect("f32 cache populated"));
                Ok(F32TensorGuard::Cached(guard))
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            GGUFTensor::F32(tensor) => tensor.len(),
            GGUFTensor::F16 { tensor, .. } => tensor.len(),
            GGUFTensor::BF16 { tensor, .. } => tensor.len(),
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            GGUFTensor::F32(tensor) => tensor.dims(),
            GGUFTensor::F16 { tensor, .. } => tensor.dims(),
            GGUFTensor::BF16 { tensor, .. } => tensor.dims(),
        }
    }
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    pub tensors: HashMap<String, GGUFTensor>,
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
    pub fn get_tensor(&self, name: &str) -> Option<&GGUFTensor> {
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
