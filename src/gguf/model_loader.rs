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
use half::f16;
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

fn tensor_from_f32_slice<T: TensorElement>(
    tensor_name: &str,
    dims: Vec<usize>,
    data: &[f32],
    context: &Context<T>,
) -> Result<Tensor<T>, GGUFError> {
    let mut tensor = Tensor::<T>::new(dims, TensorStorage::Dedicated(context), TensorInit::Uninitialized)
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to allocate tensor '{}': {}", tensor_name, err)))?;

    {
        let slice = tensor.as_mut_slice();
        T::copy_from_f32_slice(data, slice);
    }

    tensor
        .flush_host_writes()
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to synchronize tensor '{}': {}", tensor_name, err)))?;

    Ok(tensor)
}

fn tensor_from_f16_slice<T: TensorElement>(
    tensor_name: &str,
    dims: Vec<usize>,
    data: &[f16],
    context: &Context<T>,
) -> Result<Tensor<T>, GGUFError> {
    let mut tensor = Tensor::<T>::new(dims, TensorStorage::Dedicated(context), TensorInit::Uninitialized)
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to allocate tensor '{}': {}", tensor_name, err)))?;

    {
        let slice = tensor.as_mut_slice();
        for (dst, src) in slice.iter_mut().zip(data.iter().copied()) {
            *dst = T::from_f32(src.to_f32());
        }
    }

    tensor
        .flush_host_writes()
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to synchronize tensor '{}': {}", tensor_name, err)))?;

    Ok(tensor)
}

fn tensor_from_q8_bytes<T: TensorElement>(
    tensor_name: &str,
    dims: Vec<usize>,
    expected_elements: usize,
    raw: &[u8],
    data_type: GGUFDataType,
    context: &Context<T>,
) -> Result<Tensor<T>, GGUFError> {
    let (block_size, delta_offset, weight_offset) = match data_type {
        GGUFDataType::Q8_0 => (34usize, None, 2usize),
        GGUFDataType::Q8_1 => (36usize, Some(2usize), 4usize),
        other => {
            return Err(GGUFError::InvalidTensorData(format!(
                "Unsupported Q8 tensor data type: {:?}",
                other
            )))
        }
    };

    if raw.len() % block_size != 0 {
        return Err(GGUFError::InvalidTensorData(format!(
            "Tensor '{}' data length {} is not a multiple of block size {}",
            tensor_name,
            raw.len(),
            block_size
        )));
    }

    let weights_per_block = 32usize;
    let num_blocks = raw.len() / block_size;
    let total_weights = num_blocks * weights_per_block;
    if total_weights < expected_elements {
        return Err(GGUFError::InvalidTensorData(format!(
            "Tensor '{}' expects {} elements but Q8 blocks only contain {}",
            tensor_name, expected_elements, total_weights
        )));
    }

    let mut tensor = Tensor::<T>::new(dims, TensorStorage::Dedicated(context), TensorInit::Uninitialized)
        .map_err(|err| GGUFError::InvalidTensorData(format!(
            "Failed to allocate tensor '{}': {}",
            tensor_name, err
        )))?;

    {
        let slice = tensor.as_mut_slice();
        let mut write_index = 0usize;

        for block_idx in 0..num_blocks {
            let block_start = block_idx * block_size;
            let block = &raw[block_start..block_start + block_size];

            let scale_bits = u16::from_le_bytes([block[0], block[1]]);
            let scale = f16::from_bits(scale_bits).to_f32();

            let delta = if let Some(offset) = delta_offset {
                let delta_bits = u16::from_le_bytes([block[offset], block[offset + 1]]);
                f16::from_bits(delta_bits).to_f32()
            } else {
                0.0f32
            };

            let weights = &block[weight_offset..weight_offset + weights_per_block];
            for &quantized in weights {
                if write_index >= expected_elements {
                    break;
                }

                let value = (quantized as i8) as f32 * scale + delta;
                slice[write_index] = T::from_f32(value);
                write_index += 1;
            }

            if write_index >= expected_elements {
                break;
            }
        }

        if write_index != expected_elements {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' expected {} elements but only wrote {} from Q8 data",
                tensor_name, expected_elements, write_index
            )));
        }
    }

    tensor
        .flush_host_writes()
        .map_err(|err| GGUFError::InvalidTensorData(format!(
            "Failed to synchronize tensor '{}': {}",
            tensor_name, err
        )))?;

    Ok(tensor)
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
                let tensor = tensor_from_f32_slice::<T>(&tensor_info.name, dims.clone(), values, context)?;
                Ok(GGUFTensor::from(tensor))
            }
            GGUFRawTensor::F16(values) => {
                if values.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: values.len(),
                    });
                }
                let tensor = tensor_from_f16_slice::<T>(&tensor_info.name, dims.clone(), values, context)?;
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
                    let tensor = tensor_from_f32_slice::<T>(&tensor_info.name, dims.clone(), &f32_data, context)?;
                    Ok(GGUFTensor::from(tensor))
                }
                GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                    let tensor = tensor_from_q8_bytes::<T>(
                        &tensor_info.name,
                        dims.clone(),
                        expected_elements,
                        raw,
                        data_type,
                        context,
                    )?;
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
