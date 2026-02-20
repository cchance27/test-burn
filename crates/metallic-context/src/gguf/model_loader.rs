use std::sync::Arc;

use half::f16;
use objc2_metal::{MTLBuffer, MTLDevice};
use rustc_hash::FxHashMap;

use super::{GGUFDataType, GGUFError, GGUFFile};
use crate::{
    Context, Tensor, TensorElement, TensorStorage, gguf::{
        GGUFValue, file::GGUFMetadata, tensor_info::{GGUFRawTensor, GGUTensorInfo}
    }, tensor::{Q8_0_BLOCK_SIZE_BYTES, Q8_0_WEIGHTS_PER_BLOCK, QuantizedQ8_0Tensor, TensorInit, quantized::swizzle_q8_0_blocks_nk}
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GGUFLayoutHint {
    Nk,
    Kn,
}

impl GGUFLayoutHint {
    fn from_architecture(arch: Option<&str>) -> Self {
        match arch.map(|s| s.to_ascii_lowercase()) {
            Some(ref a) if a.contains("falcon") || a.contains("refact") => GGUFLayoutHint::Kn,
            _ => GGUFLayoutHint::Nk,
        }
    }
}

const METALLIC_GGUF_LAYOUT_OVERRIDE_ENV: &str = "METALLIC_GGUF_LAYOUT_OVERRIDE";
const METALLIC_GGUF_LAYOUT_DEBUG_ENV: &str = "METALLIC_GGUF_LAYOUT_DEBUG";

fn parse_layout_hint(value: &str) -> Option<GGUFLayoutHint> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized.contains("kn") {
        Some(GGUFLayoutHint::Kn)
    } else if normalized.contains("nk") {
        Some(GGUFLayoutHint::Nk)
    } else {
        None
    }
}

fn layout_hint_override() -> Option<GGUFLayoutHint> {
    let value = std::env::var(METALLIC_GGUF_LAYOUT_OVERRIDE_ENV).ok()?;
    parse_layout_hint(&value)
}

fn layout_hint_from_metadata_with_key(metadata: &GGUFMetadata, arch: Option<&str>) -> Option<(GGUFLayoutHint, String, String)> {
    let mut keys = Vec::new();
    keys.push("tensor_data_layout".to_string());
    keys.push("general.tensor_data_layout".to_string());
    if let Some(arch) = arch {
        let arch_lower = arch.to_ascii_lowercase();
        keys.push(format!("{arch_lower}.tensor_data_layout"));
    }

    for key in keys {
        if let Some(GGUFValue::String(value)) = metadata.entries.get(&key)
            && let Some(hint) = parse_layout_hint(value)
        {
            return Some((hint, key, value.clone()));
        }
    }

    for (key, value) in &metadata.entries {
        if key.ends_with(".tensor_data_layout")
            && let GGUFValue::String(layout) = value
            && let Some(hint) = parse_layout_hint(layout)
        {
            return Some((hint, key.clone(), layout.clone()));
        }
    }

    None
}

fn layout_debug_enabled() -> bool {
    std::env::var(METALLIC_GGUF_LAYOUT_DEBUG_ENV)
        .ok()
        .map(|v| v.trim() != "0")
        .unwrap_or(false)
}

fn log_layout_hint(arch: Option<&str>, hint: GGUFLayoutHint, source: Option<(String, String)>) {
    if !layout_debug_enabled() {
        return;
    }
    match source {
        Some((key, value)) => {
            eprintln!(
                "[GGUF_LAYOUT] arch={:?} hint={:?} source_key={} source_value={}",
                arch, hint, key, value
            );
        }
        None => {
            eprintln!("[GGUF_LAYOUT] arch={:?} hint={:?} source_key=<none>", arch, hint);
        }
    }
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
            )));
        }
    };

    if !raw.len().is_multiple_of(block_size) {
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
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to allocate tensor '{}': {}", tensor_name, err)))?;

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
        .map_err(|err| GGUFError::InvalidTensorData(format!("Failed to synchronize tensor '{}': {}", tensor_name, err)))?;

    Ok(tensor)
}

fn metadata_usize(metadata: &GGUFMetadata, keys: &[&str]) -> Option<usize> {
    for key in keys {
        if let Some(value) = metadata.entries.get(*key) {
            match value {
                GGUFValue::U32(v) => return Some(*v as usize),
                GGUFValue::U64(v) => return Some(*v as usize),
                GGUFValue::I32(v) if *v >= 0 => return Some(*v as usize),
                GGUFValue::I64(v) if *v >= 0 => return Some(*v as usize),
                _ => {}
            }
        }
    }
    None
}

fn metadata_architecture(metadata: &GGUFMetadata) -> Option<&str> {
    match metadata.entries.get("general.architecture") {
        Some(GGUFValue::String(value)) if !value.trim().is_empty() => Some(value.as_str()),
        _ => None,
    }
}

fn metadata_arch_usize(metadata: &GGUFMetadata, arch: Option<&str>, suffixes: &[&str], fallback_keys: &[&str]) -> Option<usize> {
    if let Some(arch) = arch {
        for suffix in suffixes {
            let key = format!("{arch}.{suffix}");
            if let Some(v) = metadata_usize(metadata, &[&key]) {
                return Some(v);
            }
        }
    }

    metadata_usize(metadata, fallback_keys)
}

fn metadata_vocab_size(metadata: &GGUFMetadata, arch: Option<&str>) -> Option<usize> {
    if let Some(v) = metadata_arch_usize(metadata, arch, &["vocab_size"], &["vocab_size", "model.vocab_size"]) {
        return Some(v);
    }

    if let Some(GGUFValue::Array(values)) = metadata.entries.get("tokenizer.ggml.tokens") {
        return Some(values.len());
    }

    None
}

fn adjust_embedding_dims(_name: &str, dims: &mut [usize], metadata: &GGUFMetadata, arch: Option<&str>) {
    if dims.len() != 2 {
        return;
    }

    let d_model = metadata_arch_usize(metadata, arch, &["embedding_length", "d_model"], &["model.d_model", "d_model"]);
    let vocab = metadata_vocab_size(metadata, arch);

    if let (Some(d_model), Some(vocab)) = (d_model, vocab)
        && dims[0] == d_model
        && dims[1] == vocab
    {
        dims.swap(0, 1);
    }
}

/// A model loader that can construct a Metallic model from GGUF tensors
pub struct GGUFModelLoader {
    pub(crate) gguf_file: Arc<GGUFFile>,
}

impl GGUFModelLoader {
    /// Create a new model loader from a GGUF file
    pub fn new(gguf_file: GGUFFile) -> Self {
        Self {
            gguf_file: Arc::new(gguf_file),
        }
    }

    /// Return the size of the memory-mapped GGUF file in bytes so callers can
    /// reason about the resident host footprint of keeping the loader alive.
    pub fn mapped_len(&self) -> usize {
        self.gguf_file.mmap.len()
    }

    /// Load a model from the GGUF file
    pub fn load_model(&self) -> Result<GGUFModel, GGUFError> {
        let mut tensors: FxHashMap<String, GGUFTensor> = FxHashMap::default();

        let arch = metadata_architecture(&self.gguf_file.metadata);
        for tensor_info in &self.gguf_file.tensor_metadata {
            let mut dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
            adjust_embedding_dims(&tensor_info.name, &mut dims, &self.gguf_file.metadata, arch);
            let expected_elements: usize = dims.iter().product();

            tensors.insert(
                tensor_info.name.clone(),
                GGUFTensor {
                    info: tensor_info.clone(),
                    dims,
                    expected_elements,
                },
            );
        }

        let meta_hint = layout_hint_from_metadata_with_key(&self.gguf_file.metadata, arch);
        let layout_hint = layout_hint_override()
            .or_else(|| meta_hint.as_ref().map(|(hint, _, _)| *hint))
            .unwrap_or_else(|| GGUFLayoutHint::from_architecture(arch));
        let debug_source = if layout_hint_override().is_some() {
            Some((METALLIC_GGUF_LAYOUT_OVERRIDE_ENV.to_string(), "<env override>".to_string()))
        } else {
            meta_hint.as_ref().map(|(_, key, value)| (key.clone(), value.clone()))
        };
        log_layout_hint(arch, layout_hint, debug_source);

        Ok(GGUFModel {
            gguf_file: Arc::clone(&self.gguf_file),
            tensors,
            metadata: self.gguf_file.metadata.clone(),
            layout_hint,
        })
    }
}

impl GGUFModel {
    /// Helper to map GGUF data types to Metallic Dtype.
    pub fn map_dtype(gguf_type: GGUFDataType) -> crate::tensor::Dtype {
        match gguf_type {
            GGUFDataType::F32 => crate::tensor::Dtype::F32,
            GGUFDataType::F16 => crate::tensor::Dtype::F16,
            GGUFDataType::Q8_0 => crate::tensor::Dtype::U8,
            GGUFDataType::Q8_1 => crate::tensor::Dtype::U8,
            _ => crate::tensor::Dtype::F16, // Default fallback
        }
    }
}

/// Metadata describing a tensor stored in the GGUF file. Data is materialized on-demand.
pub struct GGUFTensor {
    info: GGUTensorInfo,
    dims: Vec<usize>,
    expected_elements: usize,
}

impl GGUFTensor {
    pub fn data_type(&self) -> GGUFDataType {
        self.info.data_type
    }

    pub fn is_empty(&self) -> bool {
        self.expected_elements == 0
    }

    pub fn len(&self) -> usize {
        self.expected_elements
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Materialize the tensor directly to a Metal buffer for use in Foundry.
    pub fn materialize_to_device(
        &self,
        file: &GGUFFile,
        device: &crate::types::MetalDevice,
    ) -> Result<crate::types::MetalBuffer, GGUFError> {
        let view = self.info.view(file)?;
        let raw_bytes = match view {
            GGUFRawTensor::F32(slice) => bytemuck::cast_slice(slice),
            GGUFRawTensor::F16(slice) => bytemuck::cast_slice(slice),
            GGUFRawTensor::Bytes(bytes, _) => bytes,
        };

        let buffer = device
            .0
            .newBufferWithLength_options(raw_bytes.len(), objc2_metal::MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Failed to allocate Metal buffer for tensor '{}'", self.info.name)))?;

        unsafe {
            let ptr = buffer.contents().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(raw_bytes.as_ptr(), ptr, raw_bytes.len());
        }

        Ok(crate::types::MetalBuffer(buffer))
    }

    pub fn raw_view<'a>(&self, file: &'a GGUFFile) -> Result<GGUFRawTensor<'a>, GGUFError> {
        self.info.view(file)
    }

    pub fn materialize<T: TensorElement>(&self, file: &GGUFFile, context: &Context<T>) -> Result<Tensor<T>, GGUFError> {
        let view = self.info.view(file)?;
        match view {
            GGUFRawTensor::F32(values) => {
                if values.len() != self.expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: self.expected_elements,
                        actual: values.len(),
                    });
                }
                tensor_from_f32_slice::<T>(&self.info.name, self.dims.clone(), values, context)
            }
            GGUFRawTensor::F16(values) => {
                if values.len() != self.expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: self.expected_elements,
                        actual: values.len(),
                    });
                }
                tensor_from_f16_slice::<T>(&self.info.name, self.dims.clone(), values, context)
            }
            GGUFRawTensor::Bytes(raw, data_type) => match data_type {
                GGUFDataType::F64 => {
                    let f32_data = convert_f64_bytes(raw)?;
                    if f32_data.len() != self.expected_elements {
                        return Err(GGUFError::DimensionMismatch {
                            expected: self.expected_elements,
                            actual: f32_data.len(),
                        });
                    }
                    tensor_from_f32_slice::<T>(&self.info.name, self.dims.clone(), &f32_data, context)
                }
                GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                    tensor_from_q8_bytes::<T>(&self.info.name, self.dims.clone(), self.expected_elements, raw, data_type, context)
                }
                _ => Err(GGUFError::InvalidTensorData(format!(
                    "Unsupported tensor data type: {:?}",
                    data_type
                ))),
            },
        }
    }

    /// Materialize a Q8_0 tensor as packed bytes on GPU without upcasting.
    /// This returns a `QuantizedQ8_0Tensor` (raw = `Tensor<U8>`) and preserves the logical dims.
    pub fn materialize_q8_0_packed<TCtx: TensorElement>(
        &self,
        file: &GGUFFile,
        ctx: &Context<TCtx>,
        layout_hint: GGUFLayoutHint,
    ) -> Result<QuantizedQ8_0Tensor, GGUFError> {
        // Validate the raw view and type
        let view = self.info.view(file)?;
        let raw = match view {
            GGUFRawTensor::Bytes(bytes, super::GGUFDataType::Q8_0) => bytes,
            _ => {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor '{}' is not Q8_0 bytes",
                    self.info.name
                )));
            }
        };

        if raw.len() % Q8_0_BLOCK_SIZE_BYTES != 0 {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' Q8_0 data length {} is not a multiple of {}",
                self.info.name,
                raw.len(),
                Q8_0_BLOCK_SIZE_BYTES
            )));
        }

        let blocks = raw.len() / Q8_0_BLOCK_SIZE_BYTES;
        let total_weights = blocks * Q8_0_WEIGHTS_PER_BLOCK;
        if total_weights < self.expected_elements {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' expects {} elements but Q8_0 blocks contain {}",
                self.info.name, self.expected_elements, total_weights
            )));
        }

        if std::env::var("Q8_SWIZZLE_DEBUG").is_ok() {
            eprintln!(
                "[Q8_SWIZZLE_DEBUG] tensor={} dims={:?} layout_hint={:?}",
                self.info.name, self.dims, layout_hint
            );
        }

        let maybe_swizzled = if matches!(layout_hint, GGUFLayoutHint::Nk) && self.dims.len() == 2 {
            // Our matmul conventions store dense weights as [K, N] (rows = K, cols = N).
            // NK row-major in GGUF means rows correspond to output N and columns to input K.
            // To swizzle NK into k-block-major, use rows_n = dims[1] (N) and cols_k = dims[0] (K).
            let cols_k = self.dims[0]; // K
            let rows_n = self.dims[1]; // N
            swizzle_q8_0_blocks_nk(rows_n, cols_k, raw)
        } else {
            None
        };

        if layout_hint != GGUFLayoutHint::Nk {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor '{}' uses unsupported layout {:?} for Q8_0 weights",
                self.info.name, layout_hint
            )));
        }

        let source = maybe_swizzled.as_deref().unwrap_or(raw);
        let mut data_bytes = Vec::with_capacity(blocks * Q8_0_WEIGHTS_PER_BLOCK);
        let mut scale_bytes = Vec::with_capacity(blocks * 2);
        for chunk in source.chunks_exact(Q8_0_BLOCK_SIZE_BYTES) {
            scale_bytes.extend_from_slice(&chunk[0..2]);
            data_bytes.extend_from_slice(&chunk[2..(2 + Q8_0_WEIGHTS_PER_BLOCK)]);
        }
        QuantizedQ8_0Tensor::from_split_bytes_in_context(self.dims.clone(), &data_bytes, &scale_bytes, ctx)
            .map_err(|e| GGUFError::InvalidTensorData(e.to_string()))
    }
}

/// A model loaded from a GGUF file
pub struct GGUFModel {
    pub gguf_file: Arc<GGUFFile>,
    pub tensors: FxHashMap<String, GGUFTensor>,
    pub metadata: GGUFMetadata,
    pub layout_hint: GGUFLayoutHint,
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

    pub fn tensor_raw_view(&self, name: &str) -> Result<GGUFRawTensor<'_>, GGUFError> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Tensor '{}' not found in GGUF metadata", name)))?;
        tensor.raw_view(&self.gguf_file)
    }

    /// Materialize a tensor into the provided Metal context.
    pub fn materialize_tensor<T: TensorElement>(&self, name: &str, context: &Context<T>) -> Result<Tensor<T>, GGUFError> {
        let descriptor = self
            .tensors
            .get(name)
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Tensor '{}' not found in GGUF metadata", name)))?;

        descriptor.materialize(self.gguf_file.as_ref(), context)
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

    pub fn layout_hint(&self) -> GGUFLayoutHint {
        self.layout_hint
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
    ///   let gguf_model = GGUFModelLoader::new(...).load_model()?;
    ///   let qwen: Qwen25 = gguf_model.instantiate(&mut ctx)?;
    pub fn instantiate<L: crate::models::LoadableModel<T>, T: TensorElement>(
        &self,
        ctx: &mut crate::Context<T>,
    ) -> Result<L, super::GGUFError> {
        // Delegate to the metallic::model::Model::load helper. Map MetalError -> GGUFError::InvalidTensorData with context.
        match crate::models::load::<L, T>(self, ctx) {
            Ok(v) => Ok(v),
            Err(e) => Err(super::GGUFError::InvalidTensorData(e.to_string())),
        }
    }

    /// Convenience: materialize a named tensor as packed Q8_0 without upcasting.
    pub fn materialize_q8_0_packed<TCtx: TensorElement>(&self, name: &str, ctx: &Context<TCtx>) -> Result<QuantizedQ8_0Tensor, GGUFError> {
        let descriptor = self
            .tensors
            .get(name)
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Tensor '{}' not found in GGUF metadata", name)))?;
        descriptor.materialize_q8_0_packed(self.gguf_file.as_ref(), ctx, self.layout_hint)
    }
}
