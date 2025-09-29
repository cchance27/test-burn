use super::{GGUFDataType, GGUFError, GGUFFile};
use crate::{
    gguf::{GGUFValue, GGUTensorInfo},
    metallic::{
        Context, Tensor,
        error::MetalError,
        operation::CommandBuffer,
        tensor::{Dtype, RetainedBuffer},
    },
};
use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLDevice, MTLResourceOptions};
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

const STAGING_ALIGNMENT: usize = 64 * 1024;

fn align_up(value: usize, alignment: usize) -> usize {
    if value == 0 {
        0
    } else {
        value.div_ceil(alignment) * alignment
    }
}

struct HostStagingBuffer {
    buffer: RetainedBuffer,
    capacity: usize,
}

struct HostStagingAllocator {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    free_list: Vec<HostStagingBuffer>,
}

impl HostStagingAllocator {
    fn new(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device: device.clone(),
            free_list: Vec::new(),
        }
    }

    fn acquire(&mut self, required_bytes: usize) -> Result<HostStagingLease<'_>, MetalError> {
        let buffer = if let Some(idx) = self.free_list.iter().position(|entry| entry.capacity >= required_bytes) {
            self.free_list.swap_remove(idx)
        } else {
            self.allocate_buffer(required_bytes)?
        };

        Ok(HostStagingLease {
            allocator: self,
            buffer: Some(buffer),
        })
    }

    fn allocate_buffer(&self, required_bytes: usize) -> Result<HostStagingBuffer, MetalError> {
        let capacity = std::cmp::max(align_up(required_bytes, STAGING_ALIGNMENT), STAGING_ALIGNMENT);
        let buffer = self
            .device
            .newBufferWithLength_options(capacity, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(capacity))?;

        Ok(HostStagingBuffer { buffer, capacity })
    }
}

struct HostStagingLease<'a> {
    allocator: &'a mut HostStagingAllocator,
    buffer: Option<HostStagingBuffer>,
}

impl<'a> HostStagingLease<'a> {
    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        let buf = self.buffer.as_ref().expect("staging buffer missing");
        assert!(len <= buf.capacity, "requested slice exceeds staging capacity");
        unsafe { std::slice::from_raw_parts_mut(buf.buffer.contents().as_ptr() as *mut u8, len) }
    }

    fn buffer(&self) -> &RetainedBuffer {
        &self.buffer.as_ref().expect("staging buffer missing").buffer
    }
}

impl<'a> Drop for HostStagingLease<'a> {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.allocator.free_list.push(buf);
        }
    }
}

fn metal_to_tensor_error(tensor_name: &str, stage: &str, err: MetalError) -> GGUFError {
    GGUFError::InvalidTensorData(format!("{} for tensor '{}': {}", stage, tensor_name, err))
}

fn upload_f32_tensor_from_bytes(
    tensor_name: &str,
    dims: Vec<usize>,
    bytes: &[u8],
    context: &Context,
    staging: &mut HostStagingAllocator,
) -> Result<Tensor, GGUFError> {
    let byte_len = bytes.len();

    let dest_buf = context
        .device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModePrivate)
        .ok_or_else(|| {
            metal_to_tensor_error(
                tensor_name,
                "Failed to allocate device buffer",
                MetalError::BufferCreationFailed(byte_len),
            )
        })?;

    if byte_len > 0 {
        let mut lease = staging
            .acquire(byte_len)
            .map_err(|err| metal_to_tensor_error(tensor_name, "Failed to allocate staging buffer", err))?;

        lease.as_mut_slice(byte_len).copy_from_slice(bytes);

        let command_buffer = CommandBuffer::new(&context.command_queue)
            .map_err(|err| metal_to_tensor_error(tensor_name, "Failed to create command buffer", err))?;

        let encoder = command_buffer
            .raw()
            .blitCommandEncoder()
            .ok_or_else(|| GGUFError::InvalidTensorData(format!("Blit encoder not available while uploading tensor '{}'", tensor_name)))?;

        unsafe {
            encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(lease.buffer(), 0, &dest_buf, 0, byte_len);
        }
        encoder.endEncoding();
        command_buffer.commit();
        command_buffer.wait();
    }

    Tensor::from_existing_buffer(dest_buf, dims, Dtype::F32, &context.device, &context.command_queue, 0, false)
        .map_err(|err| metal_to_tensor_error(tensor_name, "Failed to finalize tensor", err))
}

fn upload_f32_tensor_from_slice(
    tensor_name: &str,
    dims: Vec<usize>,
    data: &[f32],
    context: &Context,
    staging: &mut HostStagingAllocator,
) -> Result<Tensor, GGUFError> {
    let byte_len = std::mem::size_of_val(data);
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
    upload_f32_tensor_from_bytes(tensor_name, dims, bytes, context, staging)
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
        let mut staging_allocator = HostStagingAllocator::new(&context.device);

        for tensor_info in &self.gguf_file.tensors {
            match self.load_tensor(context, &mut staging_allocator, tensor_info) {
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

    fn load_tensor(&self, context: &Context, staging: &mut HostStagingAllocator, tensor_info: &GGUTensorInfo) -> Result<Tensor, GGUFError> {
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
                upload_f32_tensor_from_bytes(&tensor_info.name, dims, raw, context, staging)
            }
            GGUFDataType::F16 => {
                let f32_data = convert_f16_bytes(raw)?;
                if f32_data.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: f32_data.len(),
                    });
                }
                upload_f32_tensor_from_slice(&tensor_info.name, dims, &f32_data, context, staging)
            }
            GGUFDataType::BF16 => {
                let f32_data = convert_bf16_bytes(raw)?;
                if f32_data.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: f32_data.len(),
                    });
                }
                upload_f32_tensor_from_slice(&tensor_info.name, dims, &f32_data, context, staging)
            }
            GGUFDataType::F64 => {
                let f32_data = convert_f64_bytes(raw)?;
                if f32_data.len() != expected_elements {
                    return Err(GGUFError::DimensionMismatch {
                        expected: expected_elements,
                        actual: f32_data.len(),
                    });
                }
                upload_f32_tensor_from_slice(&tensor_info.name, dims, &f32_data, context, staging)
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
                upload_f32_tensor_from_slice(&tensor_info.name, dims, &dequant, context, staging)
            }
            _ => Err(GGUFError::InvalidTensorData(format!(
                "Unsupported tensor data type: {:?}",
                tensor_info.data_type
            ))),
        }
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
