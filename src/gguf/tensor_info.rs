use bytemuck::try_cast_slice;
use half::f16;

use crate::gguf::{GGUFDataType, GGUFError, GGUFFile};

/// Borrowed GGUF tensor data interpreted according to its element type.
pub enum GGUFRawTensor<'data> {
    F32(&'data [f32]),
    F16(&'data [f16]),
    Bytes(&'data [u8], GGUFDataType),
}

#[derive(Debug)]
pub struct GGUTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub data_type: GGUFDataType,
    pub offset: u64,
}

impl GGUTensorInfo {
    /// Interpret the raw tensor buffer as a typed slice when possible.
    pub fn into<'a>(&self, file: &'a GGUFFile) -> Result<GGUFRawTensor<'a>, GGUFError> {
        let raw = file.get_tensor_data(self)?;

        match self.data_type {
            GGUFDataType::F32 => {
                let casted = try_cast_slice(raw)
                    .map_err(|_| GGUFError::InvalidTensorData(format!("Tensor '{}' is not valid f32 data", self.name)))?;
                Ok(GGUFRawTensor::F32(casted))
            }
            GGUFDataType::F16 => {
                let casted = try_cast_slice(raw)
                    .map_err(|_| GGUFError::InvalidTensorData(format!("Tensor '{}' is not valid f16 data", self.name)))?;
                Ok(GGUFRawTensor::F16(casted))
            }
            _ => Ok(GGUFRawTensor::Bytes(raw, self.data_type)),
        }
    }
}
