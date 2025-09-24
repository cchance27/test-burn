#![allow(dead_code)]
use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

// Import quantization modules
pub mod quant;

// Import model loader
pub mod model_loader;

#[derive(Debug, Error)]
pub enum GGUFError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Invalid data")]
    InvalidData,
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    #[error("Invalid tensor data: {0}")]
    InvalidTensorData(String),
    #[error("Memory mapping error: {0}")]
    MemoryMappingError(String),
    #[error("Dequantization error: {0}")]
    DequantizationError(String),
    #[error("Dimension mismatch: expected {expected}, actual {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGUFDataType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q4_2,
    Q4_3,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ1S,
    IQ4NL,
    IQ3S,
    IQ2S,
    IQ4XS,
    I8,
    I16,
    I32,
    I64,
    F64,
    IQ1M,
    BF16,
    Q4044,
    Q4048,
    Q4088,
    TQ10,
    TQ20,
    IQ4NL44,
    IQ4NL48,
    IQ4NL88,
    Array,
    Unknown(u32),
}

impl GGUFDataType {
    fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            4 => Self::Q4_2,
            5 => Self::Q4_3,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            31 => Self::Q4044,
            32 => Self::Q4048,
            33 => Self::Q4088,
            34 => Self::TQ10,
            35 => Self::TQ20,
            36 => Self::IQ4NL44,
            37 => Self::IQ4NL48,
            38 => Self::IQ4NL88,
            39 => Self::Array,
            _ => Self::Unknown(value),
        }
    }

    /// Return the raw numeric code used in the GGUF file for this data type.
    fn as_u32(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q4_2 => 4,
            Self::Q4_3 => 5,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ2XXS => 16,
            Self::IQ2XS => 17,
            Self::IQ3XXS => 18,
            Self::IQ1S => 19,
            Self::IQ4NL => 20,
            Self::IQ3S => 21,
            Self::IQ2S => 22,
            Self::IQ4XS => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::IQ1M => 29,
            Self::BF16 => 30,
            Self::Q4044 => 31,
            Self::Q4048 => 32,
            Self::Q4088 => 33,
            Self::TQ10 => 34,
            Self::TQ20 => 35,
            Self::IQ4NL44 => 36,
            Self::IQ4NL48 => 37,
            Self::IQ4NL88 => 38,
            Self::Array => 39,
            Self::Unknown(value) => *value,
        }
    }
}

#[derive(Debug)]
pub struct GGUFHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub entries: HashMap<String, GGUFValue>,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug)]
pub struct GGUTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub data_type: GGUFDataType,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GGUFFile {
    pub header: GGUFHeader,
    pub metadata: GGUFMetadata,
    pub tensors: Vec<GGUTensorInfo>,
    mmap: Mmap,
    // For memory management
    pub file_path: String,
}

impl GGUFFile {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GGUFError> {
        let file_path = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&path).map_err(GGUFError::Io)?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| GGUFError::MemoryMappingError(e.to_string()))?;

        let mut reader = &mmap[..];

        // Read header
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        //println!("Magic: {:?}", &magic);
        if &magic != b"GGUF" {
            return Err(GGUFError::InvalidMagic);
        }

        let version = reader.read_u32::<LittleEndian>()?;
        //println!("Version: {}", version);
        if version != 3 {
            return Err(GGUFError::UnsupportedVersion(version));
        }

        let tensor_count = reader.read_u64::<LittleEndian>()?;
        //println!("Tensor count: {}", tensor_count);
        let metadata_count = reader.read_u64::<LittleEndian>()?;
        //println!("Metadata count: {}", metadata_count);

        let header = GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_count,
        };

        // Read metadata
        //println!(
        //    "Mmap length: {}, Reader length: {}",
        //    mmap.len(),
        //    reader.len()
        //);
        let metadata = Self::read_metadata(&mut reader, metadata_count as usize)?;
        //println!(
        //    "Finished reading metadata, reader position: {}",
        //    mmap.len() - reader.len()
        //);

        // Read tensor infos
        //println!(
        //    "About to read tensors, reader position: {}",
        //    mmap.len() - reader.len()
        //);
        let tensors = Self::read_tensors(&mut reader, tensor_count as usize)?;
        //println!(
        //    "Finished reading tensors, reader position: {}",
        //    mmap.len() - reader.len()
        //);

        Ok(Self {
            header,
            metadata,
            tensors,
            mmap,
            file_path,
        })
    }

    fn read_metadata(reader: &mut &[u8], count: usize) -> Result<GGUFMetadata, GGUFError> {
        let mut entries = HashMap::new();
        for _i in 0..count {
            let key = Self::read_string_direct(reader)?;
            let value = Self::read_value(reader)?;
            entries.insert(key, value);
        }
        Ok(GGUFMetadata { entries })
    }

    fn read_string(reader: &mut &[u8]) -> Result<String, GGUFError> {
        let len = reader.read_u64::<LittleEndian>()? as usize;
        if len > 1024 * 1024 {
            return Err(GGUFError::InvalidData);
        }

        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;

        String::from_utf8(buf).map_err(|_| GGUFError::InvalidData)
    }

    fn read_string_direct(reader: &mut &[u8]) -> Result<String, GGUFError> {
        let len = reader.read_u64::<LittleEndian>()? as usize;

        // Add a sanity check for string length to prevent memory issues
        if len > 1024 * 1024 {
            // 1MB limit for strings
            return Err(GGUFError::InvalidData);
        }

        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;

        String::from_utf8(buf).map_err(|_| GGUFError::InvalidData)
    }

    fn read_value(reader: &mut &[u8]) -> Result<GGUFValue, GGUFError> {
        let value_type = reader.read_u32::<LittleEndian>()?;
        match value_type {
            0 => {
                let val = reader.read_u8()?;
                Ok(GGUFValue::U8(val))
            }
            1 => {
                let val = reader.read_i8()?;
                Ok(GGUFValue::I8(val))
            }
            2 => {
                let val = reader.read_u16::<LittleEndian>()?;
                Ok(GGUFValue::U16(val))
            }
            3 => {
                let val = reader.read_i16::<LittleEndian>()?;
                Ok(GGUFValue::I16(val))
            }
            4 => {
                let val = reader.read_u32::<LittleEndian>()?;
                Ok(GGUFValue::U32(val))
            }
            5 => {
                let val = reader.read_i32::<LittleEndian>()?;
                Ok(GGUFValue::I32(val))
            }
            6 => {
                let val = reader.read_f32::<LittleEndian>()?;
                Ok(GGUFValue::F32(val))
            }
            7 => {
                let val = reader.read_u8()?;
                let bool_val = val != 0;
                Ok(GGUFValue::Bool(bool_val))
            }
            8 => {
                let val = Self::read_string(reader)?;
                Ok(GGUFValue::String(val))
            }
            9 => {
                let element_type = reader.read_u32::<LittleEndian>()?;
                let len = reader.read_u64::<LittleEndian>()? as usize;

                // Add a sanity check for array length to prevent memory issues
                if len > 1024 * 1024 {
                    // 1MB limit for arrays
                    return Err(GGUFError::InvalidData);
                }

                let mut values = Vec::with_capacity(len);
                for _i in 0..len {
                    let value = match element_type {
                        0 => {
                            let val = reader.read_u8()?;
                            GGUFValue::U8(val)
                        }
                        1 => {
                            let val = reader.read_i8()?;
                            GGUFValue::I8(val)
                        }
                        2 => {
                            let val = reader.read_u16::<LittleEndian>()?;
                            GGUFValue::U16(val)
                        }
                        3 => {
                            let val = reader.read_i16::<LittleEndian>()?;
                            GGUFValue::I16(val)
                        }
                        4 => {
                            let val = reader.read_u32::<LittleEndian>()?;
                            GGUFValue::U32(val)
                        }
                        5 => {
                            let val = reader.read_i32::<LittleEndian>()?;
                            GGUFValue::I32(val)
                        }
                        6 => {
                            let val = reader.read_f32::<LittleEndian>()?;
                            GGUFValue::F32(val)
                        }
                        7 => {
                            let val = reader.read_u8()?;
                            let bool_val = val != 0;
                            GGUFValue::Bool(bool_val)
                        }
                        8 => {
                            let val = Self::read_string(reader)?;
                            GGUFValue::String(val)
                        }
                        9 => {
                            let val = reader.read_u64::<LittleEndian>()?;
                            GGUFValue::U64(val)
                        }
                        10 => {
                            let val = reader.read_i64::<LittleEndian>()?;
                            GGUFValue::I64(val)
                        }
                        11 => {
                            let val = reader.read_f64::<LittleEndian>()?;
                            GGUFValue::F64(val)
                        }
                        _ => return Err(GGUFError::InvalidData),
                    };
                    values.push(value);
                }
                Ok(GGUFValue::Array(values))
            }
            _ => Err(GGUFError::InvalidData),
        }
    }

    fn read_tensors(reader: &mut &[u8], count: usize) -> Result<Vec<GGUTensorInfo>, GGUFError> {
        let mut tensors = Vec::with_capacity(count);
        for _i in 0..count {
            let name = Self::read_string(reader)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()? as usize;
            let mut dimensions = Vec::with_capacity(n_dimensions);
            for _ in 0..n_dimensions {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let data_type = GGUFDataType::from_u32(reader.read_u32::<LittleEndian>()?);
            let offset = reader.read_u64::<LittleEndian>()?;

            //if name == "token_embd.weight" {
            //    println!("token_embd.weight tensor #{}: offset={}, dimensions={:?}, data_type={:?}",
            //             i, offset, dimensions, data_type);
            //}

            tensors.push(GGUTensorInfo {
                name,
                dimensions,
                data_type,
                offset,
            });
        }
        Ok(tensors)
    }

    pub fn get_tensor_data(&self, tensor: &GGUTensorInfo) -> Result<&[u8], GGUFError> {
        // Calculate the actual start of tensor_data section
        // This includes the header + tensor_infos + padding
        let tensor_data_start = self.calculate_tensor_data_start();
        let start = tensor_data_start + tensor.offset as usize;
        let size = self.calculate_actual_tensor_size(tensor);
        let end = start + size;

        // Bounds checking
        if end > self.mmap.len() {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor data out of bounds: start={}, size={}, end={}, mmap_len={}, tensor_data_start={}",
                start,
                size,
                end,
                self.mmap.len(),
                tensor_data_start
            )));
        }

        // println!("Tensor '{}' data: file_start={}, tensor_offset={}, calculated_start={}, size={}",
        //          tensor.name, tensor_data_start, tensor.offset, start, size);

        Ok(&self.mmap[start..end])
    }

    /// Calculate where the tensor_data section starts in the file
    fn calculate_tensor_data_start(&self) -> usize {
        // Get the alignment value (default to 32 if not specified)
        let alignment =
            if let Some(GGUFValue::U32(val)) = self.metadata.entries.get("general.alignment") {
                *val as usize
            } else {
                32 // Default alignment
            };

        // Calculate the size of header + metadata + tensor infos accurately
        let mut offset = 4 + 4 + 8 + 8; // magic + version + tensor_count + metadata_count

        // Add metadata size
        for (key, value) in &self.metadata.entries {
            offset += 8 + key.len(); // key length + key string
            offset += 4; // value type
            match value {
                GGUFValue::String(s) => offset += 8 + s.len(),
                GGUFValue::U32(_) => offset += 4,
                GGUFValue::U64(_) => offset += 8,
                GGUFValue::I32(_) => offset += 4,
                GGUFValue::I64(_) => offset += 8,
                GGUFValue::F32(_) => offset += 4,
                GGUFValue::F64(_) => offset += 8,
                GGUFValue::U8(_) => offset += 1,
                GGUFValue::I8(_) => offset += 1,
                GGUFValue::U16(_) => offset += 2,
                GGUFValue::I16(_) => offset += 2,
                GGUFValue::Bool(_) => offset += 1,
                GGUFValue::Array(arr) => {
                    offset += 8 + 4; // array length + element type
                    // Add size for each element in the array
                    for elem in arr {
                        match elem {
                            GGUFValue::String(s) => offset += 8 + s.len(),
                            GGUFValue::U32(_) => offset += 4,
                            GGUFValue::U64(_) => offset += 8,
                            GGUFValue::I32(_) => offset += 4,
                            GGUFValue::I64(_) => offset += 8,
                            GGUFValue::F32(_) => offset += 4,
                            GGUFValue::F64(_) => offset += 8,
                            GGUFValue::U8(_) => offset += 1,
                            GGUFValue::I8(_) => offset += 1,
                            GGUFValue::U16(_) => offset += 2,
                            GGUFValue::I16(_) => offset += 2,
                            GGUFValue::Bool(_) => offset += 1,
                            GGUFValue::Array(_) => unimplemented!("Unsupported Nested Array"),
                        }
                    }
                }
            }
        }

        // Add tensor info size for each tensor
        for tensor in &self.tensors {
            offset += 8 + tensor.name.len(); // name length + name
            offset += 4; // n_dimensions
            offset += tensor.dimensions.len() * 8; // dimensions
            offset += 4; // type
            offset += 8; // offset
        }

        // Align to the required boundary
        offset.div_ceil(alignment) * alignment
    }

    fn calculate_tensor_size(&self, tensor: &GGUTensorInfo) -> usize {
        // Simplified: just multiply dimensions
        let mut size = 1;
        for &dim in &tensor.dimensions {
            size *= dim as usize;
        }
        // Multiply by element size based on type
        size * self.get_element_size(tensor.data_type)
    }

    fn get_element_size(&self, data_type: GGUFDataType) -> usize {
        match data_type {
            GGUFDataType::F32 => 4,
            GGUFDataType::F16 => 2,
            //GGUFDataType::Q4_0 => 2 + block_size / 2,
            //GGUFDataType::Q4_1 => 2 + 2 + block_size / 2,
            GGUFDataType::Q4_2 => 0,
            GGUFDataType::Q4_3 => 0,
            //GGUFDataType::Q5_0 => 2 + 4 + block_size / 2,
            //GGUFDataType::Q5_1 => 2 + 2 + 4 + block_size / 2,
            //GGUFDataType::Q8_0 => 2 + block_size,
            //GGUFDataType::Q8_1 => 4 + 4 + block_size,
            //GGUFDataType::Q2K => block_size / 16 + block_size / 4 + 2 + 2,
            //GGUFDataType::Q3K => block_size / 8 + block_size / 4 + 12 + 2,
            //GGUFDataType::Q4K => 2 + 2 + 12 + block_size / 2,
            //GGUFDataType::Q5K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
            //GGUFDataType::Q6K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
            //GGUFDataType::Q8K => 4 + block_size + block_size / 16 * 2,
            //GGUFDataType::IQ2XXS => 2 + block_size / 8 * 2,
            //GGUFDataType::IQ2XS => 2 + block_size / 8 * 2 + block_size / 32,
            //GGUFDataType::IQ3XXS => 2 + 3 * (block_size / 8),
            //GGUFDataType::IQ1S => 2 + block_size / 8 + block_size / 16,
            //GGUFDataType::IQ4NL => 2 + 16,
            //GGUFDataType::IQ3S => 2 + 13 * (block_size / 32) + block_size / 64,
            //GGUFDataType::IQ2S => 2 + block_size / 4 + block_size / 16,
            //GGUFDataType::IQ4XS => 2 + 2 + block_size / 64 + block_size / 2,
            GGUFDataType::I8 => 1,
            GGUFDataType::I16 => 2,
            GGUFDataType::I32 => 4,
            GGUFDataType::I64 => 8,
            GGUFDataType::F64 => 8,
            //GGUFDataType::IQ1M => block_size / 8 + block_size / 16 + block_size / 32,
            GGUFDataType::BF16 => 2,
            GGUFDataType::IQ4NL44 => 0,
            GGUFDataType::IQ4NL48 => 0,
            GGUFDataType::IQ4NL88 => 0,
            //GGUFDataType::TQ10 => 2 + block_size / 64 + (block_size - 4 * block_size / 64) / 5,
            //GGUFDataType::TQ20 => 2 + block_size / 4,
            GGUFDataType::Q4044 => 0,
            GGUFDataType::Q4048 => 0,
            GGUFDataType::Q4088 => 0,
            _ => unreachable!("Unsupported data type size"),
        }
    }

    /// Calculate the actual size of tensor data in bytes
    fn calculate_actual_tensor_size(&self, tensor: &GGUTensorInfo) -> usize {
        match tensor.data_type {
            GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                // For Q8 types, calculate based on blocks
                let element_count: usize = tensor.dimensions.iter().map(|&d| d as usize).product();
                let weights_per_block = 32;
                let blocks = element_count.div_ceil(weights_per_block);

                let block_size = match tensor.data_type {
                    GGUFDataType::Q8_0 => 34, // 2 (scale) + 32 (weights)
                    GGUFDataType::Q8_1 => 36, // 2 (scale) + 2 (delta) + 32 (weights)
                    _ => 36,                  // Should not happen
                };

                blocks * block_size
            }
            _ => {
                // For other types, use the original calculation
                let mut size = 1;
                for &dim in &tensor.dimensions {
                    size *= dim as usize;
                }
                size * self.get_element_size(tensor.data_type)
            }
        }
    }

    /// Validate tensor data for common issues
    pub fn validate_tensor_data(&self, tensor: &GGUTensorInfo) -> Result<(), GGUFError> {
        let data = self.get_tensor_data(tensor)?;
        let expected_size = self.calculate_actual_tensor_size(tensor);

        if data.len() != expected_size {
            return Err(GGUFError::InvalidTensorData(format!(
                "Tensor {} has incorrect size: expected {}, got {}",
                tensor.name,
                expected_size,
                data.len()
            )));
        }

        // For Q8 types, validate block structure
        if tensor.data_type == GGUFDataType::Q8_0 || tensor.data_type == GGUFDataType::Q8_1 {
            let block_size = match tensor.data_type {
                GGUFDataType::Q8_0 => 34,
                GGUFDataType::Q8_1 => 36,
                _ => 36,
            };

            if data.len() % block_size != 0 {
                return Err(GGUFError::InvalidTensorData(format!(
                    "Tensor {} has invalid block alignment: data size {} is not divisible by block size {}",
                    tensor.name,
                    data.len(),
                    block_size
                )));
            }
        }

        Ok(())
    }

    /// Offload tensor data to disk to free up memory
    #[allow(dead_code)]
    pub fn offload_tensor(&self, tensor_info: &GGUTensorInfo) -> Result<(), GGUFError> {
        // In a real implementation, this would save tensor data to disk
        // and mark it as offloaded in some tracking structure
        println!("Offloading tensor: {} to disk", tensor_info.name);
        Ok(())
    }

    /// Load tensor data back from disk
    #[allow(dead_code)]
    pub fn load_tensor(&self, tensor_info: &GGUTensorInfo) -> Result<Vec<u8>, GGUFError> {
        // In a real implementation, this would load tensor data from disk
        // For now, we'll just return the data from the memory-mapped file
        let data = self.get_tensor_data(tensor_info)?;
        println!("Loading tensor: {} from disk/file", tensor_info.name);
        Ok(data.to_vec())
    }

    /// Blockswap implementation - load/unload tensors in blocks
    #[allow(dead_code)]
    pub fn blockswap_tensors(
        &self,
        tensors_to_load: &[&GGUTensorInfo],
        tensors_to_unload: &[&GGUTensorInfo],
    ) -> Result<(), GGUFError> {
        // Unload tensors first
        for tensor in tensors_to_unload {
            self.offload_tensor(tensor)?;
        }

        // Load tensors
        for tensor in tensors_to_load {
            let _data = self.load_tensor(tensor)?;
            // In a real implementation, we would store this data somewhere
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::metallic;

    use super::*;

    #[test]
    fn test_load_gguf_file() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
        match GGUFFile::load(path) {
            Ok(gguf) => {
                println!("Successfully loaded GGUF file:");
                println!(
                    "Magic: {:?}",
                    std::str::from_utf8(&gguf.header.magic).unwrap_or("Invalid UTF-8")
                );
                println!("Version: {}", gguf.header.version);
                println!("Tensor count: {}", gguf.header.tensor_count);
                println!("Metadata count: {}", gguf.header.metadata_count);

                println!("Metadata entries: {}", gguf.metadata.entries.len());
                for (key, value) in &gguf.metadata.entries {
                    // For arrays, just print the count to avoid huge output
                    match value {
                        GGUFValue::Array(tokens) => {
                            println!("  {}: Array with {} elements", key, tokens.len());
                        }
                        _ => {
                            println!("  {}: {:?}", key, value);
                        }
                    }
                }

                // Print tensor count and first few tensor names
                println!("First 10 tensors:");
                let mut count = 0;
                for tensor in &gguf.tensors {
                    if count < 10 {
                        println!(
                            "  {}: {:?} ({:?})",
                            tensor.name, tensor.dimensions, tensor.data_type
                        );
                        count += 1;
                    } else {
                        break;
                    }
                }
                println!(
                    "... and {} more tensors",
                    gguf.tensors.len().saturating_sub(10)
                );
            }
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        }
    }

    #[test]
    fn test_gguf_to_metallic_tensor() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
        match GGUFFile::load(path) {
            Ok(gguf) => {
                // Initialize Metallic context
                let _context = match metallic::Context::new() {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        panic!("Failed to create Metallic context: {:?}", e);
                    }
                };

                // Test with the first tensor (should be Q8_1)
                if let Some(first_tensor) = gguf.tensors.first() {
                    println!(
                        "Testing tensor: {} ({:?})",
                        first_tensor.name, first_tensor.data_type
                    );
                    println!("Dimensions: {:?}", first_tensor.dimensions);

                    // Calculate expected element count
                    let expected_elements: usize = first_tensor
                        .dimensions
                        .iter()
                        .map(|&d| d as usize)
                        .product();
                    println!("Expected elements: {}", expected_elements);

                    // Get tensor data
                    match gguf.get_tensor_data(first_tensor) {
                        Ok(data) => {
                            println!("Raw data length: {}", data.len());
                        }
                        Err(e) => {
                            panic!("Error getting tensor data: {}", e);
                        }
                    }

                    match metallic::Tensor::try_from((&gguf, first_tensor)) {
                        Ok(metallic_tensor) => {
                            println!(
                                "Successfully converted tensor '{}' to Metallic tensor",
                                first_tensor.name
                            );
                            println!("Tensor dimensions: {:?}", metallic_tensor.dims);
                            println!("Tensor size (elements): {}", metallic_tensor.len());
                        }
                        Err(e) => {
                            panic!("Error converting tensor '{}': {}", first_tensor.name, e);
                        }
                    }
                }
            }
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        }
    }

    #[test]
    fn test_debug_q8_format() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
        match GGUFFile::load(path) {
            Ok(gguf) => {
                // Debug the first tensor
                if let Some(_first_tensor) = gguf.tensors.first() {}

                // Debug a few more tensors
                for (i, _tensor) in gguf.tensors.iter().enumerate() {
                    if i >= 3 {
                        break;
                    }
                    println!("\n--- Tensor {} ---", i);
                }
            }
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        }
    }

    #[test]
    fn test_initial_inference() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
        match GGUFFile::load(path) {
            Ok(gguf) => {
                // Initialize Metallic context
                let _context = match metallic::Context::new() {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        panic!("Failed to create Metallic context: {:?}", e);
                    }
                };

                // Test with the first tensor (should be Q8_1)
                if let Some(first_tensor) = gguf.tensors.first() {
                    println!(
                        "Testing tensor: {} ({:?})",
                        first_tensor.name, first_tensor.data_type
                    );
                    println!("Dimensions: {:?}", first_tensor.dimensions);
                    match metallic::Tensor::try_from((&gguf, first_tensor)) {
                        Ok(metallic_tensor) => {
                            println!(
                                "Successfully converted tensor '{}' to Metallic tensor",
                                first_tensor.name
                            );
                            println!("Tensor dimensions: {:?}", metallic_tensor.dims);
                            println!("Tensor size (elements): {}", metallic_tensor.len());

                            // Print first few values for verification
                            let slice = metallic_tensor.as_slice();
                            let print_count = std::cmp::min(10, slice.len());
                            println!("First {} values: {:?}", print_count, &slice[..print_count]);
                        }
                        Err(e) => {
                            panic!("Error converting tensor '{}': {}", first_tensor.name, e);
                        }
                    }
                }
            }
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod memory_tests {
    use super::*;

    #[test]
    fn test_memory_management() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
        match GGUFFile::load(path) {
            Ok(gguf) => {
                // Test with the first few tensors
                let tensors_to_test: Vec<&GGUTensorInfo> = gguf.tensors.iter().take(3).collect();

                for tensor in &tensors_to_test {
                    println!("Testing memory management for tensor: {}", tensor.name);

                    // Test offloading
                    match gguf.offload_tensor(tensor) {
                        Ok(_) => println!("Successfully offloaded tensor: {}", tensor.name),
                        Err(e) => println!("Error offloading tensor {}: {}", tensor.name, e),
                    }

                    // Test loading
                    match gguf.load_tensor(tensor) {
                        Ok(data) => println!(
                            "Successfully loaded tensor: {} ({} bytes)",
                            tensor.name,
                            data.len()
                        ),
                        Err(e) => println!("Error loading tensor {}: {}", tensor.name, e),
                    }
                }

                // Test blockswapping with first two tensors
                if tensors_to_test.len() >= 2 {
                    let load_tensors = vec![tensors_to_test[1]];
                    let unload_tensors = vec![tensors_to_test[0]];

                    match gguf.blockswap_tensors(&load_tensors, &unload_tensors) {
                        Ok(_) => println!("Successfully blockswapped tensors"),
                        Err(e) => println!("Error blockswapping tensors: {}", e),
                    }
                }
            }
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        }
    }
}
