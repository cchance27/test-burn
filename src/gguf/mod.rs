use byteorder::{LittleEndian, ReadBytesExt};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

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
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGUFDataType {
    F64,
    F32,
    F16,
    BF16,
    Q8_0,
    Q8_1,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    String,
    Array,
    Unknown(u32),
}

impl GGUFDataType {
    fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::F64,
            1 => Self::F32,
            2 => Self::F16,
            3 => Self::BF16,
            7 => Self::Q8_0,
            8 => Self::Q8_1,
            9 => Self::Q4_0,
            10 => Self::Q4_1,
            11 => Self::Q5_0,
            12 => Self::Q5_1,
            13 => Self::Q2K,
            14 => Self::Q3K,
            15 => Self::Q4K,
            16 => Self::Q5K,
            17 => Self::Q6K,
            18 => Self::Q8K,
            19 => Self::I8,
            20 => Self::I16,
            21 => Self::I32,
            22 => Self::I64,
            23 => Self::U8,
            24 => Self::U16,
            25 => Self::U32,
            26 => Self::U64,
            27 => Self::Bool,
            28 => Self::String,
            29 => Self::Array,
            _ => Self::Unknown(value),
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

#[derive(Debug)]
pub struct GGUFMetadata {
    pub entries: HashMap<String, GGUFValue>,
}

#[derive(Debug)]
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
}

impl GGUFFile {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, GGUFError> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

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
        })
    }

    fn read_metadata(reader: &mut &[u8], count: usize) -> Result<GGUFMetadata, GGUFError> {
        //println!("Reading {} metadata entries", count);
        let mut entries = HashMap::new();
        for _i in 0..count {
            //println!("Reading metadata entry {}", i);

            let key = Self::read_string_direct(reader)?;
            //println!("Key: {}", key);

            let value = Self::read_value(reader)?;
            //println!("Value: {:?}", value);
            entries.insert(key, value);
        }
        Ok(GGUFMetadata { entries })
    }

    fn read_string(reader: &mut &[u8]) -> Result<String, GGUFError> {
        //println!("Reading string, reader length: {}", reader.len());

        let len = reader.read_u64::<LittleEndian>()? as usize;

        // Add a sanity check for string length to prevent memory issues
        if len > 1024 * 1024 {
            // 1MB limit for strings
            return Err(GGUFError::InvalidData);
        }

        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        let result = String::from_utf8(buf).map_err(|_| GGUFError::InvalidData);
        //println!("Read string: {:?}", result);

        result
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
        let result = String::from_utf8(buf).map_err(|_| GGUFError::InvalidData);
        //println!("Read string: {:?}", result);

        result
    }

    fn read_value(reader: &mut &[u8]) -> Result<GGUFValue, GGUFError> {
        let value_type = reader.read_u32::<LittleEndian>()?;
        //println!("Value type: {}", value_type);
        match value_type {
            0 => {
                let val = reader.read_u8()?;
                //println!("Read U8: {}", val);
                Ok(GGUFValue::U8(val))
            }
            1 => {
                let val = reader.read_i8()?;
                //println!("Read I8: {}", val);
                Ok(GGUFValue::I8(val))
            }
            2 => {
                let val = reader.read_u16::<LittleEndian>()?;
                //println!("Read U16: {}", val);
                Ok(GGUFValue::U16(val))
            }
            3 => {
                let val = reader.read_i16::<LittleEndian>()?;
                //println!("Read I16: {}", val);
                Ok(GGUFValue::I16(val))
            }
            4 => {
                let val = reader.read_u32::<LittleEndian>()?;
                //println!("Read U32: {}", val);
                Ok(GGUFValue::U32(val))
            }
            5 => {
                let val = reader.read_i32::<LittleEndian>()?;
                //println!("Read I32: {}", val);
                Ok(GGUFValue::I32(val))
            }
            6 => {
                let val = reader.read_f32::<LittleEndian>()?;
                //println!("Read F32: {}", val);
                Ok(GGUFValue::F32(val))
            }
            7 => {
                let val = reader.read_u8()?;
                let bool_val = val != 0;
                //println!("Read Bool: {} ({})", bool_val, val);
                Ok(GGUFValue::Bool(bool_val))
            }
            8 => {
                let val = Self::read_string(reader)?;
                //println!("Read String: {}", val);
                Ok(GGUFValue::String(val))
            }
            9 => {
                //println!("Reading array, reader length: {}", reader.len());

                let element_type = reader.read_u32::<LittleEndian>()?;
                //println!("Array element type: {}", element_type);
                let len = reader.read_u64::<LittleEndian>()? as usize;
                //println!("Array length: {}", len);

                // Add a sanity check for array length to prevent memory issues
                if len > 1024 * 1024 {
                    // 1MB limit for arrays
                    return Err(GGUFError::InvalidData);
                }

                let mut values = Vec::with_capacity(len);
                //println!("Array of {} elements", len);
                for _i in 0..len {
                    //println!("Reading array element {}", i);
                    // For array elements, we don't read the type field since all elements have the same type
                    let value = match element_type {
                        0 => {
                            let val = reader.read_u8()?;
                            //println!("Read U8: {}", val);
                            GGUFValue::U8(val)
                        }
                        1 => {
                            let val = reader.read_i8()?;
                            //println!("Read I8: {}", val);
                            GGUFValue::I8(val)
                        }
                        2 => {
                            let val = reader.read_u16::<LittleEndian>()?;
                            //println!("Read U16: {}", val);
                            GGUFValue::U16(val)
                        }
                        3 => {
                            let val = reader.read_i16::<LittleEndian>()?;
                            //println!("Read I16: {}", val);
                            GGUFValue::I16(val)
                        }
                        4 => {
                            let val = reader.read_u32::<LittleEndian>()?;
                            //println!("Read U32: {}", val);
                            GGUFValue::U32(val)
                        }
                        5 => {
                            let val = reader.read_i32::<LittleEndian>()?;
                            //println!("Read I32: {}", val);
                            GGUFValue::I32(val)
                        }
                        6 => {
                            let val = reader.read_f32::<LittleEndian>()?;
                            //println!("Read F32: {}", val);
                            GGUFValue::F32(val)
                        }
                        7 => {
                            let val = reader.read_u8()?;
                            let bool_val = val != 0;
                            //println!("Read Bool: {} ({})", bool_val, val);
                            GGUFValue::Bool(bool_val)
                        }
                        8 => {
                            let val = Self::read_string(reader)?;
                            //println!("Read String: {}", val);
                            GGUFValue::String(val)
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
        for _ in 0..count {
            let name = Self::read_string(reader)?;
            let n_dimensions = reader.read_u32::<LittleEndian>()? as usize;
            let mut dimensions = Vec::with_capacity(n_dimensions);
            for _ in 0..n_dimensions {
                dimensions.push(reader.read_u64::<LittleEndian>()?);
            }
            let data_type = GGUFDataType::from_u32(reader.read_u32::<LittleEndian>()?);
            let offset = reader.read_u64::<LittleEndian>()?;
            tensors.push(GGUTensorInfo {
                name,
                dimensions,
                data_type,
                offset,
            });
        }
        Ok(tensors)
    }

    pub fn get_tensor_data(&self, tensor: &GGUTensorInfo) -> &[u8] {
        let start = tensor.offset as usize;
        let end = start + self.calculate_tensor_size(tensor);
        &self.mmap[start..end]
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
            GGUFDataType::F64 => 8,
            GGUFDataType::F32 => 4,
            GGUFDataType::F16 | GGUFDataType::BF16 => 2,
            GGUFDataType::Q8_0 | GGUFDataType::Q8_1 | GGUFDataType::I8 => 1,
            GGUFDataType::Q4_0 | GGUFDataType::Q4_1 => 1, // 4 bits per element, but stored as bytes
            // Add more as needed
            _ => 4, // Default
        }
    }
}


#[cfg(test)]
mod tests {
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
                    // For the tokenizer, just print the count to avoid huge output
                    if let GGUFValue::Array(tokens) = value {
                        println!("  {}: Array with {} elements", key, tokens.len());
                    } else {
                        println!("  {}: {:?}", key, value);
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
}
