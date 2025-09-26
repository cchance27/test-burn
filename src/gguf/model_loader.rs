use super::{GGUFDataType, GGUFError, GGUFFile};
use crate::{
    gguf::GGUFValue,
    metallic::{Context, Tensor, resource_cache::ResourceCache},
};
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

    /// Load a model from the GGUF file
    pub fn load_model(&self, _context: &Context) -> Result<GGUFModel, GGUFError> {
        // Create Metallic tensors from GGUF tensors
        let mut tensors = HashMap::new();
        let _cache = ResourceCache::new();

        for tensor_info in &self.gguf_file.tensors {
            // Special handling for F16 tensors: convert to F32
            if tensor_info.data_type == GGUFDataType::F16 {
                match self.gguf_file.get_tensor_data(tensor_info) {
                    Ok(raw) => {
                        // TODO: Why is this necessary??? shouldnt this be handled like we do for Q8... and others?
                        //println!("Converting F16 tensor '{}' with {} bytes", tensor_info.name, raw.len());
                        //if tensor_info.name == "token_embd.weight" {
                        //    println!("First F16 bytes: {:02x?} {:02x?}", raw[0], raw[1]);
                        //}
                        let elem_count = raw.len() / 2;
                        let mut f32_data = Vec::with_capacity(elem_count);
                        for i in 0..elem_count {
                            let lo = raw[2 * i] as u16;
                            let hi = raw[2 * i + 1] as u16;
                            let bits = lo | (hi << 8); // Little-endian: low byte first
                            let h = half::f16::from_bits(bits);
                            f32_data.push(h.to_f32());
                        }
                        let mut dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
                        // Special case for embedding: swap dims if it's [d_model, vocab] but data laid out as [vocab, d_model]
                        if tensor_info.name == "token_embd.weight" && dims.len() == 2 && dims[0] == 896 && dims[1] == 151936 {
                            dims = vec![151936, 896];
                            //println!("Swapped dims for token_embd.weight to [vocab, d_model]");
                        }
                        match crate::metallic::Tensor::create_tensor_from_slice(&f32_data, dims, _context) {
                            Ok(t) => {
                                tensors.insert(tensor_info.name.clone(), t);
                                continue;
                            }
                            Err(e) => {
                                println!(
                                    "Warning: F16 tensor created but failed to make Metallic tensor '{}': {:?}",
                                    tensor_info.name, e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        println!("Warning: failed to read raw F16 tensor bytes '{}': {:?}", tensor_info.name, e);
                    }
                }
            }

            match Tensor::try_from((&self.gguf_file, tensor_info)) {
                Ok(tensor) => {
                    tensors.insert(tensor_info.name.clone(), tensor);
                }
                Err(e) => {
                    println!(
                        "Warning: Failed to convert tensor '{}': {:?} (type={:?})",
                        tensor_info.name, e, tensor_info.data_type
                    );
                    // If direct conversion failed, try dequantization paths for known quant formats (e.g., Q8_0/Q8_1)
                    use crate::gguf::GGUFDataType;
                    match tensor_info.data_type {
                        GGUFDataType::Q8_0 | GGUFDataType::Q8_1 => {
                            // Attempt to get raw bytes and dequantize using gguf::quant helpers
                            match self.gguf_file.get_tensor_data(tensor_info) {
                                Ok(raw) => {
                                    // Use the q8 dequant path (SIMD on aarch64)
                                    #[cfg(target_arch = "aarch64")]
                                    let deq_res = crate::gguf::quant::dequantize_q8_to_f32_simd(raw, tensor_info.data_type);
                                    #[cfg(not(target_arch = "aarch64"))]
                                    let deq_res = crate::gguf::quant::dequantize_q8_to_f32(raw, tensor_info.data_type);
                                    match deq_res {
                                        Ok(f32_data) => {
                                            let dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
                                            match crate::metallic::Tensor::create_tensor_from_slice(&f32_data, dims, _context) {
                                                Ok(t) => {
                                                    tensors.insert(tensor_info.name.clone(), t);
                                                }
                                                Err(e) => {
                                                    println!(
                                                        "Warning: dequantized tensor created but failed to make Metallic tensor '{}': {:?}",
                                                        tensor_info.name, e
                                                    );
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            println!("Warning: failed to dequantize tensor '{}': {:?}", tensor_info.name, e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("Warning: failed to read raw tensor bytes '{}': {:?}", tensor_info.name, e);
                                }
                            }
                        }
                        GGUFDataType::F16 => {
                            // Try to get raw bytes and convert F16 to F32
                            match self.gguf_file.get_tensor_data(tensor_info) {
                                Ok(raw) => {
                                    println!("Converting F16 tensor '{}' with {} bytes", tensor_info.name, raw.len());
                                    // Manual F16 conversion
                                    let elem_count = raw.len() / 2;
                                    let mut f32_data = Vec::with_capacity(elem_count);
                                    for i in 0..std::cmp::min(10, elem_count) {
                                        let lo = raw[2 * i] as u16;
                                        let hi = raw[2 * i + 1] as u16;
                                        let bits = lo | (hi << 8);
                                        let h = half::f16::from_bits(bits);
                                        let f32_val = h.to_f32();
                                        println!("  F16[{}] bits=0x{:04x} -> f32={}", i, bits, f32_val);
                                        f32_data.push(f32_val);
                                    }
                                    // Add the rest of the data
                                    for i in 10..elem_count {
                                        let lo = raw[2 * i] as u16;
                                        let hi = raw[2 * i + 1] as u16;
                                        let bits = lo | (hi << 8);
                                        let h = half::f16::from_bits(bits);
                                        f32_data.push(h.to_f32());
                                    }
                                    let mut dims: Vec<usize> = tensor_info.dimensions.iter().map(|&d| d as usize).collect();
                                    // Special case for embedding: swap dims if it's [d_model, vocab] but data laid out as [vocab, d_model]
                                    if tensor_info.name == "token_embd.weight" && dims.len() == 2 && dims[0] == 896 && dims[1] == 151936 {
                                        dims = vec![151936, 896];
                                        //println!("Swapped dims for token_embd.weight to [vocab, d_model]");
                                    }
                                    match crate::metallic::Tensor::create_tensor_from_slice(&f32_data, dims, _context) {
                                        Ok(t) => {
                                            println!("Successfully converted F16 tensor '{}'", tensor_info.name);
                                            // Check some values
                                            let slice = t.as_slice();
                                            println!(
                                                "First 10 values of '{}': {:?}",
                                                tensor_info.name,
                                                &slice[..std::cmp::min(10, slice.len())]
                                            );
                                            if tensor_info.name == "output.weight" {
                                                println!(
                                                    "First 10 values of output.weight: {:?}",
                                                    &slice[0..std::cmp::min(10, slice.len())]
                                                );
                                            }
                                            tensors.insert(tensor_info.name.clone(), t);
                                        }
                                        Err(e) => {
                                            println!(
                                                "Warning: F16 tensor created but failed to make Metallic tensor '{}': {:?}",
                                                tensor_info.name, e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("Warning: failed to read raw F16 tensor bytes '{}': {:?}", tensor_info.name, e);
                                }
                            }
                        }
                        _ => {
                            // For other types, log and skip for now.
                            println!(
                                "Warning: Could not convert tensor '{}', skipping (type={:?})",
                                tensor_info.name, tensor_info.data_type
                            );
                        }
                    }
                }
            }
        }

        Ok(GGUFModel {
            tensors,
            metadata: self.gguf_file.metadata.clone(),
        })
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
