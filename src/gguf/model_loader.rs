use super::{GGUFError, GGUFFile};
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
            match Tensor::try_from((&self.gguf_file, tensor_info)) {
                Ok(tensor) => {
                    //println!("Successfully converted tensor '{}' using normal path (type={:?})", tensor_info.name, tensor_info.data_type);
                    //if tensor_info.name == "token_embd.weight" {
                    //    let slice = tensor.as_slice();
                    //    println!("First 10 values of token_embd.weight: {:?}", &slice[..std::cmp::min(10, slice.len())]);
                    //}
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
                                    let deq_res = crate::gguf::quant::dequantize_q8_to_f32_simd(
                                        raw,
                                        tensor_info.data_type,
                                    );
                                    #[cfg(not(target_arch = "aarch64"))]
                                    let deq_res = crate::gguf::quant::dequantize_q8_to_f32(
                                        raw,
                                        tensor_info.data_type,
                                    );
                                    match deq_res {
                                        Ok(f32_data) => {
                                            let dims: Vec<usize> = tensor_info
                                                .dimensions
                                                .iter()
                                                .map(|&d| d as usize)
                                                .collect();
                                            match crate::metallic::Tensor::create_tensor_from_slice(
                                                &f32_data, dims, _context,
                                            ) {
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
                                            println!(
                                                "Warning: failed to dequantize tensor '{}': {:?}",
                                                tensor_info.name, e
                                            );
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!(
                                        "Warning: failed to read raw tensor bytes '{}': {:?}",
                                        tensor_info.name, e
                                    );
                                }
                            }
                        }
                        GGUFDataType::F16 => {
                            // Try to get raw bytes and convert F16 to F32
                            match self.gguf_file.get_tensor_data(tensor_info) {
                                Ok(raw) => {
                                    println!(
                                        "Converting F16 tensor '{}' with {} bytes",
                                        tensor_info.name,
                                        raw.len()
                                    );
                                    // Manual F16 conversion
                                    let elem_count = raw.len() / 2;
                                    let mut f32_data = Vec::with_capacity(elem_count);
                                    for i in 0..std::cmp::min(10, elem_count) {
                                        let lo = raw[2 * i] as u16;
                                        let hi = raw[2 * i + 1] as u16;
                                        let bits = lo | (hi << 8);
                                        let h = half::f16::from_bits(bits);
                                        let f32_val = h.to_f32();
                                        println!(
                                            "  F16[{}] bits=0x{:04x} -> f32={}",
                                            i, bits, f32_val
                                        );
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
                                    let dims: Vec<usize> = tensor_info
                                        .dimensions
                                        .iter()
                                        .map(|&d| d as usize)
                                        .collect();
                                    match crate::metallic::Tensor::create_tensor_from_slice(
                                        &f32_data, dims, _context,
                                    ) {
                                        Ok(t) => {
                                            println!(
                                                "Successfully converted F16 tensor '{}'",
                                                tensor_info.name
                                            );
                                            // Check some values
                                            let slice = t.as_slice();
                                            println!(
                                                "First 10 values of '{}': {:?}",
                                                tensor_info.name,
                                                &slice[..std::cmp::min(10, slice.len())]
                                            );
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
                                    println!(
                                        "Warning: failed to read raw F16 tensor bytes '{}': {:?}",
                                        tensor_info.name, e
                                    );
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
        if let Some(super::GGUFValue::String(arch)) =
            self.metadata.entries.get("general.architecture")
        {
            Some(arch)
        } else {
            None
        }
    }

    /// Get context length from metadata
    pub fn get_context_length(&self) -> Option<u64> {
        if let Some(super::GGUFValue::U32(len)) = self.metadata.entries.get("qwen2.context_length")
        {
            Some(*len as u64)
        } else {
            None
        }
    }

    /// Instantiate a concrete Metallic model that implements `LoadableModel`.
    /// This allows callers to do:
    ///   let gguf_model = GGUFModelLoader::new(...).load_model(...)?
    ///   let wan: Qwen25 = gguf_model.instantiate(&mut ctx)?;
    pub fn instantiate<T: crate::metallic::model::LoadableModel>(
        &self,
        ctx: &mut crate::metallic::Context,
    ) -> Result<T, super::GGUFError> {
        // Delegate to the metallic::model::Model::load helper. Map MetalError -> GGUFError::InvalidData with context.
        match crate::metallic::model::Model::load::<T>(self, ctx) {
            Ok(v) => Ok(v),
            Err(_e) => Err(super::GGUFError::InvalidData),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GGUFFile;

    #[test]
    fn test_model_loader() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
        match GGUFFile::load(path) {
            Ok(gguf_file) => {
                let loader = GGUFModelLoader::new(gguf_file);

                // Initialize Metallic context
                let context = match crate::metallic::Context::new() {
                    Ok(ctx) => ctx,
                    Err(e) => {
                        println!("Failed to create Metallic context: {:?}", e);
                        return;
                    }
                };

                match loader.load_model(&context) {
                    Ok(model) => {
                        println!(
                            "Successfully loaded model with {} tensors",
                            model.tensors.len()
                        );

                        // Check model metadata
                        if let Some(arch) = model.get_architecture() {
                            println!("Model architecture: {}", arch);
                        }

                        if let Some(context_len) = model.get_context_length() {
                            println!("Context length: {}", context_len);
                        }

                        // Try to get a specific tensor
                        if let Some(tensor) = model.get_tensor("token_embd.weight") {
                            println!(
                                "Found token embedding tensor with dimensions: {:?}",
                                tensor.dims
                            );
                        } else {
                            println!("Token embedding tensor not found");
                        }
                    }
                    Err(e) => {
                        println!("Failed to load model: {:?}", e);
                    }
                }
            }
            Err(e) => {
                println!("Failed to load GGUF file: {:?}", e);
            }
        }
    }
}

#[cfg(test)]
mod integration {
    use crate::gguf::GGUFFile;
    use crate::gguf::model_loader::GGUFModelLoader;
    use crate::metallic::Context;

    // Integration test: load the Qwen2.5 GGUF and run a single forward through qwen25.
    // Marked #[ignore] because it loads a large model and allocates device memory.
    #[test]
    #[ignore]
    fn integration_load_qwen2_and_forward() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";

        // Load GGUF file (memory-mapped)
        let gguf_file = match GGUFFile::load(path) {
            Ok(g) => g,
            Err(e) => panic!("Failed to load GGUF file '{}': {:?}", path, e),
        };

        // Initialize Metallic context
        let mut ctx = match Context::new() {
            Ok(c) => c,
            Err(e) => panic!("Failed to create Metallic Context: {:?}", e),
        };

        // Build GGUFModel using the loader
        let loader = GGUFModelLoader::new(gguf_file);
        let gguf_model = match loader.load_model(&ctx) {
            Ok(m) => m,
            Err(e) => panic!("Failed to create GGUFModel: {:?}", e),
        };

        // Instantiate Qwen25 from GGUFModel (uses the LoadableModel impl)
        let wan: crate::metallic::qwen25::Qwen25 = gguf_model
            .instantiate(&mut ctx)
            .expect("Failed to instantiate Qwen25 from GGUFModel");

        // Create a small input (batch=1, seq=min(4, config.seq_len), d_model matches)
        let batch = 1usize;
        let seq = std::cmp::min(4usize, wan.config.seq_len);
        let d = wan.config.d_model;
        let input_data = vec![0.0f32; batch * seq * d];
        let input = crate::metallic::Tensor::create_tensor_from_slice(
            &input_data,
            vec![batch, seq, d],
            &ctx,
        )
        .expect("Failed to create input tensor");

        // Run a single forward through the model (first block path)
        let _out = wan
            .forward(&input, &mut ctx)
            .expect("qwen25 forward failed");

        // If we reach here without panicking, the path completed successfully.
    }
}
