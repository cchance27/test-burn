use super::{GGUFError, GGUFFile};
use crate::metallic::{Context, Tensor, resource_cache::ResourceCache};
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
                    tensors.insert(tensor_info.name.clone(), tensor);
                }
                Err(e) => {
                    // For quantized tensors, we'll store them as raw data for now
                    // In the future, we'll implement dequantization
                    println!(
                        "Warning: Could not convert tensor '{}': {}",
                        tensor_info.name, e
                    );
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

impl GGUFModel {
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