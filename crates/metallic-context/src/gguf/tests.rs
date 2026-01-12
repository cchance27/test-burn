#![cfg(test)]
use crate::{
    F32Element, gguf::{GGUFFile, GGUFValue, tensor_info::GGUTensorInfo}
};

#[test]
fn test_load_gguf_file() {
    let path = "/Volumes/2TB/test-burn/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load_mmap_and_get_metadata(path) {
        Ok(gguf) => {
            println!("Successfully loaded GGUF file:");
            println!("Magic: {:?}", std::str::from_utf8(&gguf.header.magic).unwrap_or("Invalid UTF-8"));
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

            let mut count = 0;
            for tensor in &gguf.tensor_metadata {
                if count < 10 {
                    println!("  {}: {:?} ({:?})", tensor.name, tensor.dimensions, tensor.data_type);
                    count += 1;
                } else {
                    break;
                }
            }
            println!("... and {} more tensors", gguf.tensor_metadata.len().saturating_sub(10));
        }
        Err(e) => {
            panic!("Failed to load GGUF file: {:?}", e);
        }
    }
}

#[test]
fn test_gguf_to_metallic_tensor() {
    let path = "/Volumes/2TB/test-burn/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load_mmap_and_get_metadata(path) {
        Ok(gguf) => {
            // Initialize Metallic context
            let _context = match crate::Context::<F32Element>::new() {
                Ok(ctx) => ctx,
                Err(e) => {
                    panic!("Failed to create Metallic context: {:?}", e);
                }
            };

            // Test with the first tensor (should be Q8_1)
            if let Some(first_tensor) = gguf.tensor_metadata.first() {
                println!("Testing tensor: {} ({:?})", first_tensor.name, first_tensor.data_type);
                println!("Dimensions: {:?}", first_tensor.dimensions);

                // Calculate expected element count
                let expected_elements: usize = first_tensor.dimensions.iter().map(|&d| d as usize).product();
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

                match crate::Tensor::<F32Element>::try_from((&gguf, first_tensor)) {
                    Ok(metallic_tensor) => {
                        println!("Successfully converted tensor '{}' to Metallic tensor", first_tensor.name);
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
    let path = "/Volumes/2TB/test-burn/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load_mmap_and_get_metadata(path) {
        Ok(gguf) => {
            // Debug the first tensor
            if let Some(_first_tensor) = gguf.tensor_metadata.first() {}

            // Debug a few more tensors
            for (i, _tensor) in gguf.tensor_metadata.iter().enumerate() {
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
    let path = "/Volumes/2TB/test-burn/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load_mmap_and_get_metadata(path) {
        Ok(gguf) => {
            // Initialize Metallic context
            let _context = match crate::Context::<F32Element>::new() {
                Ok(ctx) => ctx,
                Err(e) => {
                    panic!("Failed to create Metallic context: {:?}", e);
                }
            };

            // Test with the first tensor (should be Q8_1)
            if let Some(first_tensor) = gguf.tensor_metadata.first() {
                println!("Testing tensor: {} ({:?})", first_tensor.name, first_tensor.data_type);
                println!("Dimensions: {:?}", first_tensor.dimensions);
                match crate::Tensor::<F32Element>::try_from((&gguf, first_tensor)) {
                    Ok(metallic_tensor) => {
                        println!("Successfully converted tensor '{}' to Metallic tensor", first_tensor.name);
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

#[test]
fn test_memory_management() {
    let path = "/Volumes/2TB/test-burn/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";
    match GGUFFile::load_mmap_and_get_metadata(path) {
        Ok(gguf) => {
            // Test with the first few tensors
            let tensors_to_test: Vec<&GGUTensorInfo> = gguf.tensor_metadata.iter().take(3).collect();

            for tensor in &tensors_to_test {
                println!("Testing memory management for tensor: {}", tensor.name);

                // Test offloading
                match gguf.offload_tensor(tensor) {
                    Ok(_) => println!("Successfully offloaded tensor: {}", tensor.name),
                    Err(e) => println!("Error offloading tensor {}: {}", tensor.name, e),
                }

                // Test loading
                match gguf.load_tensor(tensor) {
                    Ok(data) => println!("Successfully loaded tensor: {} ({} bytes)", tensor.name, data.len()),
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
