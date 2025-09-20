//! Comprehensive validation tests for GGUF implementation
use crate::gguf::GGUFFile;
use crate::metallic;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_file_validation() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";

        // Test loading the file
        let gguf = match GGUFFile::load(path) {
            Ok(gguf) => gguf,
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        };

        // Validate header
        assert_eq!(&gguf.header.magic, b"GGUF");
        assert_eq!(gguf.header.version, 3);
        assert_eq!(gguf.header.tensor_count, 290);
        assert_eq!(gguf.header.metadata_count, 27);

        // Validate metadata
        assert_eq!(gguf.metadata.entries.len(), 27);

        // Check specific metadata entries
        if let Some(arch) = gguf.metadata.entries.get("general.architecture") {
            match arch {
                crate::gguf::GGUFValue::String(s) => assert_eq!(s, "qwen2"),
                _ => panic!("Expected string value for general.architecture"),
            }
        } else {
            panic!("Missing general.architecture metadata");
        }

        // Validate tensor count
        assert_eq!(gguf.tensors.len(), 290);

        // Check first tensor
        let first_tensor = &gguf.tensors[0];
        assert_eq!(first_tensor.name, "token_embd.weight");
        assert_eq!(first_tensor.data_type, crate::gguf::GGUFDataType::Q8_1);
        assert_eq!(first_tensor.dimensions, vec![896, 151936]);

        // Test tensor data access
        match gguf.get_tensor_data(first_tensor) {
            Ok(data) => {
                // For Q8_1, we expect 153151488 bytes
                assert_eq!(data.len(), 153151488);
            }
            Err(e) => {
                panic!("Failed to get tensor data: {:?}", e);
            }
        }

        println!("GGUF file validation passed!");
    }

    #[test]
    fn test_metallic_tensor_conversion() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";

        let gguf = match GGUFFile::load(path) {
            Ok(gguf) => gguf,
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        };

        // Initialize Metallic context
        let _context = match metallic::Context::new() {
            Ok(ctx) => ctx,
            Err(e) => {
                panic!("Failed to create Metallic context: {:?}", e);
            }
        };

        // Test with the first tensor (Q8_1)
        let first_tensor = &gguf.tensors[0];
        assert_eq!(first_tensor.name, "token_embd.weight");

        match metallic::Tensor::try_from((&gguf, first_tensor)) {
            Ok(metallic_tensor) => {
                // Validate tensor properties
                assert_eq!(metallic_tensor.dims, vec![896, 151936]);
                assert_eq!(metallic_tensor.len(), 136134656); // 896 * 151936

                // Check that we can access the data
                let slice = metallic_tensor.as_slice();
                assert_eq!(slice.len(), 136134656);

                // Check first few values (these should be reasonable F32 values)
                for &val in slice.iter().take(100) {
                    // Values should be finite numbers
                    assert!(val.is_finite());
                }

                println!("Successfully converted GGUF tensor to Metallic tensor!");
                println!("Tensor dimensions: {:?}", metallic_tensor.dims);
                println!("Tensor size: {} elements", metallic_tensor.len());
            }
            Err(e) => {
                panic!("Error converting tensor: {}", e);
            }
        }
    }

    #[test]
    fn test_memory_management() {
        let path = "/Volumes/2TB/test-burn/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf";

        let gguf = match GGUFFile::load(path) {
            Ok(gguf) => gguf,
            Err(e) => {
                panic!("Failed to load GGUF file: {:?}", e);
            }
        };

        // Test with first few tensors
        let tensors_to_test: Vec<&crate::gguf::GGUTensorInfo> =
            gguf.tensors.iter().take(3).collect();

        for tensor in &tensors_to_test {
            // Test offloading
            match gguf.offload_tensor(tensor) {
                Ok(_) => {
                    println!("Successfully offloaded tensor: {}", tensor.name);
                }
                Err(e) => {
                    panic!("Error offloading tensor {}: {}", tensor.name, e);
                }
            }

            // Test loading
            match gguf.load_tensor(tensor) {
                Ok(data) => {
                    println!(
                        "Successfully loaded tensor: {} ({} bytes)",
                        tensor.name,
                        data.len()
                    );
                    // Verify we got some data
                    assert!(!data.is_empty());
                }
                Err(e) => {
                    panic!("Error loading tensor {}: {}", tensor.name, e);
                }
            }
        }

        // Test blockswapping with first two tensors
        if tensors_to_test.len() >= 2 {
            let load_tensors = vec![tensors_to_test[1]];
            let unload_tensors = vec![tensors_to_test[0]];

            match gguf.blockswap_tensors(&load_tensors, &unload_tensors) {
                Ok(_) => {
                    println!("Successfully blockswapped tensors");
                }
                Err(e) => {
                    panic!("Error blockswapping tensors: {}", e);
                }
            }
        }

        println!("Memory management tests passed!");
    }

    #[test]
    fn test_error_handling() {
        // Test loading non-existent file
        let result = GGUFFile::load("/non/existent/file.gguf");
        assert!(result.is_err());
        
        println!("Error handling tests passed!");
    }
}
