//! Q8 quantization support for GGUF format
use crate::gguf::{GGUFDataType, GGUFError};
use half::f16;

/// Dequantize Q8_0/Q8_1 tensor data to F32
pub fn dequantize_q8_to_f32(data: &[u8], data_type: GGUFDataType) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Q8_0 format:
    // - Each block contains 32 weights
    // - Each block has:
    //   * 2 bytes: half-precision (f16) scale factor
    //   * 32 bytes: 8-bit quantized weights
    //
    // Q8_1 format:
    // - Similar to Q8_0 but with additional delta value
    // - Each block has:
    //   * 2 bytes: half-precision (f16) scale factor
    //   * 2 bytes: half-precision (f16) delta value
    //   * 32 bytes: 8-bit quantized weights

    let (block_size, scale_offset, delta_offset, weight_offset) = match data_type {
        GGUFDataType::Q8_0 => (34, 0, None, 2),    // 2 (scale) + 32 (weights) = 34 bytes per block
        GGUFDataType::Q8_1 => (36, 0, Some(2), 4), // 2 (scale) + 2 (delta) + 32 (weights) = 36 bytes per block
        _ => return Err("Invalid data type for Q8 dequantization".into()),
    };

    // Calculate number of blocks
    let num_blocks = data.len() / block_size;

    let weights_per_block = 32;
    let total_weights = num_blocks * weights_per_block;
    let mut f32_data = Vec::with_capacity(total_weights);

    // Pre-allocate space to avoid repeated allocations
    f32_data.resize(total_weights, 0.0);

    // Process blocks in a more efficient way
    for block_idx in 0..num_blocks {
        let block_start = block_idx * block_size;
        let block_data = &data[block_start..block_start + block_size];

        // Extract scale (2 bytes as f16)
        let scale_bytes = [block_data[scale_offset], block_data[scale_offset + 1]];
        let scale_f16 = f16::from_bits(u16::from_le_bytes(scale_bytes));
        let scale = scale_f16.to_f32();

        // For Q8_1, extract delta (2 bytes as f16)
        let delta = if let Some(delta_offset) = delta_offset {
            let delta_bytes = [block_data[delta_offset], block_data[delta_offset + 1]];
            let delta_f16 = f16::from_bits(u16::from_le_bytes(delta_bytes));
            delta_f16.to_f32()
        } else {
            0.0
        };
        // Extract quantized weights
        let weight_data = &block_data[weight_offset..weight_offset + weights_per_block];

        // Calculate output offset for this block
        let output_offset = block_idx * weights_per_block;

        // Dequantize weights using loop unrolling for better performance
        // Process 4 weights at a time to reduce loop overhead
        let mut i = 0;
        while i + 3 < weights_per_block {
            let w0 = weight_data[i] as i8;
            let w1 = weight_data[i + 1] as i8;
            let w2 = weight_data[i + 2] as i8;
            let w3 = weight_data[i + 3] as i8;

            f32_data[output_offset + i] = (w0 as f32) * scale + delta;
            f32_data[output_offset + i + 1] = (w1 as f32) * scale + delta;
            f32_data[output_offset + i + 2] = (w2 as f32) * scale + delta;
            f32_data[output_offset + i + 3] = (w3 as f32) * scale + delta;

            i += 4;
        }

        // Handle remaining weights
        while i < weights_per_block {
            let weight_i8 = weight_data[i] as i8;
            let f32_value = (weight_i8 as f32) * scale + delta;
            f32_data[output_offset + i] = f32_value;
            i += 1;
        }
    }

    Ok(f32_data)
}

/// Temporary function to debug Q8 format
#[allow(dead_code)]
pub fn debug_q8_format(tensor_info: &crate::gguf::GGUTensorInfo, data: &[u8]) -> Result<(), GGUFError> {
    println!("Tensor: {}", tensor_info.name);
    println!("Data type: {:?}", tensor_info.data_type);
    println!("Data length: {}", data.len());
    println!("Dimensions: {:?}", tensor_info.dimensions);

    // Calculate expected size
    let element_count: usize = tensor_info.dimensions.iter().map(|&d| d as usize).product();
    println!("Element count: {}", element_count);

    match tensor_info.data_type {
        GGUFDataType::Q8_0 => {
            // Q8_0: 34 bytes per block (2 scale + 32 weights)
            let block_size = 34;
            let blocks = data.len() / block_size;
            let remainder = data.len() % block_size;
            println!("Q8_0 - Block size: {}, Blocks: {}, Remainder: {}", block_size, blocks, remainder);
            println!("Expected total weights: {}", blocks * 32);
        }
        GGUFDataType::Q8_1 => {
            // Q8_1: 36 bytes per block (2 scale + 2 delta + 32 weights)
            let block_size = 36;
            let blocks = data.len() / block_size;
            let remainder = data.len() % block_size;
            println!("Q8_1 - Block size: {}, Blocks: {}, Remainder: {}", block_size, blocks, remainder);
            println!("Expected total weights: {}", blocks * 32);
        }
        _ => {}
    }

    // Print first few bytes
    let print_len = std::cmp::min(68, data.len());
    println!("First {} bytes: {:?}", print_len, &data[..print_len]);
    Ok(())
}
