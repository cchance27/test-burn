//! SIMD-optimized Q8 quantization support for GGUF format
use crate::gguf::GGUFDataType;
use half::f16;

// SIMD imports for AArch64 (macOS with Metal)
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Dequantize Q8_0/Q8_1 tensor data to F32 using SIMD optimization
#[cfg(target_arch = "aarch64")]
pub fn dequantize_q8_to_f32_simd(
    data: &[u8],
    data_type: GGUFDataType,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let (block_size, scale_offset, delta_offset, weight_offset) = match data_type {
        GGUFDataType::Q8_0 => (34, 0, None, 2), // 2 (scale) + 32 (weights) = 34 bytes per block
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

    // Process blocks using NEON SIMD instructions
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

        // Use NEON SIMD to process 16 weights at a time (2 operations of 8 elements each)
        let mut i = 0;
        while i + 15 < weights_per_block {
            unsafe {
                // Load 16 i8 values and convert to i16 first
                let weights_i8x16 = vld1q_s8(weight_data.as_ptr().add(i) as *const i8);

                // Split into two 8-element vectors
                let weights_i8x8_low = vget_low_s8(weights_i8x16);
                let weights_i8x8_high = vget_high_s8(weights_i8x16);

                // Convert i8 to i16 (2 operations)
                let weights_i16x8_low = vmovl_s8(weights_i8x8_low);
                let weights_i16x8_high = vmovl_s8(weights_i8x8_high);

                // Convert i16 to f32 (4 operations)
                let weights_f32x4_low_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(weights_i16x8_low)));
                let weights_f32x4_low_1 =
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(weights_i16x8_low)));
                let weights_f32x4_high_0 =
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(weights_i16x8_high)));
                let weights_f32x4_high_1 =
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(weights_i16x8_high)));

                // Create scale and delta vectors
                let scale_vec = vdupq_n_f32(scale);
                let delta_vec = vdupq_n_f32(delta);

                // Perform SIMD computation: (weights * scale) + delta (4 operations)
                let result_f32x4_low_0 = vmlaq_f32(delta_vec, weights_f32x4_low_0, scale_vec);
                let result_f32x4_low_1 = vmlaq_f32(delta_vec, weights_f32x4_low_1, scale_vec);
                let result_f32x4_high_0 = vmlaq_f32(delta_vec, weights_f32x4_high_0, scale_vec);
                let result_f32x4_high_1 = vmlaq_f32(delta_vec, weights_f32x4_high_1, scale_vec);

                // Store results
                vst1q_f32(
                    f32_data.as_mut_ptr().add(output_offset + i),
                    result_f32x4_low_0,
                );
                vst1q_f32(
                    f32_data.as_mut_ptr().add(output_offset + i + 4),
                    result_f32x4_low_1,
                );
                vst1q_f32(
                    f32_data.as_mut_ptr().add(output_offset + i + 8),
                    result_f32x4_high_0,
                );
                vst1q_f32(
                    f32_data.as_mut_ptr().add(output_offset + i + 12),
                    result_f32x4_high_1,
                );
            }
            i += 16;
        }

        // Handle remaining weights (16 or fewer)
        while i < weights_per_block {
            let weight_i8 = weight_data[i] as i8;
            let f32_value = (weight_i8 as f32) * scale + delta;
            f32_data[output_offset + i] = f32_value;
            i += 1;
        }
    }

    Ok(f32_data)
}

/// Fallback implementation for non-AArch64 architectures
#[cfg(not(target_arch = "aarch64"))]
pub fn dequantize_q8_to_f32_simd(
    data: &[u8],
    data_type: GGUFDataType,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Fall back to the regular implementation
    super::q8::dequantize_q8_to_f32(data, data_type)
}
