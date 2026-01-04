use metallic_macros::MetalStruct;

// Parameter struct matching the Metal side
#[derive(Clone, Copy, Debug, MetalStruct, serde::Serialize, serde::Deserialize, Default)]
#[repr(C)]
pub struct GemvParams {
    #[metal(name = "K")]
    #[serde(default)]
    pub k: u32,
    #[metal(name = "N")]
    #[serde(default)]
    pub n: u32,
    #[serde(default)]
    pub blocks_per_k: u32,
    #[serde(default)]
    pub weights_per_block: u32,
    #[serde(default)]
    pub batch: u32,
    #[serde(default)]
    pub stride_x: u32,
    #[serde(default)]
    pub stride_y: u32,
    #[serde(default)]
    pub stride_a: u32,
    #[serde(default)]
    pub stride_w: u32,
    #[serde(default)]
    pub stride_scale: u32,
}

// Layout-specific kernels
pub mod row_major;
pub use row_major::GemvRowMajor;

pub mod col_major;
pub use col_major::GemvColMajor;

pub mod canonical;
pub use canonical::GemvCanonical;
// Re-export GemvRowMajor as GemvDense for backwards compatibility
pub use row_major::GemvRowMajor as GemvDense;
pub mod step;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemv_params_metal_struct_def() {
        let def = GemvParams::METAL_STRUCT_DEF;
        assert!(def.contains("uint K;"), "Should have K field: {}", def);
        assert!(def.contains("uint N;"), "Should have N field: {}", def);
        assert!(def.contains("uint blocks_per_k;"), "Should have blocks_per_k: {}", def);
        assert!(def.starts_with("struct GemvParams {"), "Should start with struct: {}", def);
        assert!(def.ends_with("};"), "Should end with brace: {}", def);
    }
}
