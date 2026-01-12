use crate::fusion::{Epilogue, MetalPolicy};

/// Standard F16 loading policy.
#[derive(Default, Clone, Copy, Debug)]
pub struct PolicyF16;

impl MetalPolicy for PolicyF16 {
    fn header(&self) -> &'static str {
        "policies/policy_f16.metal"
    }

    fn struct_name(&self) -> &'static str {
        "PolicyF16"
    }

    fn init_params_code(&self) -> &'static str {
        "pp.matrix = (const device half*)matrix;"
    }

    fn buffer_types(&self) -> &'static [(&'static str, &'static str)] {
        &[("matrix", "const device uchar*"), ("scale_bytes", "const device uchar*")]
    }

    fn define_loader(&self) -> Option<String> {
        Some("struct PolicyF16; using Policy = PolicyF16;".to_string())
    }

    fn load_and_dequant(&self, data_ptr: &str, _scale_ptr: &str, offset: &str) -> (&'static str, String) {
        let code = format!(
            r#"half4 h_vec = *(const device half4*)({data_ptr} + {offset});
    float4 w_vec = float4(h_vec);"#
        );
        ("w_vec", code)
    }

    fn cols_per_threadgroup(&self) -> usize {
        4 // 4 warps for strided layout
    }

    fn threadgroup_width(&self) -> usize {
        128 // 4 warps * 32 lanes
    }
}

/// Q8 Block-wise quantization policy.
#[derive(Default, Clone, Copy, Debug)]
pub struct PolicyQ8;

impl MetalPolicy for PolicyQ8 {
    fn header(&self) -> &'static str {
        "policies/policy_q8.metal"
    }

    fn struct_name(&self) -> &'static str {
        "PolicyQ8"
    }

    fn buffer_types(&self) -> &'static [(&'static str, &'static str)] {
        &[("matrix", "const device uchar*"), ("scale_bytes", "const device uchar*")]
    }

    fn define_loader(&self) -> Option<String> {
        Some("struct PolicyQ8; using Policy = PolicyQ8;".to_string())
    }

    fn load_and_dequant(&self, _data_ptr: &str, _scale_ptr: &str, _offset: &str) -> (&'static str, String) {
        ("w_vec", "// data loaded via macro".to_string())
    }

    fn cols_per_threadgroup(&self) -> usize {
        8
    }

    fn threadgroup_width(&self) -> usize {
        256 // 8 warps * 32 lanes
    }
}

/// No-op epilogue (passthrough).
#[derive(Default, Clone, Copy, Debug)]
pub struct EpilogueNone;

impl Epilogue for EpilogueNone {
    fn header(&self) -> &'static str {
        "policies/epilogue_none.metal"
    }

    fn struct_name(&self) -> &'static str {
        "EpilogueNone"
    }
}
