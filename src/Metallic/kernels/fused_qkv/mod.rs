//! Metal kernels for fused QKV projection post-processing.

// The compute pipeline is consumed directly from `Context::fused_qkv_projection`.
// This module exists to keep the directory structure consistent with other kernels.

#[cfg(test)]
mod tests {
    use super::super::*;

    #[test]
    fn kernel_symbol_exists() {
        // Ensure the Metal source is included during compilation by touching the library enum.
        let name = KernelFunction::FusedQkvBiasSplit.name();
        assert_eq!(name, "fused_qkv_bias_split");
    }
}
