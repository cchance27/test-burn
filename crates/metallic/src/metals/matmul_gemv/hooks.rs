use metallic_macros::{GemvHook, GemvPrologue};

// =============================================================================
// Prologues — Pre-main setup stages for SIMD GEMV
// =============================================================================

/// RMSNorm prologue — computes `inv_rms` once per threadgroup.
///
/// This produces the `inv_rms` variable used by `*Rmsnorm` policy hooks.
///
/// # Usage
/// ```ignore
/// #[derive(GemvKernel)]
/// #[gemv_kernel(
///     prologue = RmsnormPrologue,
///     hook = F16CanonicalRmsnormHook,
///     // ...
/// )]
/// pub struct MyFusedKernel;
/// ```
#[derive(GemvPrologue, Clone, Copy, Default)]
#[gemv_prologue(emit = r#"    threadgroup float inv_rms_s;
    const uint lane_id = lid.x & 31u;
    const uint warp_id = lid.x / 32u;
    const float inv_rms = gemv_compute_inv_rms(vector_x, params->K, lane_id, warp_id, &inv_rms_s, epsilon);
"#)]
pub struct RmsnormPrologue;

// =============================================================================
// Hooks — Policy selection and parameter initialization
// =============================================================================

/// F16 canonical SIMD GEMV policy.
///
/// Works with both `NoPrologue` (no normalization) and `RmsnormPrologue` (applies RMSNorm).
/// The prologue provides `gamma` and `inv_rms` variables.
#[derive(GemvHook, Clone, Copy, Default)]
#[gemv_hook(
    id = "f16_canonical",
    policy_struct = "CanonicalStrategy<QuantF16>",
    includes("strategies/simd_gemv_canonical.metal", "policies/quant_types.metal"),
    policy_params = r#"    CanonicalStrategy<QuantF16>::Params p = {
        (const device half**)data_arr,
        gamma,
        inv_rms,
        params->weights_per_block
    };
"#
)]
pub struct F16CanonicalHook;

/// Q8 canonical SIMD GEMV policy.
///
/// Works with both `NoPrologue` (no normalization) and `RmsnormPrologue` (applies RMSNorm).
#[derive(GemvHook, Clone, Copy, Default)]
#[gemv_hook(
    id = "q8_canonical",
    policy_struct = "CanonicalStrategy<QuantQ8>",
    includes("strategies/simd_gemv_canonical.metal", "policies/quant_types.metal"),
    policy_params = r#"    CanonicalStrategy<QuantQ8>::Params p = {
        (const device uchar**)data_arr,
        (const device uchar**)scale_arr,
        gamma,
        inv_rms,
        params->weights_per_block
    };
"#
)]
pub struct Q8CanonicalHook;
