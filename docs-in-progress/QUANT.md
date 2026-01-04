# Quantization (Foundry + Metal) — Adding a New Quant

This doc is the “single place” overview for adding a new quantization format to Metallic’s Foundry + Metal kernel stack with minimal churn.

**Design goals**
- **Centralize** quant logic into one Rust policy/hook and one Metal policy file (or folder).
- **Avoid touching templates/kernels** when adding a new quant.
- Keep **runtime performance** at least on-par with legacy (no “nice DX” regressions at runtime).

## Terminology

- **Policy (MSL)**: A Metal header that defines how a kernel loads/decodes weights for a specific quant (e.g. F16, Q8, Q4…).
- **Policy (Rust)**: A Rust type that references a policy header and emits parameter binding code (e.g. `#[derive(MetalPolicy)]`).
- **SIMD GEMV Policy (MSL)**: A policy that implements the `run_simd_gemv_template` contract used by the decode-oriented SIMD GEMV template.
- **SIMD GEMV Hook (Rust)**: A tiny Rust type (via `#[derive(GemvHook)]`) that selects the SIMD GEMV policy + emits the `Params` initializer.

## The Driver-Strategy-Quant Model (System Architecture)

The system now follows a **Driver-Strategy-Quant** separation of concerns:

1.  **Driver (`GemvStage`)**: The main kernel logic driver. It handles memory wiring, threadgroup management, and high-level execution flow. It is generic over a `Strategy` and `QuantPolicy`.
2.  **Strategy (`GemvStrategy`)**: Defines the *execution loop* structure (e.g., `Canonical` for standard 1-token decode, or eventual `Block` strategies). It "drives" the loop index logic.
3.  **Quant (`QuantPolicy`)**: Defines the *data access* behavior (e.g., `F16`, `Q8`). It handles pointer arithmetic, data loading, and type conversion.
    - **`AutoQuant`**: An extension trait for `QuantPolicy` that enables runtime validity checks (`valid_for_dtype`), powering dynamic dispatch.

This composition allows us to reuse the exact same `Canonical` loop logic for `F16`, `Q8`, and future quants without duplicating the loop unrolling/barrier code.

## Where quant code lives

### 1) Generic policies (used by many kernels)
- Metal: `crates/metallic/src/metals/policies/policy_*.metal`
- Rust: types deriving `MetalPolicy` that reference the Metal header.

### 2) SIMD GEMV (Driver-Strategy-Quant)
This path is *very* perf-sensitive and has a dedicated template:
- Driver: `crates/metallic/src/metals/matmul_gemv/simd_common.metal`
- Strategies (Rust): `crates/metallic/src/metals/matmul_gemv/strategy.rs` (Implementation of `Canonical<Q>`)
- Policies (Metal):
  - `crates/metallic/src/metals/policies/simd_gemv_f16_canonical.metal` (Generic over Quant)
  - `crates/metallic/src/metals/policies/simd_gemv_q8_canonical.metal`
- Hooks (Rust): `crates/metallic/src/metals/matmul_gemv/hooks.rs`

The SIMD GEMV template is intentionally policy-agnostic; quant-specific code must live in the policy headers.

## Adding a new quant (checklist)

### Step 1 — Add the Metal policy for your quant

1. Create a new file:
   - `crates/metallic/src/metals/policies/policy_q4.metal` (or whatever quant name you use).
2. Keep it self-contained (header-style) and follow the conventions used by `policy_f16.metal` / `policy_q8.metal`.

If your quant format needs global helpers shared across kernels, put them in the policy header, not in each kernel.

### Step 2 — Add the Rust policy wrapper (generic kernels)

Add a Rust type using `#[derive(MetalPolicy)]` that points to the policy header you created.

This keeps kernel source generation clean and avoids scattering “quant implementation details” across kernel code.

### Step 3 — (If needed) Add SIMD GEMV support for your quant

If the model uses SIMD GEMV (decode-oriented GEMV), add:

1. A SIMD GEMV policy header:
   - `crates/metallic/src/metals/policies/simd_gemv_q4_canonical.metal`
2. A hook in Rust:
   - `crates/metallic/src/metals/matmul_gemv/hooks.rs`

**Important contract: `data_arr` is bytes**

`GemvStage` always declares the local weight-pointer array as:
- `const device uchar* data_arr[HEADS]`

Your hook/policy must cast as needed:
- F16 policy casts to `const device half**`
- Q8 policy uses `const device uchar**` (+ `scale_arr`)

This is what allows new quants to plug in without changing the SIMD GEMV template or config derivations.

### Step 4 — Update the model wiring (spec/steps)

Your new quant must be reachable from the model execution plan:
- Foundry specs select concrete ops/steps (e.g. `QkvF16CanonicalFusedRmsnorm` in `crates/metallic/src/foundry/spec/qwen25.json`).
- If you add a new quant-specific fused path, add a corresponding step and hook/policy selection.

## Common pitfalls

- **Spreading quant code across kernels**: avoid. Put quant logic in `policies/` and select via policy/hook.
- **Naive fusion of normalization**: a standalone RMSNorm stage cannot be safely fused ahead of GEMV without a global sync. SIMD GEMV “fused RMSNorm” must compute `inv_rms` inside each threadgroup (see the RMSNorm preamble used by SIMD GEMV hooks).
- **Inconsistent pointer types**: for SIMD GEMV, always assume `data_arr` is `uchar*[]` and cast in hook/policy.

## References

- Macros: `docs-in-progress/MACROS.md`
- Kernel composition: `docs-in-progress/KERNELS.md`
