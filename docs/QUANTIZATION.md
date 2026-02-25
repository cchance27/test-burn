# Foundry Quantization Architecture

> [!IMPORTANT]
> This document is the source of truth for Foundry's quantization system. It details how quantization is modeled, dispatched, and executed across the Rust backend and Metal kernels.

Foundry uses a centralized **Policy-Based Architecture** to handle mixed-precision inference. The goal is to separate **Kernel Logic** (math) from **Quantization Logic** (data access).

To achieve this, we use a Unified Policy model where a single Rust struct (e.g., `PolicyQ8`) implements two key traits: one for **Code Generation** (`MetalPolicy`) and one for **Runtime Loading / Binding** (`MetalPolicyRuntime`).

## Policy Variant Runtime State (Updated)

Foundry now has a centralized **policy variant resolver** (`crates/metallic-foundry/src/policy/variant.rs`) that describes, per source dtype:

- `source_dtype` (model tensor dtype)
- `storage_dtype` (materialized GPU tensor dtype)
- `compute_dtype` (target compute dtype metadata for future kernel selection)
- `lossy_cast` (explicit precision-loss signal)

This provides one policy entrypoint for runtime dtype behavior across dense + quant tensors.

Current behavior:

- Default variant is `preserve:f16`.
- Dense `F32` materializes as true `F32` via `PolicyF32Native` (`policy_f32.metal`) when preserve mode is selected.
- Dense `F32` can be capped to `F16` via dense variant selection (`dense:f16` or `dense=f16`).
- Quantized storage stays source-quantized; quant compute target metadata is represented by `QuantComputeVariant` (`F16`/`BF16`/`F32`).

Runtime selection:

- `METALLIC_POLICY_VARIANT` controls the active variant used by `resolve_policy(...)`.
- Supported values:
  - `dense:f16` or `dense=f16` (cap dense tensors to `F16`; quant tensors unchanged)
  - `preserve`
  - `preserve:f16` / `preserve:bf16` / `preserve:f32`
  - key/value form: `dense=preserve,quant=bf16`
- Invalid values fail fast during policy resolution.

Important:

- F16/F32 runtime dtype migration is complete for current kernels.
- Hot internal vector lanes use compile-time fast aliases (`FastScalarT`, `FastVec2T`, `FastVec4T`) so fast paths are not tied to raw `half` spellings.
- BF16 is intentionally deferred to a future sprint (`docs/FUTURE_BF16.md`).
- Fail-fast remains required for unsupported dtype/variant combinations.
- GEMM/GEMV/Flash and fused paths resolve storage/compute/accum via runtime dtype contracts and fail fast on unsupported combinations.

### Kernel dtype helper layer (runtime contract)

To avoid per-kernel dtype branching, Foundry now has a shared Metal helper include:

- `crates/metallic-foundry/src/metals/dtypes/runtime_types.metal`

It defines storage/compute/accum aliases, fast-lane aliases, and helper APIs:

- `InputStorageT`, `OutputStorageT`, `BiasStorageT`, `ResidualStorageT`, `GammaStorageT`
- `ComputeT`, `AccumT`
- `FastScalarT`, `FastVec2T`, `FastVec4T`
- `metallic_load_*`, `metallic_to_*`, `metallic_store_output`
- indexed vs contiguous output helper variants (`metallic_store_output{2,4}` and `*_contig`)

Kernels use this helper layer broadly (including FlashAttention, GEMV, QKV, SwiGLU, RoPE, softmax/materialized SDPA paths) so kernel code can stay policy/runtime-typed without per-kernel dtype branching.

## Authoritative Runtime DType Matrix

This table is the current source of truth for runtime dtype behavior.

| Area | Storage dtype(s) | Compute dtype | Accum dtype | Contract / fail-fast behavior |
|---|---|---|---|---|
| Runtime helper layer (`runtime_types.metal`) | Bound tensor storage (`F16/F32/BF16` for dense, `uchar` for quant storage) | Inferred from runtime tensor mix, optionally overridden by `METALLIC_COMPUTE_DTYPE` / `--compute-dtype` | `METALLIC_ACCUM_DTYPE` / `--accum-dtype` (default `F32`) | Invalid override value fails fast. If accum precision is narrower than compute, compute is clamped to accum and a warning is emitted. |
| FlashAttention dense kernels | Dense `F16`/`F32` only (uniform per op) | Runtime helper compute contract | Runtime helper accum contract | Non-dense or mixed dense dtypes are rejected. Invalid accum override fails fast. Non-`F32` accum is allowed with warning about possible stability loss. |
| FusedQkv (tuple policy) | Per-buffer tuple policy `(q,k,v)` (uniform and mixed-policy supported) | Runtime helper compute contract | Runtime helper accum contract | Tuple-aware fused variants are generated/dispatched via policy tuples. Unresolved/invalid policy contracts fail fast. |
| Fused SwiGLU / fused FFN SwiGLU RMSNorm (tuple policy) | Per-buffer tuple policy `(gate,up)` | Runtime helper compute contract | Runtime helper accum contract | Tuple-aware fused variants are generated/dispatched via policy tuples. Unresolved/invalid policy contracts fail fast. |
| Runtime configuration surface | N/A | CLI: `--compute-dtype`; API: `FoundryConfig::with_compute_dtype`; generic override: `--foundry-env METALLIC_COMPUTE_DTYPE=...` | CLI: `--accum-dtype`; API: `FoundryConfig::with_accum_dtype`; generic override: `--foundry-env METALLIC_ACCUM_DTYPE=...` | All override sources flow through `Foundry::new_with_config(...)` and `metallic_env` scoped overrides. |

## Core Concepts

### 1. The Unified Policy Struct

A specific quantization format (like Q8) is defined by a single Rust struct. This struct implements the logic for both compile-time metal generation and runtime data loading.

We use `#[derive(MetalPolicy)]` to minimize boilerplate for the kernel generation aspect.

```rust
// src/policy/q8.rs

#[derive(Clone, Debug, MetalPolicy)]
#[policy(header = "policies/policy_q8.metal", struct_name = "PolicyQ8")]
pub struct PolicyQ8;

impl MetalPolicyRuntime for PolicyQ8 {
    // ... Runtime loading logic (load_weights) + loader_stage binding ...
}
```

### 2. `MetalPolicy` Trait (Kernel Generation)

This trait (often auto-implemented via macro) provides the metadata needed to compile the Metal kernel.

*   **`header()`**: Path to the Metal implementation (e.g., `policies/policy_q8.metal`).
*   **`struct_name()`**: The C++ struct name to template the kernel with (e.g., `PolicyQ8`).
*   **`meta()`**: Returns a plain-data `PolicyMeta` snapshot (header/name/bytes-per-address-unit/blocks/etc.).

**Kernel Stages should consume this trait, not an enum.** This ensures kernels remain decoupled from the list of available quantizations.

```rust
pub struct VectorizedDotStage {
    policy: Arc<dyn MetalPolicy>, // Decoupled: Accepts ANY policy
}

impl Stage for VectorizedDotStage {
    fn emit(&self, ...) {
        let name = self.policy.struct_name(); // "PolicyQ8"
        // Emits: "run_kernel<PolicyQ8>(...)"
    }
}
```

### 3. `MetalPolicyRuntime` Trait (Runtime Loading)

This trait handles the runtime aspects: reading GGUF files, reshuffling bytes, and binding buffers.

*   **`load_weights(...)`**: Reads generic source model data and converts it to the format expected by the Metal policy (e.g., splitting Q8 into data + scale planes).
*   **`loader_stage()`**: Returns a `LoaderStage` that binds these processed buffers to the GPU.

#### Shared Helpers (Block Quants)

For block-quant formats (e.g. Q8_0, Q4_0 found in GGUF) there is shared loader/splitting infrastructure to avoid repeating:

- Source tensor dtype validation
- dims/layout validation (2D / canonical / Nk)
- block count math
- `{raw blocks} -> {weights plane, scales plane}` splitting
- optional canonical reorder

See `crates/metallic-foundry/src/policy/block_quant.rs`.

#### Centralized GGUF Quant Specs (Loader Crate)

GGUF block geometry constants are centralized in the loader crate:

- `crates/metallic-loader/src/quant_spec.rs`
- `crates/metallic-loader/src/gguf/quant_spec.rs`

This keeps source-of-truth `(weights_per_block, block_bytes)` definitions in one place and avoids policy-local duplication.
Foundry policies should import these constants instead of re-defining static byte math.

#### Centralized Dtype Parsing (No Per-Loader `contains(...)` Chains)

String-to-dtype parsing for model metadata now routes through `Dtype::parse_fuzzy(...)` in `crates/metallic-sdk/src/tensor.rs`.

This keeps alias handling (`q6k`, `Q6_K`, model-label strings, etc.) in one place and avoids adding new ad-hoc
`if s.contains("QX_Y")` branches in each loader.

#### Derive Macro For GGUF Runtime Wiring

For GGUF block quant policies, prefer deriving runtime plumbing with:

- `#[derive(GgufBlockQuantRuntime)]`
- `#[gguf_runtime(...)]`

This generates the repetitive `BlockQuantCodec + LoaderStage + MetalPolicyRuntime` boilerplate with typed `Dtype`/spec arguments (no stringly-typed dtype names).

### 4. `Quantization` Enum (Configuration)

The enum exists **only** as a serializable configuration key. It is used in Steps (e.g., `GemvStep`) to request a specific policy.

```rust
// Configuration only!
#[derive(Serialize, Deserialize)]
pub enum Quantization { F16, Q8, Q4 }
```

## Execution Flow

1.  **Config**: The `GemvStep` deserializes containing `quantization: Quantization::Q8`.
2.  **Resolution**: The step calls `resolve_policy(Quantization::Q8)` to get an `Arc<PolicyQ8>`.
3.  **Compilation**:
    *   The `PolicyQ8` (as `dyn MetalPolicy`) is passed to `VectorizedDotStage`.
    *   The stage generates Metal code referencing `#include "policy_q8.metal"`.
4.  **Loading**:
    *   The `PolicyQ8` (as `dyn MetalPolicyRuntime`) loads weights from the model file (via `LoadedModel`).
    *   It binds the weights to the pipeline.

## Critical Policy Contract (Packed Formats)

Some quant formats (e.g. `Q4_0`) are **packed**, meaning “logical element index” is **not** a constant number of bytes.

This has two consequences:

1. **Kernels must not do byte addressing with `policy.element_size()` for packed weights.**
   * `element_size()` is **not** “bytes per logical weight” for packed types.
   * For packed formats, `PolicyMeta.address_unit_bytes` (and `element_size()`) may be `0` to make misuse fail loudly.
2. **Kernel stages must treat weight offsets as *logical element indices*, and let the policy translate them.**
   * The Metal policy functions take `offset` as a **logical element index** into the conceptual `[row, k]` view (after `WEIGHT_INDEX(...)`), not a byte offset and not a pre-scaled pointer.

In other words, this pattern is correct for *all* policies (F16/Q8/Q4/etc.):

```cpp
ulong idx = WEIGHT_INDEX(row_idx, k, K, N);      // logical element index
Policy::template load_weights<8>(weights, idx);  // policy maps idx -> packed bytes
```

And this pattern is **not** safe for packed policies:

```cpp
// WRONG for packed formats like Q4_0:
const device uchar* w_ptr = weights + idx * policy.meta().address_unit_bytes;
Policy::template load_weights<8>(w_ptr, 0);
```

## Affine Quant Contract (e.g. Q4_1)

Some block quants are affine, with per-block decode:

`w = scale * q + affine`

To keep kernels generic and policy-owned:

- policy headers expose `HAS_AFFINE`, `SCALE_BYTES`, `load_affine(...)`
- callsites must not hardcode scale stride (e.g. `* 2`) and should use `Policy::SCALE_BYTES`
- kernels that previously used only `scale` must add the affine contribution through the policy hooks
- non-affine policies compile affine logic out via `METALLIC_POLICY_HAS_AFFINE` preprocessor gating
  (no extra affine loads/dot-sum work on Q4_0/Q8_0/Q6_K/F16 paths)

This keeps Q4_0/Q8_0/Q6_K and Q4_1 on the same kernel paths without per-model special-casing.

## Weights-Per-Block Is Policy-Owned

`weights_per_block` must come from the active policy metadata (`policy.meta().weights_per_block`) for quantized paths.
Do **not** hardcode `32` in stage/step execution code.

Practical rules:

- In Rust execution paths (GEMV/GEMM/fused paths), derive `weights_per_block` from policy metadata when `policy.has_scale() == true`.
- In Metal policy structs, expose `WEIGHTS_PER_BLOCK` and use it in kernels that compute scale/block indexing.
- Model JSON may still carry a `weights_per_block` field, but policy-derived values are authoritative for quantized weights.

This prevents format-specific regressions (e.g., `Q6_K` using a different logical block size than `Q8_0`/`Q4_0`).

## Mixed-Quant Fused Paths (Current Status)

Tuple-policy fused execution is now supported:

- `FusedQkv` dispatches by `(q,k,v)` policy tuple.
- `FusedSwiglu` / `FusedFfnSwiGluRmsNorm` dispatch by `(gate,up)` policy tuple.

Kernel cache keys include policy tuples so mixed-policy variants are cached independently from uniform fast paths.

Foundry remains fail-fast for invalid/unresolved policy contracts and unsupported dtype contracts.

## Scales Are Opaque Bytes

For block-quant formats, “scales” buffers are treated as **opaque bytes** (`device uchar*`) and interpreted by the policy (e.g., `load_scale(scales, block_idx)` reading fp16 bits). Stages should not assume a typed element layout for scales beyond the policy’s contract.

## Q6_K Bring-Up Notes (Gotchas)

`Q6_K` exposed a few easy-to-miss implementation traps that are relevant for future quant formats:

- **Do not assume block field order.**  
  GGUF/ggml `block_q6_K` is laid out as `ql[128] + qh[64] + scales[16] + d(f16)` (not `d + scales + q`).
  If this is decoded with the wrong offsets, inference may run but produce nonsense outputs.

- **Treat GGUF quant packing axis as `ne[0]` (K axis).**  
  Source block packing is along `ne[0]`, with rows over `ne[1]`.  
  For many row-major bindings this means source dims look like `[K, N]`, not `[N, K]`.

- **Validate from bytes first, then dims.**  
  For super-block formats, derive source block count from `raw.len() / BLOCK_BYTES`, then cross-check against expected rows/K.
  This catches bad axis assumptions early and gives clearer errors than only checking shape divisibility.

- **Preserve logical-index contract in kernels.**  
  Keep policy access in terms of logical `WEIGHT_INDEX(...)` offsets; policy code handles packed representation details.
  Avoid introducing byte-stride assumptions in shared GEMV/GEMM paths.

- **Symptom heuristic:** all `!`/gibberish output with no crash usually indicates decode/layout mismatch, not sampling.

### Q6_K Validation Checklist

When adding a new block-quant policy, validate in this order:

1. Decode one block against ggml reference (`dequantize_row_*`) with a focused unit test.
2. Verify block byte-size math matches GGUF tensor raw byte length.
3. Verify row/block indexing by running a known-good prompt on a small model and checking output sanity.

## How to Add a New Quantization Type

To add a new quantization type (e.g. `Q4_0`):

1.  **Rust**: Create `src/policy/q4.rs`.
    *   Define `pub struct PolicyQ4;`
    *   Add `#[derive(MetalPolicy)]` pointing to your metal file.
    *   Implement `MetalPolicyRuntime` to handle Q4-specific loading (packing/unpacking).
    *   If it is a block-quant format, prefer using the shared loader in `policy/block_quant.rs` with a small `write_block(...)` adapter.
2.  **Metal**: Create `src/metals/policies/policy_q4.metal`.
    *   Implement the C++ struct `PolicyQ4` with `load_weights` and `load_scale`.
3.  **Registry**: Add `Q4` to the `Quantization` enum and `resolve_policy` match statement.

You do **not** need to modify `GemvStage`, `VectorizedDotStage`, or any kernel code. They will automatically accept your new `PolicyQ4` via the trait object.

## Migration Note

*Legacy Note:* Some older kernels might hardcode enum matches or use `PolicyStage<T>` generics. The preferred pattern for the Core Engine (Gemv/Gemm) is the **Trait Object** approach described above (`Arc<dyn MetalPolicy>`), which offers the best balance of performance (via dynamic dispatch during compilation) and extensibility.
