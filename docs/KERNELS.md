# Foundry Kernels (2025-12) — Composition, Fusion, and SIMD GEMV

This document describes how Foundry kernels are structured, composed, and fused, with special focus on the SIMD GEMV decode path used for QKV + FFN projections.

## Kernel types

### Standalone kernels
- A single `[[kernel]]` entry point in Metal.
- A Rust `Kernel` impl that binds args and dispatches.
- Use when fusion is not required or the kernel is already “one logical unit”.

### Compound kernels (preferred for fusion)
- A `CompoundKernel` is assembled from `Stage` pieces in Rust.
- Stages contribute:
  - `includes()` (Metal headers)
  - `buffer_args()` (Metal argument list)
  - `emit()` (the Metal snippet injected into the final function)
- This enables **text-level fusion** with **Rust-typed composition** and avoids hand-maintaining “giant fused .metal files”.

## Performance rules (decode path)

Some operations cannot be safely fused as separate stages because they would require a **global synchronization** between threadgroups. The primary example is:

- `RMSNorm stage -> GEMV stage` (not safe to “just fuse” as two independent stages)

Instead, decode GEMV fusions compute normalization inside each GEMV threadgroup and apply it while loading/using `vector_x`.

## SIMD GEMV (decode GEMV template)

### Why SIMD GEMV exists
Decode-time GEMV (seq_len=1) is one of the hottest paths in inference. The SIMD GEMV template is a performance-tuned implementation that follows the **Driver-Strategy-Quant** architecture (see `docs-in-progress/QUANT.md`). We want to keep it “DRY” and composable.

### How it is structured

**Template/common code**
- `src/metals/matmul_gemv/simd_common.metal`
  - Defines `run_simd_gemv_template<Policy, HEADS, COLS_PER_TG, FAST_PATH, Epilogue?>`
  - Contains shared structs (`GemvParams`, `QkvFusedParams`, `Q2FusedParams`) and helpers (e.g. `gemv_compute_inv_rms`).

**Quant/policy code**
- `src/metals/policies/simd_gemv_*_canonical.metal`
  - Implements the policy contract required by the template (F16/Q8 today).
  - New quants should add a new policy header here (do not edit the template).

**Hook glue (Rust)**
- `src/metals/matmul_gemv/hooks.rs`
  - `#[derive(GemvHook)]` types select:
    - policy struct name
    - policy include header(s)
    - optional preamble (e.g. fused RMSNorm `inv_rms` computation)
    - policy params initializer snippet

**Config glue (Rust)**
- `#[derive(GemvConfig)]` on a zero-sized struct describes:
  - number of heads (e.g. 3 for QKV)
  - pointer names for weights / outputs / biases
  - N expressions per head
  - optional scale pointers (for Q8-style scales)

**Epilogues**
- `GemvEpilogue` is the SIMD GEMV “tail” (e.g. SwiGLU).
- `#[derive(Epilogue)]` can also implement `GemvEpilogue` via `gemv_struct`/`gemv_id` to avoid duplication.

**Unified GemvKernel (NEW — Preferred)**
- `#[derive(GemvKernel)]` combines Config + Hook + Epilogue into one derive:
  ```rust
  #[derive(GemvKernel)]
  #[gemv_kernel(
      args = "SwiGluF16CanonicalFusedRmsnormArgs",
      heads = 2, cols_per_tg = 8,
      data_ptrs("data_g", "data_u"), result_ptrs("out_res", "nullptr"),
      // ... more config ...
      hook = F16CanonicalRmsnormHook,
      epilogue = SwiGluEpilogue,
  )]
  pub struct SwiGluFused;
  
  // Usage:
  let stage = SwiGluFused::main_stage();
  ```
- See `docs-in-progress/MACROS.md` Section 10 for full attribute reference.

### Key DX rule: weight pointers are bytes

The SIMD GEMV stage standardizes the local weight pointer array to bytes:
- `const device uchar* data_arr[HEADS]`

Hooks/policies cast as needed (`half**`, packed formats, etc.). This keeps quant additions centralized to policy + hook code and prevents “edit every kernel/template” churn.

## Where to look for an example

The canonical example is the “Foundry fused” decode GEMVs:
- `src/metals/matmul_gemv/fused/swiglu.rs`
- `src/metals/matmul_gemv/fused/qkv.rs`

These show:
- How to build a `CompiledCompoundKernel` and cache it in a `OnceLock`
- How to apply fused RMSNorm inside GEMV (safe + fast)
- How to keep quant code in policy headers + hooks
- **NEW**: How to use `#[derive(GemvKernel)]` for cleaner DX

## Adding a new quant

See `docs-in-progress/QUANT.md`.
