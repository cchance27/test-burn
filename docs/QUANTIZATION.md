# Foundry Quantization Architecture

> [!IMPORTANT]
> This document is the source of truth for Foundry's quantization system. It details how quantization is modeled, dispatched, and executed across the Rust backend and Metal kernels.

Foundry uses a centralized **Policy-Based Architecture** to handle mixed-precision inference. The goal is to separate **Kernel Logic** (math) from **Quantization Logic** (data access).

To achieve this, we use a Unified Policy model where a single Rust struct (e.g., `PolicyQ8`) implements two key traits: one for **Code Generation** (`MetalPolicy`) and one for **Runtime Loading** (`QuantizationPolicy`).

## Core Concepts

### 1. The Unified Policy Struct

A specific quantization format (like Q8) is defined by a single Rust struct. This struct implements the logic for both compile-time metal generation and runtime data loading.

We use `#[derive(MetalPolicy)]` to minimize boilerplate for the kernel generation aspect.

```rust
// src/policy/q8.rs

#[derive(Clone, Debug, MetalPolicy)]
#[policy(header = "policies/policy_q8.metal", struct_name = "PolicyQ8")]
pub struct PolicyQ8;

impl QuantizationPolicy for PolicyQ8 {
    // ... Runtime loading logic (load_weights, optimization hints) ...
}
```

### 2. `MetalPolicy` Trait (Kernel Generation)

This trait (often auto-implemented via macro) provides the metadata needed to compile the Metal kernel.

*   **`header()`**: Path to the Metal implementation (e.g., `policies/policy_q8.metal`).
*   **`struct_name()`**: The C++ struct name to template the kernel with (e.g., `PolicyQ8`).

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

### 3. `QuantizationPolicy` Trait (Runtime Loading)

This trait handles the runtime aspects: reading GGUF files, reshuffling bytes, and binding buffers.

*   **`load_weights(...)`**: Reads generic GGUF data and converts it to the format expected by the Metal policy (e.g., splitting Q8 into data + scale planes).
*   **`loader_stage()`**: Returns a `LoaderStage` that binds these processed buffers to the GPU.

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
    *   The `PolicyQ8` (as `dyn QuantizationPolicy`) loads weights from disk.
    *   It binds the weights to the pipeline.

## How to Add a New Quantization Type

To add `Q4`:

1.  **Rust**: Create `src/policy/q4.rs`.
    *   Define `pub struct PolicyQ4;`
    *   Add `#[derive(MetalPolicy)]` pointing to your metal file.
    *   Implement `QuantizationPolicy` to handle Q4 specific loading (packing/unpacking).
2.  **Metal**: Create `src/metals/policies/policy_q4.metal`.
    *   Implement the C++ struct `PolicyQ4` with `load_weights` and `load_scale`.
3.  **Registry**: Add `Q4` to the `Quantization` enum and `resolve_policy` match statement.

You do **not** need to modify `GemvStage`, `VectorizedDotStage`, or any kernel code. They will automatically accept your new `PolicyQ4` via the trait object.

## Migration Note

*Legacy Note:* Some older kernels might hardcode enum matches or use `PolicyStage<T>` generics. The preferred pattern for the Core Engine (Gemv/Gemm) is the **Trait Object** approach described above (`Arc<dyn MetalPolicy>`), which offers the best balance of performance (via dynamic dispatch during compilation) and extensibility.
