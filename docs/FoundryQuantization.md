# Foundry Quantization Architecture

> [!IMPORTANT]
> This document is the source of truth for Foundry's quantization system. It details how quantization is modeled, dispatched, and executed across the Rust backend and Metal kernels.

Foundry uses a centralized **Policy-Based Architecture** to handle mixed-precision inference. Instead of hardcoding checks for specific data types (like `is_q8`) scattered throughout the codebase, Foundry defines high-level **Policies** that encapsulate all logic required to load, bind, and execute kernels for a specific quantization scheme.

## Core Concepts

### 1. `QuantizationPolicy`
The `QuantizationPolicy` trait is the brain of the operation. It defines *how* a specific quantization format behaves. It is responsible for:
- **Dispatch**: Determining which compute kernel (Metal function) to use.
- **Loading**: Providing a `LoaderStage` to handle argument binding.
- **Configuration**: Defining block sizes, grid dimensions, and threadgroup layouts.

```rust
pub trait QuantizationPolicy: Send + Sync + 'static {
    // Returns the associated LoaderStage for argument binding
    fn loader_stage(&self) -> Arc<dyn LoaderStage>;

    // Returns the specific Metal kernel function name (e.g., "gemv_q8")
    fn kernel_name(&self) -> String;

    // Defines optimal threadgroup configuration
    fn threadgroup_size(&self) -> MTLSize;
}
```

### 2. `LoaderStage`
The `LoaderStage` trait handles the "plumbing" of tensors. It bridges the gap between high-level logical steps (like "Matrix Multiply") and low-level Metal buffers.
- **Binding**: Given a weight name (e.g., "layers.0.feed_forward.w1"), it resolves all necessary physical buffers (weights, scales, zeros) from the `SymbolTable`.
- **Validation**: Ensures all required tensors are present before execution.

```rust
pub trait LoaderStage: Send + Sync + Debug {
    /// Bind arguments using pre-resolved indices (Zero Allocation).
    fn bind(
        &self,
        fast_bindings: &FastBindings,
        resolved: &ResolvedSymbols
    ) -> SmallVec<[TensorArg; 4]>;
    
    fn quantization_type(&self) -> QuantizationType;
}
```

### 3. `QuantizationType`
An enum acting as a high-level identifier for dispatch logic.
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationType {
    F16,
    Q8_0,
    Q4_0, // Future extension
    // ...
}
```

## Dispatch & Execution Flow

When a `Step` (like `GemvCanonicalStep`) is compiled, the following happens:

1.  **Policy Resolution**: The system checks the `quantization` field of the step (or defaults to F16).
2.  **Kernel Selection**: `resolve_policy(quantization_type)` is called to retrieve the correct `QuantizationPolicy` implementation (e.g., `PolicyQ8`).
3.  **Symbol Resolution**: The `compile` method resolves all necessary symbol IDs into a `ResolvedSymbols` struct (e.g., weights index, scales index).
4.  **Binding**: At runtime, `loader.bind(fast_bindings, &resolved)` is called. This uses direct integer indexing (no strings) for zero-allocation binding.

## Kernel Integration (Metal)

Foundry uses C++ templates in Metal to write generic kernels that adapt to different quantization types.

### The generic `Policy` Struct
Kernels should access weights via a `Policy` template parameter.

```metal
// embedding.metal example

template <typename Policy> // Policy is injected at compile time
kernel void run_embedding_core(...) {
    // Load generic weight type (float, half, or quantized block)
    thread Policy::WeightType val;
    
    // Use Policy to load specific format
    Policy::load_weights(table, offset, val);
    
    // Use Policy to apply dequantization scale
    half scale = Policy::load_scale(scale_bytes, scale_idx);
    
    out[gid] = convert_half(val) * scale;
}
```

### Defining a New Policy in Metal
To add a new quantization backend in Metal, define a struct that matches the Policy interface:
```metal
struct Q8Policy {
    static constant uint BLOCK_SIZE = 32;
    typedef uchar4 WeightType; // 4x Q8 weights packed
    
    static void load_weights(...) { ... }
    static half load_scale(...) { ... }
};
```

## How to Add a New Quantization Type

Follow this checklist to implement a new quantization format (e.g., Q4_0).

### 1. Rust Implementation
1.  **Update Enum**: Add `Q4_0` to `QuantizationType` in `src/compound/stages.rs`.
2.  **Create Policy**: Create `src/foundry/policy/q4.rs`.
    - Implement `QuantizationPolicy` for `PolicyQ4`.
    - Implement `LoaderStage` directly on `PolicyQ4`.
    - In `bind()`, use `resolved.weights` (and scales) to get tensors from `FastBindings`.
3.  **Register Policy**: Update `resolve_policy` in `src/foundry/policy.rs` to return `PolicyQ4` for `QuantizationType::Q4_0`.

### 2. Metal Implementation
1.  **Create Metal Policy**: In your kernel file (or a shared header like `policies/q4.metal`), define `struct Q4Policy`.
    - It must conform to the interface expected by the kernel template (e.g., `load_weights`, `load_scale`).
2.  **No Explicit Instantiation**: You do **not** need to instantiate the kernel in the Metal file using `[[host_name]]`. The Rust `CompoundKernel` system will handle this.

### 3. Rust Stage Integration
1.  **Implement Stage**: In your `Stage` implementation (e.g., `GemvStage`), update the `emit` method.
2.  **Inject Policy**:
    ```rust
    fn emit(&self, ...) -> (String, String) {
        let policy_name = self.quant.policy_name(); // Returns "Q4Policy"
        (
            "output".into(),
            format!("generic_kernel<{policy_name}>(...);")
        )
    }
    ```
3.  **Ensure Includes**: Make sure `Stage::includes()` returns the path to your new Metal policy file.

### 4. Usage
You can now use this quantization in any Step by setting its `quantization` field.
```rust
let step = GemvCanonicalStep {
    // ...
    quantization: QuantizationType::Q4_0,
};
```

The system will automatically:
1.  Route to `PolicyQ4`.
2.  Use `PolicyQ4` to bind your packed Q4 weights.
3.  Dispatch the `gemv_q4` Metal kernel.

---

> [!TIP]
> **Debugging**: If you see "Symbol not found", 99% of the time you forgot to `get_or_create` a derived symbol (like scales) in your Step's `compile` method.

