# Foundry Kernels Architecture

Foundry is the kernel backend for Metallic, designed to support **high-performance, low-latency inference** on Apple Silicon. It provides a modular system for writing, composing, and fusing Metal kernels.

## Core Concepts

The kernel system is built around three primary abstractions:

1.  **Standalone Kernels**: Traditional, single-function Metal kernels.
2.  **Compound Kernels**: Fused kernels composed of multiple "Stages" (Prologue → Main → Epilogue) that are stitched together at runtime or compile-time.
3.  **Conditional Kernels**: Logic for dispatching to different kernel variants based on runtime conditions (e.g., batch size, sequence length).

---

## 1. Kernel Types

### 1.1 Standalone Kernels

Use standalone kernels for simple operations that do not require fusion or complex configuration.

**Rust Definition:**
```rust
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "ops/simple.metal",
    function = "simple_op",
    args = "SimpleParams" // Injects SimpleParams C++ struct definition
)]
pub struct SimpleKernel {
    pub input: TensorArg,
    #[arg(output)]
    pub output: TensorArg,
    pub params: SimpleParams,
}
```

**Metal Side (`ops/simple.metal`):**
> **Note:** Do NOT define `struct SimpleParams` here. It is automatically injected by the runtime based on the Rust struct.

```metal
[[kernel]] void simple_op(
    const device half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant SimpleParams *params [[buffer(2)]], // Struct defined by injection
    // Implicit arguments automatically available:
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    // ...
}
```

### 1.2 Compound Kernels (Fusion)

Compound kernels are the preferred way to write compute-intensive operations. They allow you to write small, reusable "Stages" in Metal and fuse them together in Rust.

**Architecture:**
*   **Prologue:** Handles data loading and dequantization (e.g., `PolicyStage`).
*   **Main:** The core computation.
*   **Epilogue:** Post-processing.

**Example:**
```rust
let kernel = CompoundKernel::new("fused_q8_gemv_silu")
    .prologue(PolicyStage::<PolicyQ8>::new()) // Load Q8 weights
    .main(GemvCoreStage::new())               // Compute Dot Product
    .epilogue(EpilogueStage::<SiLU>::new())   // Apply SiLU
    .build();
```

---

## 2. Developing Compound Stages

A `Stage` is a Rust struct that wraps a Metal helper function. You can use `#[derive(Kernel)]` to auto-generate the Stage implementation.

### 2.1 Rust Implementation

```rust
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "ops/my_op.metal",
    function = "my_op_entry",        // Entry point for standalone usage
    stage_function = "run_my_op",    // Template function for fusion
    args = "MyParams",
    threadgroup = "float shared_mem[256]" // Declares shared memory for this stage
)]
pub struct MyOp {
    // Input/Scales are handled by PolicyStage, so we skip them here for the stage signature
    #[arg(stage_skip)] 
    pub input: TensorArg,
    
    // Output is managed by this stage
    #[arg(output)]
    pub output: TensorArg,
    
    pub params: MyParams,
}
```

### 2.2 Metal Template Contract

When using `stage_function`, your Metal function must follow a strict signature contract to be compatible with the auto-generated code.

**Signature:**
```metal
template<typename Policy>
ALWAYS_INLINE void run_my_op(
    const device uchar *matrix,       // Provided by PolicyStage
    device half *output,              // Buffer 2 (Your Arg)
    constant MyParams *params,        // Buffer 3 (Your Arg)
    const device uchar *scales,       // Provided by PolicyStage
    uint3 gid,                        // Implicit
    uint3 lid,                        // Implicit
    threadgroup float *shared_mem     // Passed from entry point
) {
    // ... logic ...
}
```

**Key Rules:**
1.  **Implicit Arguments:** `matrix`, `scales`, `gid`, `lid` are passed by name from the caller.
2.  **Threadgroup Memory:** Must be declared in the `threadgroup` attribute in Rust. It is allocated in the `[[kernel]]` entry point and passed as a pointer to your template. **Never** declare `threadgroup` variables inside the template function itself.
3.  **Struct Injection:** Again, do not define `MyParams` in the Metal file.

---

## 3. The Policy System (Quantization)

Foundry abstracts data types (F16, Q8, Q4) using **Policies**. A Policy defines how data is loaded from memory and presented to the kernel.

**Rust Convention:**
*   **Buffer 0**: `matrix` (Input/Weights) - Managed by `PolicyStage`.
*   **Buffer 1**: `scales` (Quantization Metadata) - Managed by `PolicyStage`.
*   **Buffer 2+**: Kernel-specific arguments.

---

## 4. SIMD GEMV System (Decode Path)

For the ultra-critical decode path (Batch Size = 1), Foundry uses a specialized, highly fused GEMV system.

### 4.1 Unified `GemvKernel` Macro (Preferred for New Kernels)

The `#[derive(GemvKernel)]` macro generates a fully fused kernel configuration.

```rust
#[derive(GemvKernel)]
#[gemv_kernel(
    args = "SwiGluParams",
    heads = 2,
    cols_per_tg = 8,
    // Configuration
    gemv_n0 = "params->N",
    data_ptrs("data_gate", "data_up"),
    result_ptrs("out_gate", "nullptr"),
    // Components
    hook = F16CanonicalRmsnormHook, // F16 Weights + Fused RMSNorm
    epilogue = SwiGluEpilogue,      // SwiGLU Activation
)]
pub struct SwiGluFused;
```

### 4.2 Manual Stage Implementation (Advanced/Legacy)

Some complex kernels, such as **FusedQKV** (`qkv_stages.rs`), currently use manual `Stage` implementations instead of the `GemvKernel` macro. This is necessary when:
*   Binding complex argument sets that exceed standard patterns (e.g., Q, K, V weights + scales = 6 buffers).
*   Implementing custom emit logic that doesn't fit the standard template.

**Manual Implementation Pattern:**
1.  Implement the `Stage` trait manually.
2.  Define `buffer_args()` to return the exact list of buffers (remembering to skip 0/1 if using PolicyStage).
3.  Implement `emit()` to generate the specific C++ call site for your helper function.

```rust
// Example from qkv_stages.rs
impl Stage for ParallelProjectStage {
    fn buffer_args(&self) -> Vec<BufferArg> {
        // Manually define 6 buffers for Q/K/V weights & scales
        vec![
            BufferArg { name: "w_q", metal_type: "const device uchar*", buffer_index: 0 },
            // ...
        ]
    }

    fn emit(&self, input_var: &str) -> (String, String) {
        // manually construct the C++ call
        let code = format!("... {policy}::template dot<{vec_width}>(...); ...");
        ("qkv_partial".to_string(), code)
    }
}
```

---

## 5. Macro Reference

### `#[derive(MetalStruct)]`
Generates a C++ struct definition string `METAL_STRUCT_DEF` that matches the Rust struct layout.
*   **Usage:** Apply to param structs.
*   **Note:** Handles `DynamicValue<T>` fields automatically for compilation.

### `#[derive(Kernel)]`
Implements `Kernel` and optionally `Stage` traits.
*   `source`: Path to `.metal` file.
*   `function`: Name of `[[kernel]]` function.
*   `stage_function`: Name of template function (enables Stage generation).
*   `args`: Name of the params struct (injects its definition).
*   `threadgroup`: Declaration string for shared memory (e.g., `"float s[256]"`).
*   `dispatch`: Enables default dispatch config or presets (e.g., `dispatch="per_row"`).

### `#[derive(ConditionalKernel)]`
Creates a dispatcher enum.
*   `selector`: Variables used for dispatch (e.g., `"batch: u32"`).
*   `#[when(condition)]`: Variants selected by condition.

---

## 6. Development Checklist

- **Structs:** Defined in Rust with `#[derive(MetalStruct)]`. **Not** defined in `.metal`.
- **Threadgroup:** Declared in Rust via `threadgroup` attribute. Passed as pointer in Metal.
- **Buffers:** `PolicyStage` owns buffers 0 and 1. Your args start at 2 (unless standalone).
- **Implict Args:** `gid`, `lid`, `tptg` are available in Metal signatures.
