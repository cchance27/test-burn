# New Kernel Development Guide

> A comprehensive guide for implementing new Metal kernels in the Foundry system, covering standalone kernels, compound kernel stages, and policy integration.

## Quick Reference

| Component | Location | Purpose |
|-----------|----------|---------|
| Metal kernel | `src/metals/<name>.metal` | GPU computation code |
| Rust wrapper | `src/metals/<name>.rs` | `Kernel` trait impl |
| Stage (optional) | `src/compound/stages/<name>.rs` | Compound kernel fusion |
| Tests | `tests/<name>.rs` | Parity + correctness tests |

---

## 1. File Structure

```
crates/metallic/src/
├── metals/
│   ├── <kernel>.rs       # Rust Kernel implementation
│   ├── <kernel>.metal    # Metal shader source
│   └── mod.rs            # Add: pub mod <kernel>;
├── compound/stages/
│   ├── <kernel>.rs       # Stage for compound kernels (optional)
│   └── mod.rs            # Add: pub mod <kernel>; pub use <kernel>::*;
└── tests/
    └── <kernel>.rs       # Comprehensive test suite
```

---

## 2. Metal Kernel Implementation

### 2.1 Basic Template

```metal
// NOTE: ALWAYS_INLINE is provided by policies/base.metal (included automatically)
#include <metal_stdlib>
using namespace metal;

// Struct is injected by Foundry via struct_defs() - DO NOT define here!
// struct MyParams { ... };

// Core computation template (policy-aware)
template<typename Policy>
void run_my_kernel_core(
    const device uchar *input,
    device half *output,
    constant MyParams *params,
    const device uchar *scale_bytes,
    uint3 gid,
    uint3 lid,
    threadgroup float *shared_mem  // Passed from kernel entry point
) {
    // Implementation using Policy::load_weights<N>() and Policy::load_scale()
}

// Entry point - standalone only (guarded)
#ifndef FUSED_KERNEL
#ifndef POLICY_F16_DEFINED
#define POLICY_F16_DEFINED
struct PolicyF16 { /* ... */ };
#endif

[[kernel]] void my_kernel_f16(
    const device uchar *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant MyParams *params [[buffer(2)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    // Threadgroup memory MUST be declared in kernel, not template!
    threadgroup float shared_mem;
    run_my_kernel_core<PolicyF16>(input, output, params, input, gid, lid, &shared_mem);
}
#endif
```

### 2.2 Critical Rules

> [!CAUTION]
> **Threadgroup Memory**: Cannot declare `threadgroup` variables inside template functions. Declare in the `[[kernel]]` entry point and pass as pointer.

> [!CAUTION]
> **Struct Definitions**: Do NOT define param structs in the Metal file. Foundry injects them via `struct_defs()`. Duplicate definitions cause compilation errors.

> [!WARNING]
> **PolicyF16 Guard**: Always wrap inline `PolicyF16` with `#ifndef POLICY_F16_DEFINED` to prevent redefinition when policy headers are auto-included.

---

## 3. Rust Kernel Implementation

### 3.1 Required Derives

```rust
use metallic_macros::{MetalStruct, KernelArgs};

/// Parameter struct - generates Metal struct definition
#[derive(MetalStruct, Clone, Copy, Debug)]
#[repr(C)]
pub struct MyParams {
    pub dim: u32,
    pub total: u32,
}

/// Kernel struct - generates buffer binding code
/// Note: All fields are buffer args by default (indices auto-assigned 0, 1, 2...)
#[derive(KernelArgs, Clone)]
pub struct MyKernel {
    pub input: TensorArg,
    #[arg(output)]
    pub output: TensorArg,
    pub params: MyParams,
}
```

### 3.2 Kernel Trait Implementation

```rust
/// Kernel ID for pipeline caching
pub struct MyKernelId;

impl Kernel for MyKernel {
    type Args = MyParams;
    type Id = MyKernelId;

    fn source(&self) -> KernelSource { KernelSource::File("my_kernel.metal") }

    fn function_name(&self) -> &str { "my_kernel_f16" }

    fn includes(&self) -> Includes { Includes(vec![]) }

    fn dtype(&self) -> Option<Dtype> {
        Some(Dtype::F16)  // Auto-includes policies/policy_f16.metal
    }

    fn struct_defs(&self) -> String {
        MyParams::METAL_STRUCT_DEF.to_string()
    }

    fn bind(&self, encoder: &ComputeCommandEncoder) {
        self.bind_args(encoder);  // From KernelArgs derive
    }

    fn dispatch_config(&self) -> DispatchConfig {
        let num_rows = self.params.total / self.params.dim;
        DispatchConfig {
            grid: GridSize::d1(num_rows as usize),
            group: ThreadgroupSize::d1(256),
        }
    }

    fn as_stage(&self) -> Box<dyn Stage> {
        Box::new(MyCoreStage::new())
    }
}
```

---

## 4. Compound Kernel Stage

### 4.1 Buffer Layout Convention

> [!IMPORTANT]
> **PolicyStage reserves buffers 0 and 1:**
> - `buffer(0)`: `matrix` (input data as `uchar*`)
> - `buffer(1)`: `scale_bytes` (quantization scales)
>
> Your stage should start at buffer 2 and NOT declare `matrix` or `scale_bytes`.

```rust
impl Stage for MyCoreStage {
    fn buffer_args(&self) -> Vec<BufferArg> {
        // DO NOT include matrix(0) or scale_bytes(1) - PolicyStage provides them
        vec![
            BufferArg { name: "output", metal_type: "device half*", buffer_index: 2 },
            BufferArg { name: "params", metal_type: "constant MyParams*", buffer_index: 3 },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        // Declare threadgroup in emit code, pass to template
        let code = r#"
    threadgroup float shared_mem;
    run_my_kernel_core<Policy>(matrix, output, params, scale_bytes, gid, lid, &shared_mem);"#
            .to_string();
        ("void".to_string(), code)
    }
}
```

### 4.2 Common Gotchas

| Issue | Symptom | Fix |
|-------|---------|-----|
| Duplicate `scale_bytes` | "redefinition of parameter" | Don't declare in Stage - PolicyStage provides it |
| Duplicate `matrix` | "cannot reserve buffer location" | Use PolicyStage naming, don't duplicate at buffer 0 |
| Duplicate `PolicyF16` | "redefinition of 'PolicyF16'" | Add `#ifndef POLICY_F16_DEFINED` guard |
| Threadgroup in template | "cannot declare in non-qualified function" | Declare in kernel, pass as pointer |

---

## 5. Policy-Aware Kernels

### 5.1 Using Policy for Data Loading

```metal
template<typename Policy>
void run_kernel_core(...) {
    float vals[8];
    Policy::template load_weights<8>(input, offset, vals);  // Load + dequant
    half scale = Policy::load_scale(scale_bytes, block_idx); // For Q8
    
    // vals[] now contains dequantized floats
}
```

### 5.2 Dispatch with Policy

```rust
// F16 (standard)
foundry.run(&kernel)?;

// Q8 (with policy)
foundry.run_with_policy::<PolicyQ8, _>(&kernel)?;
```

---

## 6. Parallel Reduction Pattern

Learned from legacy fused GEMV+RMSNorm kernels:

```metal
// 1. Declare threadgroup in KERNEL (not template)
[[kernel]] void my_kernel(...) {
    threadgroup float tg_result;
    my_core<Policy>(..., &tg_result);
}

// 2. Helper function takes threadgroup pointer
ALWAYS_INLINE float compute_reduction(
    const device half *data,
    uint K,
    uint lane_id,
    uint warp_id,
    threadgroup float *tg_result
) {
    float sum = 0.0f;
    if (warp_id == 0u) {  // Only warp 0 does work
        // Parallel load/accumulate
        sum = simd_sum(sum);  // Fast warp reduction
        if (lane_id == 0u) {
            tg_result[0] = result;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return tg_result[0];  // Broadcast to all threads
}
```

---

## 7. Testing

### 7.1 Test Structure

```rust
// CPU reference implementation
fn cpu_my_kernel(input: &[f32], params: &Params) -> Vec<f32> { /* ... */ }

// Parity test (vs legacy + CPU)
fn run_parity_test(cfg: TestConfig) {
    // 1. Run legacy kernel
    // 2. Run new Foundry kernel
    // 3. Run CPU reference
    // 4. Compare all three
}

// Q8 policy test
fn run_q8_policy_test(cfg: TestConfig) {
    // Quantize input
    // Run with run_with_policy::<PolicyQ8>()
    // Verify doesn't crash + reasonable output
}
```

### 7.2 Tolerances

```rust
const CPU_TOLERANCE: f32 = 0.02;    // f16 precision loss
const PARITY_TOLERANCE: f32 = 0.005; // Parallel reduction order differs
```

---

## 8. Checklist for New Kernels

- [ ] Create `src/metals/<name>.metal` with policy template
- [ ] Create `src/metals/<name>.rs` with `Kernel` trait impl
- [ ] Add `pub mod <name>;` to `src/metals/mod.rs`
- [ ] Create `src/compound/stages/<name>.rs` for compound support
- [ ] Add to `src/compound/stages/mod.rs`
- [ ] Create `tests/<name>.rs` with parity tests
- [ ] Add `#ifndef POLICY_F16_DEFINED` guard if defining inline policy
- [ ] Verify buffer indices don't conflict with PolicyStage (0, 1)
- [ ] Declare threadgroup memory in kernel entry, not template
- [ ] Test with both `run()` (F16) and `run_with_policy::<PolicyQ8>()` (Q8)

## Policy Support Gotchas

When implementing kernels that support `Policy` templates (like `PolicyF16` or `PolicyQ8`):

1.  **Buffer Naming Convention**:
    *   **Buffer 0**: Must be the Input/Matrix data (e.g., `matrix` or `input`).
    *   **Buffer 1**: Must be the Scale bytes (e.g., `scale_bytes`).
    *   Even if your F16 policy doesn't use scales, you must bind this buffer (e.g., to the input tensor) in your Rust wrapper to maintain index alignment for Q8 policies that *do* expect it here.
    *   Subsequent buffers (Output, Params, etc.) should start from Buffer 2.

2.  **Threadgroup Memory**:
    *   **Do NOT** declare `threadgroup` memory inside the templated core function or the Policy struct.
    *   **DO** declare `threadgroup` memory in the `[[kernel]]` entry point function.
    *   Pass pointers to this threadgroup memory into your templated core function.
    *   Foundry's `run_with_policy` mechanism relies on the entry point being able to instantiate the template with specific shared memory bindings.

3.  **Rust Stage Emit**:
    *   When implementing `Stage::emit`, ensure the C++ code string you generate passes the `threadgroup` pointers (e.g., `tg_mem`) to the template function.
    *   The `PolicyStage` machinery automatically handles `matrix` (Buffer 0) and `scale_bytes` (Buffer 1) binding, so your `buffer_args()` should usually start from index 2 for your stage-specific arguments.

---

## 9. SIMD GEMV Fused Kernels (Decode Path)

For decode-time GEMV (seq_len=1), use the SIMD GEMV template system. This is faster than manually writing kernels and allows fusing RMSNorm + projections.

### 9.1 Using `#[derive(GemvKernel)]` (Preferred)

```rust
use metallic_macros::{GemvKernel, Epilogue};
use metallic::metals::matmul_gemv::hooks::F16CanonicalRmsnormHook;

// Define epilogue if needed (or use an existing one)
#[derive(Epilogue, Clone, Copy, Default)]
#[epilogue(gemv_struct = "SwiGluEpilogue", gemv_id = "swiglu", include = "swiglu/swiglu.metal")]
pub struct SwiGluEpilogue;

// Unified kernel definition
#[derive(GemvKernel)]
#[gemv_kernel(
    args = "SwiGluF16CanonicalFusedRmsnormArgs",
    heads = 2,
    cols_per_tg = 8,
    fast_path = true,
    gemv_n0 = "params->N0",
    data_ptrs("data_g", "data_u"),
    result_ptrs("out_res", "nullptr"),
    n_exprs("params->N0", "params->N1"),
    bias_ptrs("bias_g", "bias_u"),
    has_bias_flags("params->has_bias0", "params->has_bias1"),
    struct_defs_type(Q2FusedParams),
    hook = F16CanonicalRmsnormHook,
    epilogue = SwiGluEpilogue,
)]
pub struct SwiGluFusedKernel;

// Usage in compound kernel
let kernel = CompoundKernel::new("swiglu_fused")
    .main_dyn(Box::new(SwiGluFusedKernel::main_stage()))
    .build();
```

### 9.2 Key Attributes

| Attribute | Purpose |
|-----------|---------|
| `heads = N` | Number of output heads (QKV=3, Gate/Up=2) |
| `hook = Type` | `F16CanonicalRmsnormHook`, `Q8CanonicalHook`, etc. |
| `epilogue = Type` | Post-GEMV processing (SwiGLU, default, etc.) |
| `struct_defs_type(T)` | MetalStruct to inject (fused params) |

### 9.3 Available Hooks

| Hook | Description |
|------|-------------|
| `F16CanonicalHook` | F16 weights, no fused norm |
| `F16CanonicalRmsnormHook` | F16 weights + fused RMSNorm |
| `Q8CanonicalHook` | Q8 weights, no fused norm |
| `Q8CanonicalRmsnormHook` | Q8 weights + fused RMSNorm |

For more details, see:
- `docs-in-progress/KERNELS.md` — Architecture overview
- `docs-in-progress/MACROS.md` Section 10 — Full attribute reference
