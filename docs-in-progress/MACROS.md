# Metallic Macros Reference

This document provides an overview of the derive macros in `metallic-macros` for use when porting kernels to the Foundry system.

---

## Quick Reference

| Macro | Purpose | Required Traits/Attrs |
|-------|---------|----------------------|
| `MetalStruct` | Generate Metal struct from Rust `#[repr(C)]` struct | `#[metal(...)]` |
| `MetalPolicy` | Implement `MetalPolicy` trait for dtype handling | `#[policy(...)]`, `#[param(...)]` |
| `KernelArgs` | Generate buffer binding + Metal signature | `#[arg(...)]` |
| `Kernel` | Implement `Kernel` trait for standalone kernels | `#[kernel(...)]` |
| `CompoundKernel` | Compose stages into fused kernel | `#[compound(...)]`, `#[prologue]`, `#[main]`, `#[epilogue]` |

---

## 1. `#[derive(MetalStruct)]`

Generates a Metal struct definition string from a Rust `#[repr(C)]` struct.

### Usage

```rust
#[derive(MetalStruct)]
#[repr(C)]
#[metal(name = "GemvParams")]  // Optional: override struct name
pub struct GemvParams {
    #[metal(name = "K")]  // Optional: override field name
    pub k: u32,
    pub n: u32,
    #[metal(skip)]        // Skip this field in Metal output
    pub rust_only: bool,
}
```

### Generated Output

```rust
impl GemvParams {
    pub const METAL_STRUCT_DEF: &'static str = r#"
struct GemvParams {
    uint K;
    uint n;
};"#;
}
```

### Type Mappings

| Rust | Metal |
|------|-------|
| `u8` | `uchar` |
| `u16` | `ushort` |
| `u32` | `uint` |
| `u64` | `ulong` |
| `i32` | `int` |
| `f32` | `float` |
| `f16` | `half` |

---

## 2. `#[derive(MetalPolicy)]`

Implements the `MetalPolicy` trait for dtype-specific loading/storing patterns (e.g., Q8, F16).

### Usage

```rust
#[derive(MetalPolicy)]
#[policy(header = "policies/policy_q8.metal", struct_name = "PolicyQ8")]
pub struct PolicyQ8 {
    #[param(from = "matrix")]
    matrix: DevicePtr<u8>,
    
    #[param(from = "scale_bytes")]
    scales: DevicePtr<u8>,
    
    #[param(from = "params.weights_per_block")]  // Dot notation → pointer deref
    weights_per_block: u32,
}
```

### Generated Output

```rust
impl MetalPolicy for PolicyQ8 {
    fn header(&self) -> &'static str { "policies/policy_q8.metal" }
    fn struct_name(&self) -> &'static str { "PolicyQ8" }
}

impl PolicyQ8 {
    pub const INIT_PARAMS_CODE: &'static str = 
        "pp.matrix = matrix; pp.scales = scale_bytes; pp.weights_per_block = params->weights_per_block;";
}
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `#[policy(header = "...")]` | Metal header file to include |
| `#[policy(struct_name = "...")]` | Metal struct name for `using Policy = ...` |
| `#[param(from = "arg")]` | Maps kernel buffer arg → policy param |

---

## 3. `#[derive(KernelArgs)]`

Generates buffer binding code and Metal signature metadata. **Most commonly used.**

### Usage

```rust
#[derive(KernelArgs, Clone)]
pub struct GemvColMajor {
    #[arg(buffer = 0)]
    pub matrix: TensorArg,           // Buffer type → auto-detect
    
    #[arg(buffer = 1, output)]       // Mark as output (no auto-flush)
    pub result: TensorArg,
    
    #[arg(buffer = 2)]
    pub params: GemvParams,           // Struct → setBytes
    
    #[arg(buffer = 3)]
    pub scale: f32,                   // Primitive → setBytes
    
    #[arg(buffer = 4, metal_type = "const device uchar*")]  // Override inference
    pub quantized: TensorArg,
}
```

### Generated Output

```rust
impl GemvColMajor {
    pub fn bind_args(&self, encoder: &ComputeCommandEncoder) {
        // Buffer 0: matrix (auto-flushed, then setBuffer)
        // Buffer 1: result (output, no flush, setBuffer)
        // Buffer 2: params (setBytes with size)
        // ...
    }
}

impl HasMetalArgs for GemvColMajor {
    const METAL_ARGS: &'static [(&'static str, u64, &'static str)] = &[
        ("matrix", 0, "const device half*"),
        ("result", 1, "device half*"),
        ("params", 2, "const constant GemvParams*"),
        // ...
    ];
}

impl BindArgs for GemvColMajor { ... }
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `#[arg(buffer = N)]` | Buffer index in Metal signature |
| `#[arg(buffer = N, output)]` | Mark as output (skips auto-flush) |
| `#[arg(buffer = N, metal_type = "...")]` | Override inferred Metal type |

### Type Inference Rules

| Rust Type | Inferred Metal Type |
|-----------|---------------------|
| `TensorArg`, `&Tensor<T>` | `const device half*` (or `device half*` if output) |
| `u32`, `i32`, `f32`, etc. | `constant uint&`, `constant int&`, etc. |
| `GemvParams` (PascalCase) | `const constant GemvParams*` |

---

## 4. `#[derive(Kernel)]`

Implements the `Kernel` trait for standalone (non-compound) kernels.

### Usage

```rust
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "gemv/col_major.metal",
    function = "gemv_col_major_f16",
    args = "GemvParams",  // Type with METAL_STRUCT_DEF for struct_defs()
    include = ["utils/simd.metal", "utils/math.metal"]  // Optional
)]
pub struct GemvColMajorKernel {
    #[arg(buffer = 0)]
    pub matrix: TensorArg,
    // ...
}

impl GemvColMajorKernel {
    // REQUIRED: implement dispatch_config()
    fn dispatch_config(&self) -> DispatchConfig {
        DispatchConfig::d1(self.n / 8, 256)
    }
}
```

### Generated Output

```rust
pub struct GemvColMajorKernelId;  // For pipeline caching

impl Kernel for GemvColMajorKernel {
    type Args = GemvParams;
    type Id = GemvColMajorKernelId;
    
    fn function_name(&self) -> &'static str { "gemv_col_major_f16" }
    fn source(&self) -> KernelSource { KernelSource::File("gemv/col_major.metal") }
    fn includes(&self) -> Includes { Includes(vec!["utils/simd.metal", ...]) }
    fn bind(&self, encoder: &ComputeCommandEncoder) { self.bind_args(encoder); }
    fn dispatch_config(&self) -> DispatchConfig { Self::dispatch_config(self) }
    fn struct_defs(&self) -> String { GemvParams::METAL_STRUCT_DEF.to_string() }
}
```

### Required Attributes

| Attribute | Description |
|-----------|-------------|
| `source = "..."` | Path to `.metal` file (relative to `src/metals/`) |
| `function = "..."` | Metal kernel function name |

### Optional Attributes

| Attribute | Description |
|-----------|-------------|
| `args = "TypeName"` | Type with `METAL_STRUCT_DEF` for struct injection |
| `include = [...]` | Additional Metal headers to include |
| `stage_function = "..."` | Template function name for Stage fusion (enables Stage generation) |
| `threadgroup = "..."` | Threadgroup memory declaration for Stage emit (e.g., `"float shared[256]"`) |

### Stage Generation

When `stage_function` is specified, the macro automatically generates:
- `{Kernel}Stage` struct implementing `Stage` trait
- `as_stage()` returns this generated Stage

```rust
#[derive(Kernel, KernelArgs, Clone)]
#[kernel(
    source = "rmsnorm/rmsnorm.metal",
    function = "rmsnorm_kernel_f16",
    stage_function = "run_rmsnorm_core",  // Enables Stage generation
    args = "RmsNormParams",
    threadgroup = "float tg_inv_rms",
)]
pub struct RmsNorm {
    #[arg(buffer = 0, stage_skip)]  // Excluded from Stage (PolicyStage provides)
    pub input: TensorArg,
    #[arg(buffer = 1, stage_skip)]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 2, output)]
    pub output: TensorArg,
    #[arg(buffer = 3)]
    pub gamma: TensorArg,
    #[arg(buffer = 4)]
    pub params: RmsNormParams,
}
```

### Metal Source Requirements

> [!CAUTION]
> Your Metal file must follow the Stage signature convention for `stage_function` to work correctly.

**Stage Function Signature Convention:**

```metal
// Template function with Policy type parameter
// CONVENTION: fn<Policy>(matrix, [stage_args in buffer order], scale_bytes, gid, lid, [threadgroup])
template<typename Policy>
ALWAYS_INLINE void run_my_kernel_core(
    const device uchar *matrix,        // Buffer 0 - Policy input (stage_skip)
    device half *output,                // Buffer 2 - your output
    constant uint &seq_len,             // Buffer 3 - scalar uses reference (constant T&)
    constant uint &batch_size,          // Buffer 4
    constant MyParams *params,          // Buffer 5 - struct uses pointer (constant Struct*)
    const device uchar *scale_bytes,    // Buffer 1 - Policy scales (stage_skip)
    uint3 gid,
    uint3 lid,
    threadgroup float *shared           // Threadgroup - array passed as pointer
) {
    // Use Policy::load_weights<N>() for dequantized loading
    float4 w = Policy::template load_weights<4>(matrix, offset, scale_bytes);
    // Scalars are direct values (reference), no dereferencing needed
}
```

**Type Convention:**
| Rust Type | Metal Type | Notes |
|-----------|------------|-------|
| `TensorArg` | `device half*` or `const device uchar*` | Buffers as pointers |
| `u32`, `i32`, etc. | `constant uint&`, `constant int&` | Scalars as references |
| `SomeParams` | `constant SomeParams*` | Structs as const pointers |
| Threadgroup arrays | `threadgroup T*` | Arrays decay to pointers |

**Standalone Entry Point:**

```metal
#ifndef FUSED_KERNEL
#ifndef POLICY_F16_DEFINED
#define POLICY_F16_DEFINED
struct PolicyF16 { /* inline load helpers */ };
#endif

[[kernel]] void my_kernel_f16(
    const device uchar *input [[buffer(0)]],
    const device uchar *scale_bytes [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &seq_len [[buffer(3)]],  // Reference style for scalars
    constant uint &batch_size [[buffer(4)]],
    constant MyParams *params [[buffer(5)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float shared[256];  // Declare here, not in template
    run_my_kernel_core<PolicyF16>(input, output, seq_len, batch_size, params, scale_bytes, gid, lid, shared);
}
#endif
```

**Key Requirements:**
1. **Template function** (`run_{name}_core<Policy>`) - The `stage_function` value
2. **Type matching** - Rust types must match Metal types per convention above
3. **Guards** - `#ifndef FUSED_KERNEL` and `#ifndef POLICY_F16_DEFINED`
4. **Threadgroup in entry** - Declare in `[[kernel]]`, pass as pointer

---

## 5. `#[derive(CompoundKernel)]`

Composes multiple `Stage` implementations into a fused kernel with generated Metal source.

### Usage

```rust
#[derive(CompoundKernel, KernelArgs, Clone)]
#[compound(name = "gemv_q8_silu", manual_output = true)]
pub struct GemvQ8SiluCompound {
    #[prologue]
    pub policy: PolicyStage<PolicyQ8>,
    
    #[main]
    pub gemv: GemvCoreStage,
    
    #[epilogue]
    pub activation: EpilogueStage<SiLUEpilogue>,
    
    // Also include KernelArgs for buffer bindings
    #[arg(buffer = 0)]
    pub matrix: TensorArg,
    // ...
}

impl GemvQ8SiluCompound {
    // REQUIRED: implement dispatch_config()
    fn dispatch_config(&self) -> DispatchConfig { ... }
}
```

### Stage Order

1. **Prologues** — Input loading, dequantization (e.g., `PolicyStage`)
2. **Main** — Core computation (e.g., `GemvCoreStage`)
3. **Epilogues** — Post-processing, activations (e.g., `SiLUEpilogue`)

### Attributes

| Attribute | Description |
|-----------|-------------|
| `#[compound(name = "...")]` | Metal function name |
| `#[compound(manual_output = true)]` | Disable auto `output[idx] = result;` |
| `#[prologue]` | Mark field as prologue stage |
| `#[main]` | Mark field as main stage (exactly one required) |
| `#[epilogue]` | Mark field as epilogue stage |

---

## Common Patterns

### 1. Simple Kernel (No Fusion)

```rust
#[derive(KernelArgs, Clone)]
pub struct MyKernel {
    #[arg(buffer = 0)]
    pub input: TensorArg,
    #[arg(buffer = 1, output)]
    pub output: TensorArg,
    #[arg(buffer = 2)]
    pub params: MyParams,
}

impl Kernel for MyKernel {
    type Id = MyKernelId;
    type Args = MyParams;
    
    fn function_name(&self) -> &'static str { "my_kernel" }
    fn source(&self) -> KernelSource { KernelSource::File("my_kernel.metal") }
    fn bind(&self, enc: &ComputeCommandEncoder) { self.bind_args(enc); }
    fn dispatch_config(&self) -> DispatchConfig { ... }
    fn struct_defs(&self) -> String { MyParams::METAL_STRUCT_DEF.to_string() }
}
```

### 2. Fused Kernel with Policy

```rust
#[derive(CompoundKernel, KernelArgs, Clone)]
#[compound(name = "fused_gemv_q8", manual_output = true)]
pub struct FusedGemvQ8 {
    #[prologue]
    policy: PolicyStage<PolicyQ8>,
    #[main]
    gemv: GemvCoreStage,
    #[epilogue]
    none: EpilogueStage<EpilogueNone>,
    
    #[arg(buffer = 0)]
    matrix: TensorArg,
    // ... other args
}
```

---

## Troubleshooting

### "KernelArgs only supports named struct fields"
Ensure your struct uses named fields, not tuple syntax.

### Buffer indices not working
Buffer indices must be unique. Args are sorted by index in the Metal signature.

### Type inference wrong
Use `#[arg(buffer = N, metal_type = "...")]` to override.

### Struct not included in Metal source
Ensure the params type has `#[derive(MetalStruct)]` and is referenced via `args = "TypeName"` in `#[kernel(...)]`.
