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
| `Stage` | Implement `Stage` trait from stage templates + bindings | `#[stage(...)]`, `#[arg(...)]` |
| `CompoundKernel` | Compose stages into fused kernel | `#[compound(...)]`, `#[prologue]`, `#[main]`, `#[epilogue]` |
| `Epilogue` | Implement `Epilogue` and `Stage` for manual fusion | `#[epilogue(...)]` |
| `ConditionalKernel` | Dispatch to kernel variants based on runtime conditions | `#[conditional(...)]`, `#[when(...)]` |

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

> [!NOTE]
> **Dynamic Values**: The macro automatically unwraps `DynamicValue<T>` fields (e.g., `DynamicValue<u32>` becomes `uint`). It also generates a `{Name}Resolved` companion struct and implements the `Resolvable` trait, allowing these structures to be used in model definitions while maintaining strict C-layout compatibility at runtime.

### DynamicValue resolution contract

`DynamicValue<T>` is used by the Foundry DSL to reference runtime values by name (e.g. `"{seq_len}"`).

- **Resolution sources:** values are resolved from `TensorBindings` at runtime.
- **Fail-fast behavior:** if a required variable is missing or cannot be parsed into `T`, resolution **panics** and the error message directs you to add the value to the DSL (`architecture.prepare.globals` / `architecture.prepare.derived_globals`) or pass a runtime override.
- **Performance note:** for integer types (`u32`, `usize`) the resolver checks a fast `int_globals` table first to avoid string parsing in hot paths.

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

**All struct fields are treated as buffer arguments by default.** Buffer indices are auto-assigned starting from 0 in field declaration order.

```rust
#[derive(KernelArgs, Clone)]
pub struct GemvColMajor {
    pub matrix: TensorArg,           // Buffer 0 (auto-assigned)
    
    #[arg(output)]                   // Buffer 1 - mark as output (no auto-flush)
    pub result: TensorArg,
    
    pub params: GemvParams,           // Buffer 2 - Struct → setBytes
    
    pub scale: f32,                   // Buffer 3 - Primitive → setBytes
    
    #[arg(metal_type = "const device uchar*")]  // Buffer 4 - Override type inference
    pub quantized: TensorArg,
    
    #[arg(skip)]                     // Not a buffer arg - excluded from binding
    pub local_config: usize,
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
| `#[arg(output)]` | Mark as output (skips auto-flush) |
| `#[arg(skip)]` | Exclude from buffer args (not bound to Metal) |
| `#[arg(meta)]` | Metadata for kernel selection (excluded from buffer binding and Metal signature) |
| `#[arg(scale_for = "field")]` | Automatically derive scales for the target field (e.g. `{field}_scales`) |
| `#[arg(stage_skip)]` | Skip in compound stage emit (Policy provides) |
| `#[arg(metal_type = "...")]` | Override inferred Metal type |
| `#[arg(buffer = N)]` | **Optional**: Explicit buffer index (auto-assigns from max(N+1) onward) |

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
    include = ["utils/simd.metal", "utils/math.metal"],  // Optional
    dispatch = "per_row"  // Use a preset dispatch config
)]
pub struct GemvColMajorKernel {
    pub matrix: TensorArg,           // Buffer 0 (auto)
    #[arg(output)]
    pub result: TensorArg,           // Buffer 1 (output)
    pub params: GemvParams,          // Buffer 2
}
```

### Generated Output

```rust
impl Kernel for GemvColMajorKernel {
    type Args = GemvParams;
    
    fn function_name(&self) -> &str { "gemv_col_major_f16" }
    fn source(&self) -> KernelSource { KernelSource::File("gemv/col_major.metal") }
    fn includes(&self) -> Includes { Includes(vec!["utils/simd.metal", ...]) }
    fn bind(&self, encoder: &ComputeCommandEncoder) { self.bind_args(encoder); }
    fn dispatch_config(&self) -> DispatchConfig { /* generated from preset or manual impl */ }
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
| `epilogue_emit = "..."` | Template for Epilogue emission (enables Epilogue generation) |
| `epilogue_out_var = "..."` | Optional name for Epilogue output variable |
| `step = bool` | Enable `Step` implementation (default: `true`) |
| `execute = bool` | Enable `execute()` impl for `CompiledStep` (default: `true`) |
| `dispatch = bool/preset` | Enable default dispatch or use preset |
| `stage = "..."` | Custom Rust expression for `as_stage()` |
| `stage_emit = "..."` | Template for Stage emission code |
| `stage_out_var = "..."` | Optional name for Stage output variable |
| `dtype = Variant` | Optional `Dtype` override (e.g., `F16`, `Q8`) |

### Dispatch Presets

Presets automatically calculate grid and threadgroup sizes based on fields in the `params` struct.

| Preset | Description | Required Param Fields |
|--------|-------------|-----------------------|
| `per_element` | One thread per element | `total_elements` |
| `per_element_vec` | One thread per vector chunk | `total_elements`, `vector_width` |
| `per_row` | One thread per row | `total_elements`, `feature_dim` |
| `warp_per_row` | One warp (32 threads) per row | `n_dim` |
| `warp_per_row_2d` | Batched warp-per-row | `n_dim`, `batch` |
| `vec_N` | Vectorized (e.g. `vec_4`) | `total_elements` |

### Step Generation

When `step = true` (default), the macro automatically generates:
- `{Kernel}Step` — Serializable struct for use in model DSLs. String `Ref` fields replace `TensorArg`.
- `Compiled{Kernel}Step` — Optimized runtime step using `usize` index lookups.
- `Step` implementation that automatically handles `compile()` and routes `execute()` to the compiled path.

**Derived Scales in Steps:** Fields marked `#[arg(scale_for = "...")]` are automatically resolved during compilation to the `{target}_scales` symbol, making quantization transparent to the DSL.

### Stage/Epilogue Generation

When `stage_function` or `epilogue_emit` is specified, the macro automatically generates:
- `{Kernel}Stage` struct containing all `#name` fields NOT marked `stage_skip`.
- `as_stage()` method that clones current fields into the generated Stage struct.
- `Stage` implementation for `{Kernel}Stage`.
- `Epilogue` implementation for `{Kernel}Stage` (if `epilogue_emit` is present).

Since `TensorArg` implements `Default`, the generated Stage struct is always `Default`-able, allowing it to be used in `CompoundKernel::new()` planning.

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

## 4.5 `#[derive(Stage)]`

Generates a `compound::Stage` implementation directly from a Rust struct.

Use this when stage wiring should stay declarative but you still need dynamic policy/activation parameterization.

### `#[stage(...)]` attributes

| Attribute | Description |
|-----------|-------------|
| `include = "path.metal"` | Legacy single include (still supported). |
| `includes("a.metal", "b.metal")` | Static include list. |
| `include_exprs("self.activation.header()")` | Dynamic include expressions evaluated on `self`. |
| `emit = "..."` | Metal code template. |
| `out_var = "name"` | Output variable from `emit()`. |
| `template_bindings(name = "expr", ...)` | Bind template placeholders to Rust expressions. |
| `activation_field = "activation"` | Adds activation placeholders and `activation_meta()`. |
| `policy_field = "policy"` | Adds policy placeholders and `policy_meta()`. |
| `buffer_args_fn = "method"` | Delegate buffer signature construction to an instance method. |
| `struct_defs_method = "method"` | Delegate struct-def generation to an instance method. |
| `struct_defs = "TypeName"` | Inject `TypeName::METAL_STRUCT_DEF`. |
| `struct_defs("TypeA", "TypeB")` | Inject multiple MetalStruct definitions in order. |
| `struct_defs = ["TypeA", "TypeB"]` | Inject multiple MetalStruct definitions (array form). |
| `struct_defs_fn = "method"` | Call an associated function for struct defs. |

`#[derive(Stage)]` now auto-wraps non-empty `struct_defs()` output in a per-stage `#ifndef/#define` guard, so manual guarding is no longer required in stage helpers.

### Template placeholders

Built-ins:
- `{input_var}`
- `{out_var}`

Activation adapter placeholders:
- `{activation_header}`
- `{activation_struct}`

Policy adapter placeholders:
- `{policy_header}`
- `{policy_struct}`
- `{policy_short}`

Unknown placeholders fail at compile-time.
Generated `emit()` output always appends a trailing newline automatically.

---

## 5. `#[derive(Epilogue)]`

Implements both `Stage` and `Epilogue` traits for stages that are fused after a main operation.

### Usage

```rust
#[derive(Clone, Epilogue, Default)]
#[epilogue(
    include = "rmsnorm/rmsnorm.metal",
    emit = r#"
    // RMSNorm epilogue: scale by gamma
    half gamma_val = gamma[feature_idx];
    half {out_var} = {input_var} * gamma_val;"#
)]
pub struct RmsNormStage {
    #[arg(buffer = 2, metal_type = "const device half*")]
    pub gamma: TensorArg,
}
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `#[epilogue(include = "...")]` | Metal header file to include |
| `#[epilogue(emit = "...")]` | Template string with `{input_var}` and `{out_var}` |
| `#[epilogue(struct = "...")]` | Override Metal struct name (defaults to Rust name) |
| `#[epilogue(out_var = "...")]` | Explicitly name output variable |
| `#[epilogue(gemv_struct = "...")]` | (Optional) SIMD-GEMV epilogue struct name for `run_simd_gemv_template` |
| `#[epilogue(gemv_id = "...")]` | (Optional) SIMD-GEMV epilogue ID used for kernel naming/caching |
| `#[epilogue(simd_reduce = "...")]` | (Optional) SIMD reduce vars: `"gate: acc[0], up: acc[1]"` |
| `#[epilogue(simd_reduce_from = "16")]` | (Optional) Start level (default: 16 for 32-lane) |
| `#[epilogue(simd_reduce_to = "1")]` | (Optional) End level (default: 1) |
| `#[epilogue(simd_reduce_op = "add")]` | (Optional) Reduction op: `add`, `max`, or `min` |

---

## 6. `#[derive(CompoundKernel)]`

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
    pub post: EpilogueStage<MyEpilogue>,
    
    // Also include KernelArgs for buffer bindings
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
3. **Epilogues** — Post-processing (e.g., `MyEpilogue`)

**Note:** Many built-in inference kernels (e.g. `GemvV2Step`/`GemmV2Step`/`MatMulStep`) configure activations via `policy::activation::Activation` rather than through an `#[epilogue]` stage.

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
    pub input: TensorArg,
    #[arg(output)]
    pub output: TensorArg,
    pub params: MyParams,
}

impl Kernel for MyKernel {
    type Args = MyParams;
    
    fn function_name(&self) -> &str { "my_kernel" }
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

---

## 7. `#[derive(ConditionalKernel)]`

Generates dispatch logic for enums that select between kernel variants based on runtime conditions. Provides compile-time coverage analysis that errors on gaps or overlaps.

### Usage

```rust
#[derive(ConditionalKernel, Clone)]
#[conditional(selector = "batch: u32")]
pub enum MatmulDispatch {
    #[when(batch == 1)]
    Gemv(GemvKernel),
    
    #[when(batch > 1)]
    Gemm(GemmKernel),
}

// Range-based dispatch (use .in_() method syntax)
#[derive(ConditionalKernel, Clone)]
#[conditional(selector = "seq_k: usize")]
pub enum Softmax {
    #[when(seq_k.in_(0..=767))]
    Vec(SoftmaxVec),
    
    #[when(seq_k.in_(768..=1023))]
    Block(SoftmaxBlock),
    
    #[when(seq_k >= 1024)]
    Vec(SoftmaxVec),
}
```

### Generated Output

```rust
// Generated variant enum for pattern matching:
pub enum MatmulDispatchVariant {
    Gemv,
    Gemm,
}

impl MatmulDispatch {
    pub fn select(batch: u32) -> MatmulDispatchVariant {
        if batch == 1 { return MatmulDispatchVariant::Gemv; }
        if batch > 1 { return MatmulDispatchVariant::Gemm; }
        unreachable!("All conditions verified at compile-time")
    }
}

impl Kernel for MatmulDispatch {
    // Delegates to selected variant
}
```

---

### Attributes

| Attribute | Description |
|-----------|-------------|
| `#[conditional(selector = "name: Type")]` | Define selector parameters for the `select()` function |
| `#[when(condition)]` | Condition for selecting this variant |

### Selector Variables

The `selector` attribute defines parameters that become arguments to the generated `select()` function:

```rust
#[conditional(selector = "seq_k: usize")]  // → fn select(seq_k: usize)
#[conditional(selector = "batch: u32, dim: usize")]  // → fn select(batch: u32, dim: usize)
```

**The selector variable name must match the variable used in `#[when(...)]` conditions.**

### Supported Conditions

| Syntax | Example | Notes |
|--------|---------|-------|
| Equality | `batch == 1` | Exact match |
| Comparison | `batch > 1`, `seq_k >= 256` | Standard comparisons |
| Range (inclusive) | `seq_k.in_(0..=767)` | Use `.in_()` method |
| Range (exclusive) | `seq_k.in_(256..512)` | Exclusive upper bound |
| Range (open) | `seq_k >= 1024` | Unbounded upper |

> [!NOTE]
> Range conditions use the `.in_()` method syntax because Rust's `in` is a reserved keyword.

### Generated Output

The macro generates:
1. **`{Name}Variant` enum** — A discriminant enum with the same variant names
2. **`select()` method** — Returns the discriminant based on selector conditions
3. **`Kernel` impl** — Delegates all trait methods to the selected inner kernel

### Coverage Analysis

> [!IMPORTANT]
> **Both gaps and overlaps are compile-time errors:**

```rust
// ERROR: Overlapping conditions
#[when(batch == 1)]
A(KernelA),
#[when(batch >= 1)]  // Overlaps with batch == 1!
B(KernelB),

// ERROR: Coverage gap
#[when(batch == 1)]
A(KernelA),
#[when(batch > 3)]  // Gap: batch in 2..=3 not covered!
B(KernelB),
```

### Error Handling

| Error | Cause |
|-------|-------|
| *"Missing #[conditional(selector = ...)]"* | Forgot the selector attribute |
| *"Missing #[when(...)] condition"* | Variant without a condition |
| *"Overlapping conditions"* | Two variants match the same input |
| *"Coverage gap"* | Values exist that match no variant |

### Usage Pattern

```rust
impl Softmax {
    pub fn new(input: &TensorArg, output: &TensorArg, ..., seq_k: u32) -> Self {
        // Use generated select() to get discriminant
        match Self::select(seq_k as usize) {
            SoftmaxVariant::VecShort => Softmax::VecShort(SoftmaxVec::new(...)),
            SoftmaxVariant::BlockMid => Softmax::BlockMid(SoftmaxBlock::new(...)),
            // ...
        }
    }
}
```

---

## 9. CodeBuilder Utilities

The `CodeBuilder` module provides type-safe utilities for generating Metal code in compound kernels and stages.

### Location

```rust
use metallic::compound::code_builder::{
    CodeBuilder, MetalType, MetalVar, SimdReduceConfig, ReduceOp,
};
```

### `MetalType` — Type-Safe Metal Types

```rust
pub enum MetalType {
    Half, Float, Uint, Int, Bool,      // Scalars
    Half2, Half4, Float2, Float4,       // Vectors
    DevicePtr(MetalScalar),             // device T*
    ConstantPtr(MetalScalar),           // constant T*
    Custom(&'static str),               // User-defined
}
```

### `MetalVar` — Tracked Variables

```rust
let mut b = CodeBuilder::new("rmsnorm");
let inv_rms = b.declare_var("inv_rms", MetalType::Float);
// inv_rms.name() → "rmsnorm_inv_rms1"
// inv_rms.declaration() → "float rmsnorm_inv_rms1;"
```

### `SimdReduceConfig` — Flexible SIMD Reductions

```rust
// Full 32-lane reduction (default)
let config = SimdReduceConfig::default();  // from=16, to=1, op=Add

// 16-lane reduction
let config = SimdReduceConfig::lane_16(ReduceOp::Add);  // from=8, to=1

// Max reduction
let config = SimdReduceConfig::full_32_lane(ReduceOp::Max);

// Single level
let config = SimdReduceConfig::new(4, 4, ReduceOp::Add);  // just x += simd_shuffle_xor(x, 4)
```

### `ReduceOp` — Reduction Operations

```rust
pub enum ReduceOp {
    Add,  // val += simd_shuffle_xor(val, N)
    Max,  // val = max(val, simd_shuffle_xor(val, N))
    Min,  // val = min(val, simd_shuffle_xor(val, N))
}
```

### CodeBuilder Usage Example

```rust
let mut b = CodeBuilder::new("swiglu");

// Declare typed variables
let gate = b.declare_var("gate", MetalType::Float);
let up = b.declare_var("up", MetalType::Float);

// Emit declarations with initializers
b.emit_decl_init(&gate, "gate_data[idx]");
b.emit_decl_init(&up, "up_data[idx]");

// Emit SIMD reduction with config
b.emit_simd_reduce_multi(&[gate.name(), up.name()], SimdReduceConfig::default());

// Set output and finish
b.set_output_var(&up);
let (output_var, code) = b.finish();
```

### API Reference

| Method | Description |
|--------|-------------|
| `declare_var(name, type)` | Create a typed variable, returns `MetalVar` |
| `external_var(name, type)` | Register an external variable (e.g., kernel param) |
| `emit(line)` | Emit a line of code with indentation |
| `emit_decl(&var)` | Emit `type name;` |
| `emit_decl_init(&var, value)` | Emit `type name = value;` |
| `emit_assign(&var, value)` | Emit `name = value;` |
| `emit_simd_reduce(var)` | Full 32-lane Add reduction |
| `emit_simd_reduce_with_config(var, config)` | Custom reduction |
| `emit_simd_reduce_multi(vars, config)` | Reduce multiple vars interleaved |
| `finish() -> (String, String)` | Returns (output_var, code) |

---

## 10. `#[derive(GemvPrologue)]` — Pre-Main Setup Stage

Prologues run before the hook/main GEMV and set up threadgroup-shared state (e.g., computing `inv_rms` for RMSNorm).

### Usage

```rust
#[derive(GemvPrologue, Clone, Copy, Default)]
#[gemv_prologue(
    emit = r#"
    threadgroup float inv_rms_s;
    const float inv_rms = gemv_compute_inv_rms(vector_x, params->K, lid, wid, &inv_rms_s, epsilon);
    "#,
    includes("matmul_gemv/simd_common.metal")
)]
pub struct RmsnormPrologue;
```

### Attributes

| Attribute | Description |
|-----------|-------------|
| `emit = "..."` | Metal code emitted before hook preamble |
| `includes("...")` | Additional includes for prologue |

### Generated Code

```rust
impl GemvPrologue for RmsnormPrologue {
    fn includes() -> &'static [&'static str] { &["matmul_gemv/simd_common.metal"] }
    fn emit() -> String { /* emit code */ }
}
```

### Usage with GemvKernel

```rust
#[derive(GemvKernel)]
#[gemv_kernel(
    prologue = RmsnormPrologue,  // Pre-main setup
    hook = F16CanonicalRmsnormHook,
    epilogue = SwiGluEpilogue,
    // ... config ...
)]
pub struct MyFusedKernel;
```

---

## 11. `#[derive(GemvKernel)]` — Unified GEMV Kernel

Combines `GemvPrologue`, `GemvConfig`, `GemvHook`, and `GemvEpilogue` into a single derive macro for cleaner DX.

### Usage

```rust
#[derive(GemvKernel)]
#[gemv_kernel(
    // Config section
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
    
    // Hook and Epilogue references
    hook = F16CanonicalRmsnormHook,
    epilogue = SwiGluEpilogue,
)]
pub struct SwiGluFused;

// Usage: Get the composed stage via main_stage()
let stage = SwiGluFused::main_stage();  // -> GemvStage<...>
```

### Generated Code

- `impl GemvConfig for SwiGluFused { ... }` — All config constants and methods
- `SwiGluFused::main_stage()` — Returns `GemvStage<Self, Hook, Epilogue>`

### Attributes

| Attribute | Description |
|-----------|-------------|
| `args = "..."`| Args type name (must impl `HasMetalArgs`) |
| `heads = N` | Number of output heads (Q/K/V = 3, Gate/Up = 2) |
| `cols_per_tg = N` | Columns per threadgroup (default: 8) |
| `fast_path = bool` | Enable fast path optimizations |
| `gemv_n0 = "..."` | Expression for N0 dimension |
| `data_ptrs("...", ...)` | Per-head weight data pointers |
| `scale_ptrs("...", ...)` | (Optional) Per-head scale pointers for Q8 |
| `result_ptrs("...", ...)` | Per-head output pointers |
| `n_exprs("...", ...)` | Per-head output dimension expressions |
| `bias_ptrs("...", ...)` | Per-head bias pointers |
| `has_bias_flags("...", ...)` | Per-head has_bias flag expressions |
| `struct_defs_type(Type)` | (Optional) MetalStruct type to inject |
| `hook = Type` | Hook type (e.g., `F16CanonicalRmsnormHook`) |
| `epilogue = Type` | Epilogue type (e.g., `SwiGluEpilogue`) |

### Validation (at macro time)

- All array lengths (`data_ptrs`, `result_ptrs`, etc.) must equal `heads`
- `hook` and `epilogue` must be valid type paths

### vs. Separate `GemvConfig` + `GemvHook` + `Epilogue`

| Approach | Lines | Type Safety | Extensibility |
|----------|-------|-------------|---------------|
| Separate 3 derives | ~50 | Loose | High |
| Unified `GemvKernel` | ~20 | Validated | Moderate |

Use `GemvKernel` for most fused GEMV kernels. Use separate derives when you need unusual configurations or shared hooks/epilogues across multiple kernels.
