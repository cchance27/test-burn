# Implementation Plan - Backend DX Refactor & Optimization

This plan outlines the refactoring of our Metal-based backend framework to improve Developer Experience (DX), reduce boilerplate, and enable scalable kernel fusion.

## User Review Required

> [!IMPORTANT]
> This refactor involves breaking changes to the `Context<T>` and `call` APIs. The goal is to maximize flexibility without sacrificing type safety by using a standardized `TypeMap`-style registry for resources.

## Proposed Changes

### Core Framework Refactor: The "Foundry" Pattern

> [!TIP]
> To ensure stability during this massive refactor, we will implement the new system in parallel to the existing `Context<T>` under the name [Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26). This allows side-by-side verification and incremental migration.

#### [NEW] [foundry.rs](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry.rs)
- [x] **[Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26) Struct**: The new context object.
    - **Resource Registry**: Uses a `TypeMap` (via `Any` + `TypeId`) to store cached resources (`KvCache`, `TensorCache`) in a type-safe but generic way.
    - **No Generic Parameter**: [Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26) is not `Foundry<T>`. It holds resources for *any* type.
    - **Unified Dispatch**: Exposes a [dispatch(kernel, args)](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#208-239) method that replaces `call` and `call_custom`.

#### [NEW] [types/mod.rs]
- **Module Structure**:
    - [x] Create generic test harness (`run_gemv_test<T, P>`) <!-- id: 46 -->
    - [x] [mod.rs](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/mod.rs): Clean type aliases ([Device](file:///Volumes/2TB/test-burn/crates/metallic/src/types/metal.rs#14-15), [Buffer](file:///Volumes/2TB/test-burn/crates/metallic/src/compound/mod.rs#30-38)).
    - [x] [storage.rs](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs): Storage states ([Pooled](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#14-15), [Dedicated](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#9-10), [View](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#18-19)).
    - [x] `semantic.rs`: Semantic NewTypes (`Query`, `Key`, `Weight`).
    - [x] This replaces the temporary [types.rs](file:///Volumes/2TB/test-burn/crates/metallic/src/types.rs) and `typing.rs`.

#### [NEW] [helpers.rs]
- **Safe Wrappers**:
    - [x] Centralize all `unsafe { ... }` blocks here.
    - [x] Expose safe, Rusty APIs for things like `set_bytes`, `dispatch_threadgroups`, etc.
    - [x] This limits the blast radius of unsafe code to a single verified module.

#### [NEW] [typing.rs]
- **Strong Typing Patterns**:
    - [x] **NewType Wrappers**: `struct Query<T>(Tensor<T>)`, `struct Key<T>(Tensor<T>)`, `struct Value<T>(Tensor<T>)`.
        - Prevents accidental swapping of Q, K, V tensors in kernel arguments.
    - [x] **Typestate Pattern**: `struct Tensor<T, S: StorageState>`.
        - `S` can be [Pooled](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#14-15), [Dedicated](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#9-10), [View](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#18-19).
        - Ensures you don't accidentally free a [View](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#18-19) or write to a read-only tensor at compile time.

#### [NEW] [foundry/tensor.rs] (and [foundry/pool.rs])
- [x] **Foundry Tensor System**:
    - Complete clean implementation of `Tensor<T>` associated with [Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26) instead of `Context`.
    - `foundry::pool::MemoryPool`: Independent memory pool stored in Foundry's registry.
    - `TensorStorage`: Decoupled enum [Dedicated(&Foundry) | Pooled(&mut Foundry)](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/storage.rs#9-10).


#### [MODIFY] [dtypes.rs](file:///Volumes/2TB/test-burn/crates/metallic/src/tensor/dtypes.rs)
- [x] **Loaders & Storers**: Each `Dtype` defines Metal helper functions for loading/storing.
    - These are the *same* functions used currently in GEMV, but standardized into a header library.
    - This allows *any* kernel to support Q8/F16 inputs via the `FusionBuilder`.

#### [NEW] [metallic-macros](file:///Volumes/2TB/test-burn/crates/metallic-macros)
- [x] **`#[derive(KernelArgs)]`**:
    - **Rust**: Generates [bind](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/fusion.rs#136-139) logic.
    - **Metal**: Generates the matching `struct Params { ... }` definition.
    - **Layout**:
        - `[[buffer(0..10)]]`: Tensors.
        - `[[buffer(10..20)]]`: Intrinsics (e.g., threadgroup size).
        - `[[buffer(20)]]`: Parameters struct.
- [x] **`#[derive(Kernel)]`**:
    - Registers the kernel with the [Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26).
    - **Dependencies**: Supports `#[kernel(..., includes = ["helper.metal"])]`.
        - Replaces `kernel.sources` files.
        - Ensures compilation includes necessary helpers (simd, block-utils).
    - Generates the Metal function signature.

---

### Type-Safe Metal Code Generation (Priority)

> [!CAUTION]
> Current fusion.rs has fragile hardcoded Metal strings that can diverge from Rust definitions. This section addresses that critical issue.

See [macro_system_design.md](file:///Users/christopherchance/.gemini/antigravity/brain/31358513-63bb-4832-ac62-cf4e3cccd607/macro_system_design.md) for full design.

#### [NEW] Metal Type Markers ([types/metal.rs](file:///Volumes/2TB/test-burn/crates/metallic/src/types/metal.rs))
- [x] `DevicePtr<T>`, `DevicePtrMut<T>`, `ConstantRef<T>` markers
- [x] [MetalType](file:///Volumes/2TB/test-burn/crates/metallic/src/types/metal.rs#32-37) trait for Rust→Metal type mapping
- [x] Compile-time type inference for buffer signatures

#### [NEW] `#[derive(MetalStruct)]`
- [x] Generates Metal struct definition from `#[repr(C)]` Rust struct
- [x] Emits to `OUT_DIR/metal_structs.metal` via build.rs
- [x] Eliminates duplicate [GemvParams](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv.rs#9-23) definitions

#### [NEW] `#[derive(MetalPolicy)]`
- [x] Declarative `#[param(from = "arg")]` annotations
- [x] Auto-generates [init_params_code()](file:///Volumes/2TB/test-burn/crates/metallic/src/policies.rs#16-19) from bindings
- [x] Provides [buffer_types()](file:///Volumes/2TB/test-burn/crates/metallic/src/policies.rs#20-23) for signature inference

#### [MODIFY] `#[derive(KernelArgs)]`
- [x] Add [metal_signature()](file:///Volumes/2TB/test-burn/crates/metallic/src/fusion.rs#35-46) method generation
- [x] Combine with `MetalPolicy::buffer_types()` for full signature
- [x] Validate buffer index ordering at compile time

#### [NEW] Compound Kernel System (Replaces Template-Based Fusion)

> [!IMPORTANT]
> This replaces the current [FusedGemv](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/fusion.rs#17-55) template-shim approach with explicit stage composition.

See [fusion_design.md](file:///Users/christopherchance/.gemini/antigravity/brain/31358513-63bb-4832-ac62-cf4e3cccd607/fusion_design.md) for full design.

**Stage Trait:**
```rust
pub trait Stage: Send + Sync {
    fn includes(&self) -> &[&'static str];
    fn buffer_args(&self) -> &[BufferArg];
    fn emit(&self, input_var: &str) -> (String, String);
}
```

**Adapters (reuse existing types):**
- `PolicyStage<P: MetalPolicy>` - wraps policies as prologue stages
- `EpilogueStage<E: Epilogue>` - wraps epilogues as post-processing stages

**Builder API:**
```rust
let kernel = CompoundKernel::new("fused_q8_gemv_silu")
    .prologue::<PolicyQ8>()
    .main(GemvCoreStage::new())
    .epilogue::<SiLUEpilogue>()
    .build();  // → CompoundKernel<Fused>
```

**`#[derive(CompoundKernel)]` Macro (Implemented):**

> [!IMPORTANT]
> This macro auto-generates Args structs from stage requirements, eliminating manual duplication.

```rust
#[derive(CompoundKernel)]
#[compound(name = "gemv_q8")]
pub struct GemvQ8Compound {
    #[prologue]
    policy: PolicyStage<PolicyQ8>,
    #[main]
    gemv: GemvCoreStage,
    #[epilogue]
    none: EpilogueStage<EpilogueNone>,
}
// Generates: GemvQ8Compound + GemvQ8CompoundArgs (merged from stages)
```

Key features:
- [x] Each Stage provides `const BUFFER_ARGS: &'static [BufferArg]` (compile-time accessible)
- [x] Macro parses `#[prologue]`, `#[main]`, `#[epilogue]` fields
- [x] Merges all stage buffer args into `{Name}Args` struct
- [x] Generates `impl Kernel` and `impl BindArgs`

**Parity Testing:**
1. [x] Create `CompoundGemv` using [derive(CompoundKernel)](file:///Volumes/2TB/test-burn/crates/metallic-macros/src/lib.rs#474-587) macro
2. [x] Verify F16 + Q8 output matches [FusedGemv](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/fusion.rs#17-55)
3. [ ] Remove [FusedGemv](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/fusion.rs#17-55) after verification

---

### Foundry DX Improvements

> [!IMPORTANT]
> These improvements address pain points discovered during GEMV parity testing. They reduce boilerplate and prevent common errors.

#### Automatic Tensor Flushing
- **Problem**: Pooled tensors with `CopyFrom` use staging buffers that must be manually flushed before dispatch. Forgetting `flush_host_writes()` causes silent data corruption.
- **Solution**: Mark output buffers explicitly; all others flush automatically.

```rust
// Before: error-prone manual flush
a.flush_host_writes().unwrap();
x.flush_host_writes().unwrap();
foundry.dispatch(&kernel, ...);

// After: automatic (outputs are marked, inputs flush automatically)
#[arg(buffer = 2, output)]  // Only outputs need marking
pub y: &Tensor<T>,
```
- [x] Implemented in `KernelArgs` derivation (calls `KernelArg::flush` for non-outputs).

#### Accept Tensors Directly in Kernel Args
- **Problem**: `BufferBinding::with_offset(tensor.buf.clone(), tensor.offset)` is verbose and error-prone.
- **Solution**: Accept `&Tensor<T>` directly. The macro extracts `.buf` and `.offset` automatically.

```rust
// Before: verbose and error-prone
a: BufferBinding::with_offset(a.buf.clone(), a.offset),

// After: just pass the tensor reference
a: &tensor_a,
```
- [x] Implemented via `TensorArg` and macro support.

#### Non-Static Kernels Support
- **Problem**: `TypeId` requires `'static`, but kernels with `&Tensor<T>` references are not `'static`.
- **Solution**: The [Kernel](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#253-288) trait now includes a `type Id: 'static`. The macro automatically generates a unique marker struct (e.g., `GemvF16FullKernelId`) for each kernel, allowing the [Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26) cache to remain static while the kernels can have lifetimes.
- [x] Implemented.

#### Dispatch Configuration via [dispatch_config](file:///Volumes/2TB/test-burn/crates/metallic/tests/derive_compound.rs#43-57) Method
- **Problem**: Grid/group size calculation was callsite-dependent.
- **Solution**: Kernels now implement [dispatch_config()](file:///Volumes/2TB/test-burn/crates/metallic/tests/derive_compound.rs#43-57) themselves. `Foundry::run()` uses this method for a simplified "one-line" dispatch.

```rust
// Callsite becomes simple:
foundry.run(&kernel)?;  // No grid/group args!
```
- [x] Implemented in `Kernel` trait and `Foundry::run`.

---

### Pending Improvements (from Code Review)

#### 1. Safety & Correctness
- [ ] **Fix Pool/Tensor Lifetime Gap**: `Tensor<T, Pooled>` can outlive `MemoryPool`. Implement `Rc<PoolGuard>` arena pattern or phantom lifetimes.
- [ ] **Centralize Unsafe Code**: Move scattered `unsafe` blocks (e.g., pool.rs, tensor.rs) to `safety/mod.rs` with explicit `SAFETY` comments.
- [ ] **Macro Robustness**: Replace string-based type detection (`"TensorArg"`) in macros with proper `syn::Type` traversal to support aliases and qualified paths.

#### 2. Cleanup
- [ ] **Remove `TensorStorage` Enum**: It's redundant with the `StorageState` typestate pattern.

#### 3. API Polish
- [ ] **Add `foundry.alloc<T>(dims, init)` helper**: Simpler than `Tensor::new(foundry...`.
- [ ] **Add `#[must_use]`**: To high-value methods like `dispatch` and `run`.

### Kernel & Fusion Improvements

#### [MODIFY] [build_system.rs / build.rs]
- **Additive Build Logic**:
    - **Do NOT touch** existing `kernel.sources` logic for legacy kernels.
    - [x] Add a new pass to scan `src/metals` for new kernels.
    - [x] Generate `common_headers.metal` for the new `metals` directory only.
- **Verification**: This allows us to dual-compile: `LegacyKernel` and `PortedKernel` exist side-by-side.

#### [NEW] [fusion.rs]
- **Source-Level Fusion Builder**:
    - **Concept**: Instead of hardcoded switch-cases, we generate Metal source code on the fly based on Rust types.
    - **Traits**:
        - `trait MetalPolicy`: Defines the Metal header to include and the struct name to use.
        - `trait Epilogue`: Defines the Metal header and template args.
    - **Workflow**:
        1. Rust: `Gemv::new(lhs, rhs).epilogue(RmsNorm::new())`
        2. Builder: Collects headers ([policy_q8.metal](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/policies/policy_q8.metal), `epilogue_rmsnorm.metal`).
        3. Builder: Generates a "shim" kernel string.
        4. Foundry: Compiles this string using `newLibraryWithSource` and caches the pipeline.
- [x] Implemented via `CompoundKernel`.

- **Dtype Handler Injection**:
    - [x] [TensorElement](file:///Volumes/2TB/test-burn/crates/metallic/src/tensor/dtypes.rs#12-35) trait extended with `type Policy: MetalPolicy`.
    - [x] This allows [Q8](file:///Volumes/2TB/test-burn/crates/metallic/src/policies.rs#27-28), `Q4`, [F16](file:///Volumes/2TB/test-burn/crates/metallic/src/tensor/dtypes.rs#201-202) to automatically provide their load/store logic.

#### [DEPRECATE] Invocables
- `DefaultKernelInvocable` and `CustomKernelInvocable` will be phased out.
- The new `Foundry::dispatch` path handles all cases (single/multi-output) uniformly.

---

### Metrics & Observability Integration

> [!IMPORTANT]
> We must maintain parity with the legacy system's detailed kernel dispatch metrics.

#### [NEW] [foundry/profiling.rs]
- [ ] **Port `GpuScopeGuard` and Scoping Logic**:
    - Implement `gpu_scope_stack` and `pending_gpu_scope` in `Foundry`.
    - Provide `foundry.with_gpu_scope("name", || { ... })` API.
- [ ] **Ergonomic Auto-Profiling**:
    - **Auto-Scope Dispatch**: Update `Foundry::dispatch` and `Foundry::run` to automatically create a GPU scope using `kernel.function_name()`.
    - **Zero-Overhead Default**: Ensure profiling overhead is negligible when disabled (using `profiling_state` check).
    - **Unified Submission**: Centralize the complex `finalize_active_command_buffer_if_latency` logic into the `dispatch` pipeline to ensure all kernels get proper latency handling without manual calls.
- [ ] **Integrate `command_buffer_pipeline`**:
    - Ensure `Foundry` uses the existing `command_buffer_pipeline` for submission.
    - Pass hierarchical labels to `pipeline.submit()`.
- [ ] **Async Metrics Recording**:
    - Ensure `metric_instrumentation::record_metric_async!` is called on completion.

---

### Qwen2.5 Migration

> [!IMPORTANT]
> The inference pipeline requires several kernels to be ported. See [Qwen2.5 Migration Checklist](file:///Volumes/2TB/test-burn/docs-in-progress/qwen25_migration_checklist.md) for the full list.

#### [NEW] [qwen25_migration.md]
- [ ] **Port Core Kernels**: `EmbeddingLookup`, `RMSNorm`, `RoPE`, `SwiGLU`, `ElemwiseAdd`.
- [ ] **Port System Kernels**: `SampleTopKTopP`, `Arange`, `Ones`.
- [ ] **Port Fusion Kernels**: `MatmulGemvQkvFusedRmsnorm`, `MatmulF16CanonicalQkvFusedRmsnorm`.

---

### Semantic Type System & Cleanup

> [!TIP]
> We will encapsulate complex `objc2` types into semantic wrappers to improve code readability and prevent "import sprawl".

CAUTION: We need to make sure we're not breaking our legacy systems or changing those systems to use semantics as that would make our new commits very dirty and leaked from what we're working on. 

#### [MODIFY] [types/metal.rs]
- [ ] **Wrap Raw Metal Objects**:
    - `pub struct MetalDevice(Retained<ProtocolObject<dyn MTLDevice>>)`
    - `pub struct MetalQueue(Retained<ProtocolObject<dyn MTLCommandQueue>>)`
    - `pub struct MetalBuffer(Retained<ProtocolObject<dyn MTLBuffer>>)`
    - `pub struct MetalEncoder(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>)`
- [ ] **Impl safe methods**: Expose commonly used Metal methods (e.g. `label()`, `set_label()`) on these wrappers so consumers don't need `objc2` imports.
- [ ] **Update Foundry**: Switch `Foundry` struct to use these new types.

---

### Import Hygiene & DX Cleanup

> [!TIP]
> These items reduce `objc2` sprawl and improve consumer DX. See `code_review.md` §9 for full details.

#### ~~[MODIFY] [types/dispatch.rs] ← Move from `foundry/mod.rs`~~ ✅ DONE
- [x] **Move `DispatchConfig`** from `foundry/mod.rs` to `types/dispatch.rs`
- [x] **Convert to struct**: `DispatchConfig { grid: GridSize, group: ThreadgroupSize }`
- [x] **Add `GridSize`/`ThreadgroupSize`**: Pure-Rust structs with `usize` fields
- [x] **Add `From` impls**: Bidirectional conversion with MTLSize
- [x] **Consistency pass**: Updated `foundry/`, `compound/`, `metals/gemv/` (legacy files untouched)

#### [NEW] [types/prelude.rs]
- [ ] **Internal objc2 trait prelude**: Centralize `MTLCommandBuffer as _`, `MTLDevice as _`, etc.
- [ ] Update internal modules to use `use crate::types::prelude::*;`.
- [ ] Ensure consumers never need direct `objc2_metal` imports.

#### [MODIFY] Conversion Patterns
- [ ] **Rename `as_stage()` → `to_stage()`**: Allocating method should use `to_` prefix.
- [ ] **Consider `impl From<&K> for Box<dyn Stage>`**: Enable `main_dyn(kernel)` without explicit conversion.
- [ ] **`impl Into` on API methods**: Use callee-side conversion bounds for cleaner callsites.

#### [MODIFY] Import Cleanup
- [ ] Remove verbose `crate::types::` qualifications in `foundry/`, `metals/`.
- [ ] Add proper `use crate::types::{...}` at module tops.

---

## Verification Plan

### Automated Tests
- `cargo test --test fusion_test`: Verified end-to-end compilation and execution of [FusedGemv](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/fusion.rs#17-55).
  - Validated dynamic source generation with [SourceBuilder](file:///Volumes/2TB/test-burn/crates/metallic/src/fusion.rs#63-69).
  - Validated header inclusion ([includes()](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#261-264)).
  - Validated Q8/F16 policy switching (via `fusion_test`).
- `cargo test --test gemv`: Verified parity of legacy Gemv kernels.

### Manual Verification
- Verified Metal source generation output (via inspection of [Foundry](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs#18-26) logic and test success).
