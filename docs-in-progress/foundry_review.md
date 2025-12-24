# Foundry System Architecture Review

## Overview
The Foundry system represents a significant step forward in type safety and developer experience compared to the legacy `Context<T>` implementation. However, a review of the current codebase reveals several opportunities for tightening the architecture, improving performance, and reducing fragility.

## 1. Safety & Type System
**Strengths:**
- **Typestate Pattern**: `Tensor<T, S>` correctly enforces storage states (Pooled vs Dedicated) at compile time.
- **Kernel IDs**: Static `TypeId` generation for kernels allows safe caching of pipelines.

**Weaknesses:**
- **Raw `objc2` Exposure**: `Foundry` still exposes raw `Retained<ProtocolObject<dyn MTL...>>` types in its public API (e.g., `device`, `queue`). This leaks `objc2` dependencies to consumers.
- **Fragile Macro Parsing**: `#[derive(KernelArgs)]` relies on string matching (`"TensorArg"`, `"Tensor <"`) to detect buffer types. This is brittle and will break with type aliases or qualified paths.

## 2. Kernel Composition & Fusion
**Strengths:**
- **Compound Kernel Builder**: Flexible builder pattern allows dynamic composition of stages.
- **Declarative Policies**: `MetalPolicy` trait creates a clear contract for dtype-specific behavior.

**Weaknesses:**
- **Inefficient Re-Construction**: `CompoundKernel::new` is often called multiple times (once for source, once for includes, once for struct defs) within `derive` macros. This leads to redundant allocations.
- **Hot-Path Allocations**: `Foundry::run_with_policy` constructs a new `CompoundKernel` (and generates its source string) on *every call*. This is a significant performance regression compared to compiled templates.
    - **Fix**: We need a way to cache the *constructed* compound kernel or make it static.

### Critical Bug Discovered
- **Cache Collisions**: `CompoundKernel<Fused>` implements `Kernel::Id` as `String`. Since `Foundry` caches pipelines keyed by `TypeId::of::<K::Id>()`, **ALL dynamic compound kernels share the same cache key**.
    - **Result**: The first dynamic kernel compiled will overwrite/be reused for all others, leading to incorrect execution (e.g., executing a Q8 kernel when an F16 one was requested).
    - **Fix**: Pipeline cache must rely on a content hash (source code hash + constants) or a unique value-based ID, not just static TypeId.

## 3. Developer Experience (DX)
**Strengths:**
- **Unified Dispatch**: `Foundry::run` simplifies the call site significantly compared to `ctx.call`.
- **Auto-Gen Structs**: `#[derive(MetalStruct)]` eliminates the need to manually sync Rust and Metal struct definitions.

**Ambiguities:**
- **Include Resolution**: `Foundry::load_kernel` implements custom logic to resolve and strip `#include` directives. This "pre-processor" logic mimics the Metal compiler but might behave differently, leading to confusing debugging sessions if paths diverge.

## 5. Import Hygiene & DX Patterns

> [!IMPORTANT]
> Reducing external dependency sprawl (`objc2`, `objc2_metal`) is critical for a clean public API.

**Issues Observed:**
1. **`MTLSize` Tuple Sprawl**: `(grid_size: MTLSize, group_size: MTLSize)` appears in ~5 places, forcing `objc2_metal` imports at callsites.
2. **Trait Import Noise**: ~50 files import patterns like `use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, ...}`.
3. **`as_stage()` Naming**: Boxing allocates; `as_X()` conventionally implies cheap ref conversion. Should be `to_stage()` or `impl Into`.
4. **Verbose Qualifications**: `crate::types::MetalBuffer` instead of importing types at module top.

**Recommendations:**
- Create `DispatchConfig` semantic struct wrapping grid/group sizes (see code_review.md §9.1) remove it from foundry/mod.rs old tuple type!
- Create internal `types/prelude.rs` for objc2 trait imports
- Change `as_stage()` → `to_stage()` or use `impl From` for implicit conversion
- Use `impl Into<Stage>` on methods like `main_dyn()` for cleaner callsites

## 6. Recommendations & Next Steps

### Critical Fixes
- [ ] **Semantic Type Wrappers**: Prioritize the creation of `MetalDevice`, `MetalBuffer`, etc. wrappers to encapsulate `objc2`.

CAUTION: We need to make sure we're not breaking our legacy systems or changing those systems to use semantics as that would make our new commits very dirty and leaked from what we're working on. 

- [ ] **Compound Kernel Caching**: Modify `run_with_policy` to cache the specific `Fused` kernel instance based on the policy type `P` and kernel type `K`, rather than rebuilding it every frame.

### Improvements
- [ ] **Robust Macro Parsing**: Replace `type_str.contains("Tensor")` with proper `syn::Type` traversal in `metallic-macros` to handle aliases and paths robustly.
- [ ] **Include resolution**: Consider relying on `MTLCompileOptions` header search paths instead of manual manual inclusion if possible, or standardize the "virtual filesystem" approach.

### Polish
- [ ] **Unified Dispatch Logic**: Move the complex "command encoder" setup in `Foundry::dispatch` into a helper method on the new `MetalQueue` wrapper to centralize the logic.
