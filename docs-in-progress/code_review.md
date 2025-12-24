# Code Review: Foundry System Staged Changes

## Overview
47 files changed, ~7,200 new lines introducing the Foundry system, GEMV kernels, fusion builder, and TypeState patterns.

---

## 1. DX Improvements ✓

### What's Working Well
- **`TensorArg::from_tensor()`** - Unified tensor binding, auto-extracts buffer+offset
- **`foundry.run(&kernel)`** - One-liner dispatch using `dispatch_config()`
- **`KernelArgs` macro** - Auto-detects buffer vs bytes, auto-flushes inputs
- **Removed `BufferBinding`** - Simplified API surface

### Recommended Improvements

| Area | Current | Recommended |
|------|---------|-------------|
| **Tensor Construction** | `Tensor::<F16, Pooled>::new(foundry, dims, init)` | Add `foundry.alloc::<F16>(dims, init)` helper |
| **Error Context** | Generic `MetalError` | Add `thiserror` derives with `#[from]` for chaining |
| **Pool Access** | `foundry.get_resource::<MemoryPool>()` | Consider dedicated `foundry.pool()` accessor |

---

## 2. TypeState & NewType Patterns

### Current Implementation
```rust
// Storage TypeStates
pub struct Dedicated;  // Owns buffer
pub struct Pooled;     // Borrows from pool
pub struct View;       // Alias into another tensor

pub struct Tensor<T: TensorElement, S: StorageState = Dedicated>
```

### Improvements Needed

> [!IMPORTANT]
> **Lifetime Safety Gap**: `Tensor<T, Pooled>` can outlive `MemoryPool::reset()`, causing use-after-free.

**Options:**
1. **Phantom Lifetime**: `Tensor<'pool, T, S>` ties tensor to pool lifetime
2. **Arena Pattern**: `Tensor<T, Pooled>` holds `Rc<PoolGuard>` that prevents reset
3. **Generational IDs**: Pool uses generation counter; tensors check before access

**Recommended:** Option 2 (lowest friction, compile-time safety via RAII).

---

## 3. Potential Issues & Risks

### Memory Safety
| Location | Issue | Severity |
|----------|-------|----------|
| `pool.rs:41-42` | Manual `unsafe impl Send + Sync` | ⚠️ Medium |
| `tensor.rs:206` | Raw pointer transmute for upload | ⚠️ Medium |
| `tensor.rs:91-92` | Raw slice from GPU buffer | ⚠️ Medium |

**Action:** Centralize unsafe ops in `safety/*.rs` with documented invariants.

### Type System Gaps
- `TensorStorage<'a>` enum in `storage.rs` 
- `TensorInit::BorrowHost` creates Shared buffer but `Dedicated` implies Private
- `compute_strides()` is duplicated we should plan to eventually remove the legacy tensor system

### Macro Robustness
- `KernelArgs` type detection uses string matching (`"TensorArg"`, `"& Tensor"`)
- Fails for type aliases or qualified paths

**Fix:** Use syn's type traversal instead of string matching:
```rust
fn is_buffer_type(ty: &Type) -> bool {
    // Walk the Type enum properly
}
```

---

## 4. Idiomatic Rust Patterns

### Use `#[must_use]` Liberally
```rust
impl Foundry {
    #[must_use]
    pub fn dispatch(...) -> Result<(), MetalError>
    
    #[must_use] 
    pub fn run(...) -> Result<(), MetalError>
}
```


### Use `#[inline]` Liberally for hot paths
```rust
impl Foundry {
    #[inline]
    pub fn dispatch(...) -> Result<(), MetalError>
    
    #[inline] 
    pub fn run(...) -> Result<(), MetalError>
}
```

### Consider `NonZeroUsize` for Dimensions
```rust
// Current
pub struct Tensor<T, S> { dims: Vec<usize>, ... }

// Better: prevents 0-dim tensors at type level
pub struct Tensor<T, S> { dims: SmallVec<[NonZeroU32; 4]>, ... }
```

### Leverage `From`/`Into` Traits liberally used to replace custom conversion functions
```rust
// Instead of
pub fn from_tensor<K: KernelArg>(arg: &K) -> Self

// Use standard trait
impl<K: KernelArg> From<&K> for TensorArg
```

---

## 5. Maintenance Concerns

### Missing Tests
- No unit tests for `MemoryPool` edge cases (OOM, reset-after-use)
- No tests to confirm parity between legacy and Fondry Resource systems
- No integration tests for `Kernel` derive macro and other macros.
- No benchmark comparisons (legacy vs Foundry)

---

## 6. Quick Wins (Low Effort, High Value)

1. **Remove unused `TensorStorage` enum** in `storage.rs:21-26`
2. **Add `#[must_use]` to all Result-returning methods**
3. **Replace string-based type detection in macro with proper Type matching**
4. **Add `foundry.pool()` convenience accessor**
5. **Document unsafe blocks with `SAFETY:` comments**
6. **Implement `Debug` for `Tensor<T, S>`** for easier debugging

---

## 7. Language Features to Leverage

### Const Generics for Fixed Dimensions
```rust
// Future consideration for hot paths
pub struct TensorFixed<T, const N: usize, S> {
    dims: [usize; N],
    ...
}
```

### Sealed Traits for StorageState
```rust
mod private { pub trait Sealed {} }

pub trait StorageState: private::Sealed + 'static {}

impl private::Sealed for Dedicated {}
impl private::Sealed for Pooled {}
impl private::Sealed for View {}
```
This prevents external crates from implementing new storage states.

### GATs for Tensor Operations (Future)
```rust
trait TensorOps {
    type Output<S: StorageState>;
    fn reshape<S2: StorageState>(&self) -> Self::Output<S2>;
}
```

---

## Summary

| Category | Status | Priority |
|----------|--------|----------|
| DX Improvements | ✓ Good baseline | - |
| TypeState Safety | ⚠️ Pool lifetime gap | High |
| Unsafe Centralization | ⚠️ Scattered | Medium |
| Macro Robustness | ⚠️ String matching | Medium |
| Idiomatic Patterns | Could improve | Low |
| Maintenance/Divergence | Needs plan | High |

**Next Steps:**
1. Fix Pool/Tensor lifetime gap (choose arena or phantom approach)
2. Centralize unsafe code with documented invariants
3. Improve macro type detection
4. Unify all complex types into the types/typing module cleanly and replace them all.

---

## 8. **CRITICAL: Fusion & Kernel Signature Fragility**

> [!CAUTION]
> These issues can cause silent runtime failures or require manual synchronization across Rust and Metal files.

### 8.1 Hardcoded Policy-Specific Logic in fusion.rs

```rust
// fusion.rs:82-86 - FRAGILE
let init_code = if policy.struct_name() == "PolicyQ8" {
    "pp.matrix = matrix; pp.scales = scale_bytes; pp.weights_per_block = params->weights_per_block;"
} else {
    "pp.matrix = (const device half*)matrix;"
};
```

**Problems:**
- String comparison for type dispatch
- New policies require modifying this switch
- No compile-time verification

**Solution:** Move init logic to the `MetalPolicy` trait:
```rust
trait MetalPolicy {
    fn init_params_code(&self) -> &'static str;
}

impl MetalPolicy for PolicyQ8 {
    fn init_params_code(&self) -> &'static str {
        "pp.matrix = matrix; pp.scales = scale_bytes; ..."
    }
}
```

### 8.2 Raw Metal Source Strings in Rust

```rust
// fusion.rs:88-122 - FRAGILE
let full_body = format!(r#"
    using Policy = {policy_struct};
    ...
    run_simd_gemv_template<Policy, 1, 4, true, Epilogue>(...)
"#);
```

**Problems:**
- No syntax validation until runtime compilation
- Easy to introduce typos
- Template args (1, 4, true) are magic numbers

**Solution:** Generate Metal source via proc-macro from annotated Rust structs:
```rust
#[derive(MetalKernel)]
#[metal(template = "run_simd_gemv_template", heads = 1, cols_per_tg = 4)]
struct FusedGemv { ... }
```

### 8.4 Kernel Signature Divergence

**Rust (fusion.rs:124-140):**
```rust
.signature("gemv_fused_generated(
    const device uchar *matrix [[buffer(0)]],
    const device half *vector_x [[buffer(1)]],
    ...")
```

**Rust Struct (fusion.rs:24-46):**
```rust
#[arg(buffer = 0)]
pub matrix: TensorArg,
#[arg(buffer = 1)]
pub vector_x: TensorArg,
```

**Problems:**
- Buffer indices must match manually
- Type mismatch (e.g., `uchar*` vs `half*`) not caught
- Adding/reordering args requires TWO edits

**Solution:** Extend `KernelArgs` to generate the Metal signature:
```rust
impl FusedGemv {
    fn metal_signature() -> String {
        // Auto-generated from #[arg] attributes
    }
}
```

---

## 9. Import Hygiene & DX Patterns

> [!IMPORTANT]
> These patterns reduce external dependency sprawl and improve callsite ergonomics.

### 9.1 `DispatchConfig` Cleanup ✅ DONE

~~**Current:** `DispatchConfig` exists as a type alias in the wrong location.~~

**Implemented:**
```rust
// types/dispatch.rs
pub struct DispatchConfig {
    pub grid: GridSize,
    pub group: ThreadgroupSize,
}

pub struct GridSize { pub width: usize, pub height: usize, pub depth: usize }
pub struct ThreadgroupSize { pub width: usize, pub height: usize, pub depth: usize }

// Convenience constructors
impl GridSize { pub const fn d1(width: usize) -> Self { ... } }
impl DispatchConfig { pub const fn d1(grid_width: usize, group_width: usize) -> Self { ... } }

// Internal conversions
impl From<DispatchConfig> for (MTLSize, MTLSize) { ... }
impl From<(MTLSize, MTLSize)> for DispatchConfig { ... }
```

- [x] Move `DispatchConfig` from `foundry/mod.rs` to `types/dispatch.rs`
- [x] Convert from type alias to proper struct with `GridSize`/`ThreadgroupSize`
- [x] Add `From` impls for MTLSize conversion (internal use only)
- [x] Update all uses of raw `(MTLSize, MTLSize)` to use `DispatchConfig`
- [x] Remove `use objc2_metal::MTLSize` from callsites

### 9.2 Centralize `objc2_metal` Trait Imports

**Current:** ~50 scattered imports like:
```rust
use objc2_metal::{MTLCommandBuffer as _, MTLCommandEncoder as _, MTLCommandQueue as _, MTLComputeCommandEncoder as _};
```

**Solution:** Create a prelude in `types/prelude.rs`:
```rust
// types/prelude.rs (internal, not re-exported to consumers)
pub(crate) use objc2_metal::{
    MTLBlitCommandEncoder as _,
    MTLCommandBuffer as _,
    MTLCommandEncoder as _,
    MTLCommandQueue as _,
    MTLComputeCommandEncoder as _,
    MTLDevice as _,
    MTLBuffer as _,
};
```

Then internal modules use: `use crate::types::prelude::*;`

- [ ] Create `types/prelude.rs` with all trait imports
- [ ] Update internal modules to use prelude
- [ ] Document that consumers never need `objc2_metal` directly

### 9.5 General Pattern: Use `impl Into<T>` for Consumer-Friendly APIs

**Principle:** Heavy conversions should be implicit via `Into` bounds on the callee side, not explicit `as_X()`/`to_X()` calls on the caller side.

**When to use:**
- Conversion is **not allocation-heavy** OR the result is **cached anyway** → always prefer `impl Into`
- Even if heavy, if callers **must** do it → make it implicit to reduce boilerplate

**Examples:**
```rust
// Before: explicit conversion required
foundry.run_with_policy::<P, K>(&kernel.as_stage())?;

// After: implicit with Into bound
pub fn main_dyn(self, stage: impl Into<Box<dyn Stage>>) -> Self
// Callsite: .main_dyn(kernel)
```

**Pattern Summary:**
| Heaviness | Cached? | Recommendation |
|-----------|---------|----------------|
| Light     | -       | `impl Into` |
| Heavy     | Yes     | `impl Into` (cache hides cost) |
| Heavy     | No      | Explicit `to_X()` + document cost |

- [ ] Audit APIs for opportunities to add `impl Into` bounds
- [ ] Prefer callee-side conversions for cleaner consumer DX

### 9.4 Verbose `crate::types::` Qualifications

**Current:** `as_stage()` pattern:
```rust
.main_dyn(kernel.as_stage())  // foundry/mod.rs:277
fn as_stage(&self) -> Box<dyn Stage>  // 4 implementations
```

**Issue:** `as_X()` is conventionally for cheap ref-to-ref conversions. Boxing allocates.

**Recommendation:**
- `as_X()` → cheap, returns reference (`&T`)
- `to_X()` → cloning/allocation involved
- `into_X()` / `impl From` → ownership transfer

**Preferred Pattern:** Use `impl Into<Stage>` for cleaner DX:
```rust
// Current
.main_dyn(kernel.as_stage())

// Better: implicit conversion
pub fn main_dyn(self, stage: impl Into<Box<dyn Stage>>) -> Self
// Callsite becomes:
.main_dyn(&kernel)  // if From<&K> for Box<dyn Stage>
```

- [ ] Evaluate renaming `as_stage()` → `to_stage()` (allocates)
- [ ] Consider `impl From<&Kernel> for Box<dyn Stage>` for better DX
- [ ] Document pattern: heavy conversions → use `Into` bounds on callee

### 9.4 Verbose `crate::types::` Qualifications

**Current:** Full path qualifications throughout:
```rust
// Found in 16+ locations
let encoder_wrapper = crate::types::ComputeCommandEncoder(encoder.clone());
buffer: crate::types::MetalBuffer,
device: crate::types::MetalDevice,
```

**Solution:** Import the types at module level:
```rust
use crate::types::{MetalBuffer, MetalDevice, MetalQueue, ComputeCommandEncoder};
```

- [ ] Add proper imports at module tops in `foundry/`, `metals/`
- [ ] Remove inline `crate::types::` qualifications