# Foundry Kernel Infrastructure — Roadmap & Status

**Last Updated:** 2025-12-30

---

## 🎯 Roadmap: Future Kernel Fusion Improvements

### Phase 3: Advanced Fusion DX (Next Up)

| Priority | Feature | Description | Status |
|----------|---------|-------------|--------|
| **3.1** | Prologue Abstraction | Full `Stage` trait for pre-main computation (RMSNorm, etc.) | 📋 Planning |
| ~~3.x~~ | ~~Inline Hook/Epilogue~~ | ~~Dropped: adds fragility, breaks reusability~~ | ❌ Dropped |
| **3.2** | Template Validation | Parse `emit()` strings at macro time to catch Metal syntax errors | ⏳ Future |
| **3.3** | Dispatch Presets | `#[dispatch = "per_row"]` instead of manual grid math | ⏳ Future |
| **3.4** | Include Dedup | Deduplicate includes at macro time (currently runtime HashSet) | ⏳ Future |

### Vision: Compose Any Fusion at Macro Level

```
┌──────────────────────────────────────────────────────────────────┐
│                    #[derive(FusedKernel)]                        │
├─────────────┬─────────────────────────┬─────────────────────────┤
│  Prologue   │         Main            │       Epilogue          │
│ ───────────-│──────────────────────── │ ─────────────────────── │
│ • RMSNorm   │ • GEMV (F16/Q8/Q4)     │ • SwiGLU/GELU           │
│ • Embedding │ • Attention             │ • BiasAdd               │
│ • RoPE      │ • Convolution           │ • SIMD Reduce           │
│ • Custom    │ • Scatter/Gather        │ • Softmax               │
└─────────────┴─────────────────────────┴─────────────────────────┘
```

### Suggested Developer Next Steps

1. **Implement Prologue Stage abstraction** (Priority 3.1)
   - Add `GemvSimdPrologue` trait (includes, buffer_args, emit)
   - Add `NoPrologue` default + `#[derive(GemvSimdPrologue)]` macro
   - Modify `GemvSimdMainStage<P, C, H, E>` with prologue generic
   - Create `RmsnormPrologue` type, migrate existing fused kernels
   - Simplify hooks to pure policy selection (no preamble)

2. **Add dispatch presets** (Priority 3.3)
   - `#[dispatch = "per_element(total)]"` → 1D grid over total elements
   - `#[dispatch = "per_row(rows, cols)]"` → 1 threadgroup per row
   - Reduces boilerplate in `dispatch_config()`

3. **Metal template validation** (Priority 3.2)
   - Parse `emit()` strings at macro time
   - Catch Metal syntax errors before GPU compile

---

## Status Update (2025-12-29)

### Latest Changes: Kernel Consolidation

### Completed
1. **ALWAYS_INLINE standardized** — Added `inline` keyword across 7 Metal files
2. **Removed duplicate files** — `compat/always_inline_legacy.metal`, `matmul_gemv/simd_template.metal`, `matmul_gemv/epilogues.{rs,metal}`
3. **MetalStruct derive** — `QkvFusedParams`/`Q2FusedParams` use `#[derive(MetalStruct)]` with proper field name mapping
4. **Struct injection via macro** — Enhanced `GemvSimdConfig` with `struct_defs_type` attribute, removed hardcoded structs from `simd_common.metal`
5. **SwiGLU epilogue moved** — From `matmul_gemv/epilogues` to `swiglu/mod.rs`, Metal code in `swiglu/swiglu.metal`
6. **Fused GEMV reorganized** — `matmul_gemv/fused/{qkv.rs, swiglu.rs}` with collocated Params, Args, Step, dispatch

### Verified
- SwiGLU parity tests: 8/8 PASS
- Full block parity test: 9/9 steps PASS

---

## Next Steps: Kernel Infrastructure Improvements

### Priority 1: SIMD Reduce in Epilogue Macro — ✅ DONE
**Goal:** Auto-generate the simd_shuffle_xor reduction ladder in epilogues instead of manual Metal code.

**Status:** Added `simd_reduce`, `simd_reduce_from`, `simd_reduce_to`, `simd_reduce_op` attributes to `#[derive(Epilogue)]`. Supports Add, Max, Min operations with configurable reduction levels.

### Priority 2: Auto Buffer Indices in KernelArgs — ✅ DONE
**Goal:** Eliminate manual `#[arg(buffer = N)]` counting.

**Status:** All struct fields are now buffer args by default. Buffer indices auto-increment from 0. Use `#[arg(skip)]` to exclude, `#[arg(output)]` to mark outputs. Explicit `#[arg(buffer = N)]` still works if needed.

### Priority 3: DynamicParams Derive — ✅ ALREADY DONE
**Goal:** Eliminate boilerplate `FooParams` → `FooParamsResolved` pattern.

**Status:** `MetalStruct` derive automatically generates `*Resolved` types and `Resolvable` impl when struct contains `DynamicValue<T>` fields. No separate derive needed.

### Priority 4: CodeBuilder for Stage.emit() — ✅ DONE
**Goal:** Prevent variable naming bugs in compound kernel codegen.

**Status:** Added strongly-typed `CodeBuilder` with `MetalType`, `MetalVar`, `SimdReduceConfig`, `ReduceOp`. 10 unit tests pass. See `compound/code_builder.rs`.

### Priority 5: Unified GemvKernel Derive — ✅ DONE
**Goal:** Consolidate Hook + Epilogue + Config into single derive.

**Status:** Added `#[derive(GemvKernel)]` that combines config attrs + `hook = Type` + `epilogue = Type`. Generates `GemvSimdConfig` impl + `main_stage()` helper. Validates array lengths match heads at macro time.

### Lower Priority Items
- **include dedup at macro time** — Currently runtime HashSet
- ~~**struct_defs dedup** — Multiple stages could inject same struct~~ ✅ Fixed: `GemvSimdConfig` now injects `GemvParams` via `struct_defs()`, compound kernels aggregate all stage struct_defs
- **Metal syntax validation** — Validate `emit` templates at compile time
- **Dispatch presets** — `#[dispatch = "per_element"]` for common patterns
- ~~**Hardcoded Metal struct defs**~~ ✅ Fixed: Removed from all Metal files (`kv_cache`, `embedding`, `rope`, etc.) — now injected via `MetalStruct` derives with `*Resolved` naming

## Current Status

**Goal:** Achieve numerical parity between DSL-based inference and Legacy `Context<T>` implementation.

**Progress:** 🟢 Token-level parity achieved for seeded sampling - DSL matches Legacy for autoregressive generation when using the same prompt + seed (top-k/top-p sampling).

---

## Root Causes & Fixes (2025-12-26 → 2025-12-27)

**`GemvCanonicalStep` was completely missing!**

The spec used `"op": "GemvCanonical"` for attention Q/K/V projections, but there was no corresponding `Step` implementation registered with typetag. This caused the kernel to receive incorrect parameters and output zeros.

### Fixes Applied

1. **Added `GemvCanonicalStep`** to [`step.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/step.rs)
   - Registered with `#[typetag::serde(name = "GemvCanonical")]`
   - Added `resolve_gemv_canonical_params` to infer N/K from tensor dims

2. **Added `step = false`** to [`canonical.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/canonical.rs)
   - Prevents Kernel macro from auto-generating duplicate Step

### Additional fixes (2025-12-27)

1. **Canonical weight swizzle was wrong for non-square weights (K/V, FFN)**  
   - Foundry's `bind_gguf_tensor_canonical` was implicitly guessing `(K,N)` from GGUF dims, which is ambiguous when GGUF dims order is not consistent.  
   - Fix: pass expected `(K,N)` from the model arch per weight kind (attn_k/v, ffn_down, etc) and swizzle accordingly.

2. **SDPA stride/layout mismatch (head-major vs seq-major)**  
   - `KvRearrange` and `RepeatKvHeads` produce head-major packed buffers `[heads, seq, head_dim]`, but `SdpaStep` was interpreting Q as seq-major and K/V with a max-seq stride.  
   - Fixes:
     - Treat Q as head-major in `SdpaStep`.
     - Remove `max_seq_len` from the `Sdpa` step in `qwen25.json` so K/V are interpreted as packed (stride=`kv_seq_len * head_dim`), matching `RepeatKvHeads` output.

3. **GemvCanonical param inference hardened**  
   - Canonical GEMV now infers `K/blocks_per_k` from the canonical weight buffer length (and validates it), avoiding mistakes when `vector_x` is a larger preallocated workspace.

4. **Performance: Foundry `Repeat` debug spam gated**  
   - Layer/substep tensor dumps in `Repeat` are now gated behind `METALLIC_FOUNDRY_TRACE=1` to avoid `eprintln!` on the hot path.

5. **Seeded generation drift (sampling parity) due to fused-kernel differences**  
   - Even when block-level parity was “close”, top-k/top-p sampling is extremely sensitive to small logits differences, and the DSL path diverged immediately with a fixed seed.  
   - Fixes:
     - Added Foundry wrappers for legacy fused FP16 canonical kernels:
       - `gemv_f16_canonical_qkv_rmsnorm_f16`
       - `gemv_f16_canonical_swiglu_rmsnorm_f16`
     - Added a **CompoundKernel-based fused path** (now the default for Foundry) to move toward Foundry-native fusion with good DX while retaining parity/performance characteristics.
     - Updated Foundry Metal compile language version to `Version4_0` to match legacy kernel compilation (required for matmul_gemv sources).
     - Updated `qwen25.json` to use fused steps (`QkvF16CanonicalFusedRmsnorm`, `SwiGluF16CanonicalFusedRmsnorm`) and to fuse residual adds via GEMV (`beta=1.0`) instead of separate `ElemwiseAdd` steps.

---

## Parity Test Results (Latest)

| Component | Status | Notes |
|-----------|--------|-------|
| Embedding | ✅ PASS | max_diff=0.0 (exact match) |
| K projection | ✅ PASS | matches Legacy |
| V projection | ✅ PASS | matches Legacy |
| Q after RoPE | ✅ PASS | matches Legacy |
| K expanded | ✅ PASS | matches Legacy |
| V expanded | ✅ PASS | matches Legacy |
| Attention output | ✅ PASS | matches Legacy |
| Output projection | ✅ PASS | matches Legacy |
| FFN gate | ✅ PASS | - |
| FFN output | ✅ PASS | - |
| Block output | ✅ PASS | - |

---

## Generation Test Results

- Incremental generation test now produces coherent text output (no `NaN` propagation observed in hidden/logits in the debug run).
- New test: seeded generation token parity now passes (`dsl_vs_context_generation_seed_parity`).

---

## Key files changed this session

- [`src/metals/gemv/step.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/step.rs) - Added `GemvCanonicalStep`
- [`src/metals/gemv/canonical.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/gemv/canonical.rs) - Added `step = false`
- [`src/foundry/model/executor.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/model/executor.rs) - Fixed N/K swap in canonical weight swizzling
- [`tests/dsl_vs_context_parity.rs`](file:///Volumes/2TB/test-burn/crates/metallic/tests/dsl_vs_context_parity.rs) - Added `test_full_block_step_parity`
 - [`src/metals/sdpa/step.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/sdpa/step.rs) - Fix Q head-major view for SDPA
 - [`src/foundry/spec/qwen25.json`](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/spec/qwen25.json) - SDPA no longer uses `max_seq_len` for K/V stride
 - [`src/foundry/spec/repeat.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/spec/repeat.rs) - Gate debug prints behind `METALLIC_FOUNDRY_TRACE`
 - [`src/metals/matmul_gemv_fused.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/metals/matmul_gemv_fused.rs) - Foundry wrappers + DSL steps for legacy fused matmul_gemv kernels
 - [`src/foundry/mod.rs`](file:///Volumes/2TB/test-burn/crates/metallic/src/foundry/mod.rs) - Match legacy Metal language version (`Version4_0`)
 - [`tests/dsl_vs_context_generation_seed_parity.rs`](file:///Volumes/2TB/test-burn/crates/metallic/tests/dsl_vs_context_generation_seed_parity.rs) - Seeded (top-k/top-p) generation parity test

---

## Next Steps

1. Re-run the greedy generation parity test (`dsl_vs_context_generation_greedy_parity`) across several tokens to confirm parity holds beyond layer 0 in longer decode runs.
2. Remove/relax ignored status on the parity tests once model path / env handling is standardized.

---

## Quick Commands

```bash
# Block-level parity test
cargo test -p metallic --test dsl_vs_context_parity test_full_block_step_parity -- --nocapture --ignored

# Generation quality test
cargo test -p metallic --test dsl_generation -- --nocapture --ignored

# Legacy CLI (known working)
cargo run --release -p metallic_cli -q -- ./models/qwen2.5-coder-0.5b-instruct-fp16.gguf "create a short js fibonacci function"
```
