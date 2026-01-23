# Foundry Backend & CLI Frontend: Sprint Readiness Report

This document tracks the state of the Foundry backend transition, highlighting improvements made in the current uncommitted changes, regressions to fix, and architectural debt to address before 100% migration.

## ✅ Improvements & Completed Features

### 1. Interactive Multi-Turn Chat
- **TUI Integration:** Full-featured input box and event loop support for real-time chat.
- **CLI Multi-Turn:** Support for multiple `--prompts` arguments and continuous session state.
- **Session Persistence:** `ModelSession` now correctly tracks `current_pos` and updates global position indices across forward passes.
- **Log Management:** TUI now captures and redirects `tracing` logs to a dedicated pane, preventing terminal corruption.

### 2. Backend Correctness
- **SDPA Reliability:** Removed the incorrect `M=1` GEMV fast-path that was misusing quantized kernels for attention activations. Now defaults to the robust batched GEMM path.
- **Canonical Layout Support:** Optimized kernels (`ParallelProjectStage`, `FusedGemv`) now natively handle GGUF-standard blocked/canonical layouts.
- **Tokenizer Streaming:** Fixed multi-byte UTF-8 decoding across token boundaries and added support for GGUF byte tokens (`<0xXX>`).

### 3. Kernel Optimizations
- **Fused RMSNorm/Project:** Specialized `WarpWriteOutputNoResidualStage` to reduce register pressure in fused paths.
- **Quantized Embeddings:** Dedicated F16 and Q8_0 lookup kernels.
- **Unified Activations:** Activations are now a first-class `Activation` enum option (compiled into kernel variants) across GEMV/GEMM/SwiGLU paths, reducing the need for one-off Metal edits and improving composability.

### 4. Quantization Policy Architecture (Refactor)
- **Unified Policy System:** Replaced fragmented `Quantization` enum logic with a unified `Arc<dyn MetalPolicy>` trait object system across all kernel stages (`VectorizedDotStage`, `TileLoad`, etc.).
- **Runtime-Driven Selection:** Kernel policies are now strictly derived from runtime `TensorArg` dtypes, eliminating incorrect fallbacks caused by missing model hints.
- **Extensible Design:** New quantization schemes (Q4, INT8) can be added by implementing `MetalPolicy` without modifying core kernel generation logic.
- **Zero-Leakage Enforcement**: Successfully removed all explicit `short_name() == "f16"` or `is_q8` branching from kernel stages and execution steps.

### 5. Backend Reliability & Correctness
- **Q8 Inference Fixed:** Resolved garbage output regression by enforcing strict runtime dtype validation for GEMM prefill kernels (was defaulting to F16).
- **F16 Inference Fixed:** Resolved "Unsupported vector width" panic and added guardrails for scale argument binding in non-quantized policies.

### 6. Grow-on-Demand Infrastructure
Foundry no longer reserves the full model context memory at startup. Instead, it uses a **Geometric Doubling** strategy:
- **Initial Allocation:** Defaults to 2048 tokens (aligned to 128 for AMX performance).
- **On-Demand Growth:** When the sequence (prefill or decode) exceeds current capacity, all context-dependent buffers (KV caches, expanded K/V, scratch) are reallocated and doubled.
- **Max Bound:** Growth respects the `max_context_len` defined in model metadata or overridden by `METALLIC_MAX_CONTEXT_LEN`.

### 7. History Preservation & Stride Alignment
Growth is seamless and "live" during generation:
- **Head-by-Head Blit Copy:** Since KV layout is `[heads, capacity, dim]`, a change in `capacity` shifts the memory offset of every head. The system performs a high-speed GPU `blit_copy` to move existing tokens to their new head-relative offsets.
- **Dynamic Stride Synchronization:** Attention kernels pull the physical `capacity` from `FastBindings` for every dispatch. This ensures that head indexing remains perfectly aligned even if the buffer grew in the previous step.

### 8. Pluggable Eviction
The system uses an `EvictionPolicy` trait. While currently defaulting to `NoEviction` (fail-fast at max context), the architecture is ready for sliding-window or sparse attention policies by implementing the `evict()` method.

### 9. Post-Review Fixes (Implemented)
- **Fixed: capacity-dependent intermediates resize correctly:** context-dependent intermediates (`k_expanded`/`v_expanded`, `k_slice`/`v_slice`) are now explicitly reallocated during context growth (no “already bound” short-circuit).
- **Fixed: max-context override semantics:** `METALLIC_MAX_CONTEXT_LEN` / DSL caps now *cap* runtime max context and are clamped to the model max (never exceed model).
- **Fixed: RoPE scaling + growth:** RoPE cos/sin tables are sized to the *allocated capacity* and grow incrementally on demand, preserving history via GPU blit copies.
- **Fixed: storage mode defaults for performance:** hot intermediates now default to `StorageModePrivate` with `METALLIC_INTERMEDIATES_SHARED=1` as a debug/visibility override.
- **Fixed: test hygiene:** env-mutating context-config tests are now guarded with `serial_test`.
- **Fixed: avoid panic paths:** growth paths now return typed `MetalError` instead of panicking on missing bindings/buffers.

---

## ⚠️ Risks & Regressions (Immediate Priority)

** None Currently **

---

## 🛠️ Technical Debt (Next Sprint Tasks)

### 1. Dynamic Chat Templates
- **Debt:** Chat tokens (`<|im_start|>`, etc.) are currently hardcoded constants in `tokenizer.rs`.
- **Requirement:** Implement a template parser that respects the `tokenizer.chat_template` field from GGUF metadata.
- **Goal:** Support Llama 3, Phi, and Mistral without code changes.

### 3. Memory Alignment & Stride
- **Debt:** Padded strides in SDPA scratch buffers are manually calculated in `step.rs`.
- **Requirement:** Centralize stride/padding logic into `metallic-foundry::compound::layout` to prevent silent corruption when tile sizes change.

### 4. Developer Experience (Kernel Macros)
- **Debt:** Manually calculating grid/threadgroup sizes in `step.rs` (e.g. `GridSize::new(...)`).
- **Requirement:** Implement `#[dispatch = "per_row"]` or similar presets in macros to auto-generate dispatch config.
- **Goal:** Reduce boilerplate and off-by-one errors in dispatch logic.
- **Validation:** Implement compile-time validation of Metal `emit()` strings to catch syntax errors early.

### 5. Raw `objc2` Leaks in Public API
- **Issue:** The `Foundry` struct and many public methods still expose raw Metal types (e.g., `Retained<ProtocolObject<dyn MTLBuffer>>`).
- **Impact:** This couples the entire project (including the CLI and external crates) to specific versions of the `objc2` and `objc2-metal` crates.
- **Requirement:** Complete the "Semantic Wrapper" layer (`MetalBuffer`, `MetalDevice`, `MetalCommandQueue`) to encapsulate the raw pointers.

### 6. Memory Safety: Pool Lifetimes (Use-After-Free Risk)
- **Issue:** `Tensor<T, Pooled>` holds a reference to a Metal buffer but is not lifetime-bound to the `MemoryPool` that manages the offsets.
- **Risk:** If `MemoryPool::reset()` is called while a `Pooled` tensor is still alive, the tensor points to reclaimed memory (logical use-after-free).
- **Requirement:** Implement an "Arena" or "Generational" pattern where `Pooled` tensors hold a guard or generation ID that invalidates them if the pool is reset.

### 7. Non-Standard Include Resolution
- **Issue:** `Foundry::load_kernel` uses custom logic to resolve `#include` directives by searching and stripping strings. This is a "virtual pre-processor" that might behave differently than the real Metal compiler.
- **Requirement:** Standardize on `MTLCompileOptions` with explicit header search paths or move toward a more robust Virtual File System (VFS).

### 8. Import Hygiene & Naming
- **Issue:** `MTLSize` tuples `(grid, group)` are used in several places, forcing callers to import Metal types.
- **Issue:** `as_stage()` is used for a method that boxes and allocates. Conventionally, `as_` should be a cheap reference conversion.
- **Requirement:** Use the `DispatchConfig` struct everywhere and rename `as_stage` -> `to_stage` or `into_stage`.

### 9. Quantization Safety & Regression Testing
- **Goal:** Prevent future Q8/F16/OTHERS mismatches by strictly enforcing runtime policy.
- **Tasks:**
    - Enforce "runtime dtype is the source of truth" across **all** kernel routing (ignore/verify any DSL quant hints).
    - Add a regression test that loads a spec without `b_quant` and asserts Foundry still selects Q8 kernels when the bound tensor is Q8.

---

## 🔍 Quality Assurance / Debugging (From Phase 2 Status)

### 1. End-to-End Drift Investigation
- **Symptom:** Block-level parity is exact, but full-model generation shows `hidden_state` drift (~1.375 after 24 layers) and `logits` drift (~21.4).
- **Task:** Modify `dsl_vs_context_parity.rs` to perform a layer-by-layer residual stream comparison. Isolate exactly where the drift accumulates (e.g., is it uniform, or does one layer spike?).

### 2. RoPE & Position Index Verification
- **Task:** Verify that `position_offset` and `seq_len` are correctly propagated to the `RoPE` kernel in all layers, especially during multi-turn generation.

---

## 📅 Verified Backlog (Confirmed Valid 2026-01-21)

### 1. SDPA Optimization (Tile Config)
- **BLOCKER** Tile Config Auto-Tuning performance rework of auto_select is required, as switching to skinny_m (what auto_select selects), causes the sdpa path to be much slower dropping performance from 163tok/s to 150tok/s for Q8 as an example, meaning we need much better auto selection mechanics.
- **Status:** **Confirmed.** `sdpa/step.rs` (L597) hardcodes `let tile_config = TileConfig::default();`.
- **Impact:** Suboptimal performance for prefill. Small `M` or odd-sized `SeqLen` might benefit from different tile shapes (e.g., 32x16 vs 32x32).
- **Task:** Use `TileConfig::auto_select(m, kv_seq_len)` logic (similar to `gemm` module).

### 2. Advanced Instrumentation (DSL Scopes)
- **Status:** **Confirmed.** `executor.rs` loop (L528) iterates steps without pushing structured scopes.
- **Impact:** TUI metrics are brittle, relying on regex parsing of kernel names (e.g. `Gemm_...`) to guess if something is a "FeedForward" or "Attention" block.
- **Task:** Implement `foundry.push_scope("Block 0")` in the executor to enable robust, model-agnostic hierarchy.

---

## 🏗️ Architecture: Model Loading & Memory

### 1. `metallic-loader` Crate (Abstraction)
- **Goal:** Decouple file handling from the inference engine.
- **Features:**
    - Unified trait for model loading (GGUF, Safetensors).
    - Standardized weight layout normalization (e.g., ensuring `[Out, In]` layout regardless of source format).
    - Lazy loading support (memory-map first, upload to GPU on demand).

### 2. MTLHeap Memory Management
- **Goal:** Reduce peak VRAM usage and fragmentation.
- **Task:** Move from individual `MTLBuffer` allocations to a `MTLHeap`-based sub-allocator for activations and KV caches.
- **Benefit:** Allows memory aliasing (reusing memory for tensors that are never alive simultaneously) and explicit residency control.

### 3. Metal 3+ Fast Resource Loading
- **Goal:** Utilize Metal 3 sparse/fast resource loading APIs to stream weights directly to GPU memory, bypassing some CPU/Host-Ram overheads where possible.
- **Task:** Investigate `MTLIOCommandQueue` for weight loading in the `metallic-loader` crate.

### 4. GPU-Private Memory Promotion (Performance)
- **Issue:** Foundry currently uses `StorageModeShared` excessively for weights and intermediates.
- **Impact:** This misses out on the optimizations and higher cache bandwidth of `StorageModePrivate` (GPU-only) memory.
- **Task:** Implement a "Promotion" strategy where hot tensors (Activations, KV Cache, Frequently used weights) are migrated to `StorageModePrivate` to maximize bandwidth.

---

## 🚀 Performance Roadmap (Phase 4)

### 1. Memory Access Policy Infrastructure (NEW)
- **Goal:** Elevate vectorization from manual kernel optimization to a reusable system feature, mirroring the Quantization Policy.
- **Concept:** Introduce `LoadPolicy` / `IOComponent` abstraction.
    - Rust: Defines safe loading strategies based on runtime shapes (e.g., `Contiguous` vs `Strided`).
    - Metal: Provides component implementation (e.g., `VectorLoader4` using `half4`, `SimdGroupLoader`).
- **Impact:**
    - **DX:** Kernels write pure math (`Loader::load(idx)`), becoming agnostic to memory layout.
    - **Perf:** Write optimized vector loads once; apply to all kernels (RMSNorm, Softmax, etc.) automatically.
    - **Future-Proofing:** Simplifies adoption of new hardware intrinsics (e.g., AMX loads) without rewriting kernel logic.

### 2. Graph Capture / Indirect Command Buffers (Critical)
- **Goal:** Reduce CPU dispatch overhead from ~4.5ms/token to <0.1ms/token.
- **Strategy:** Implement `MTLIndirectCommandBuffer` or `Metal Graph Capture` for the `forward_step` loop.
- **Target:** Theoretical TPS ~200+.

### 3. Deep Kernel Fusion
- **Goal:** Reduce kernel launches per layer from ~18 to ~5.
- **Status:** `GEMV` can already fuse residual accumulation (via `beta=1`).
- **Missing:**
    - **Fused Add+RMSNorm:** A single kernel reading `x` and `residual`, adding them, and performing RMSNorm in one pass (avoiding memory round-trip of the added state).
    - **Fused KV Update:** Combine `RoPE` + `Cache Write` (and optionally `QKV Projection`) into a single kernel to minimize bandwidth.
    - **Fused MLP Down-Proj:** Evaluate fusing `Down` projection with the activation output of the previous SwiGLU step (likely high register pressure).
    - **Tail Fusion:** Fuse Final RMSNorm + Logits Projection + Sampling into one kernel.

### 4. M3 ISA Optimization (Q8 AMX)
- **Status:** F16 GEMM (`mma.metal`) uses `simdgroup_matrix`.
- **Goal:** Implement dedicated **Q8 Blockwise** dequantization-to-AMX path.
- **Impact:** Higher throughput for Q8 prefill batches by leveraging hardware matrix units directly from quantized data.

### 5. Tile Config Auto-Tuning (NEW)
- **Problem:** The current `TileConfig::auto_select()` heuristic is naive and can pick suboptimal configurations.
  - Example: Switching from `Default` (32×32×16) to `SkinnyM` (8×128×32) for SDPA caused a **7% regression** (165 → 153 tps).
  - The "Default" tile config was essentially a guess, not empirically optimized.
- **Industry Approaches:**

| System | Approach |
|--------|----------|
| **cuBLAS** | Runtime profiler picks from pre-tuned kernels based on (M,N,K) |
| **MLC-LLM** | Offline auto-tuning with search, stores winning configs per device |
| **FlashAttention** | Hardcoded configs per GPU architecture |
| **OneDNN** | JIT compilation with heuristic selection |

- **Proposed Strategy:**
  1. **Immediate:** Add `AttentionTileConfig` specifically tuned for attention patterns (M=1 decode, varying N)
  2. **Medium-term:** Improve `auto_select` to consider operation type, N/K ratio, and GPU family
  3. **Long-term:** Offline auto-tuning framework:
     - Run benchmark sweep over tile config space per op type
     - Store winning configs in JSON lookup table keyed by `(M_bucket, N_bucket, K_bucket, op_type, gpu_family)`
     - At runtime, look up best config from table

---

## Historical Items that we've completed, for posterity.
### 0. Batched Prefill for Multi-Turn (Performance Regression)
- **Status:** Fixed. Batched prefill re-enabled with correct `position_offset` handling and experimental warning log.
- **Impact:** Multi-turn generation now uses batched prefill for improved performance.

### 1. Quantization System Fragmentation (Critical Architectural Debt)
- **Issue:** The codebase currently splits quantization logic between a legacy `MetalPolicy` trait (for static generation) and a modern `QuantizationPolicy` trait (for runtime loading), with some kernels bypassing the trait system entirely to match on the `Quantization` enum directly.
- **Impact:** Bleeds policy infrastructure into kernel stages (e.g. `VectorizedDotStage` knows about `Q8` enum variant), violating open-closed principle and making it hard to add new quants without modifying every kernel stage.
- **Task:** Unify the system. Refactor `QuantizationPolicy` to implement (or provide) `MetalPolicy`. Update kernel stages to consume `Arc<dyn MetalPolicy>` trait objects for code generation instead of the `Quantization` enum. Re-enable `derive(MetalPolicy)` to reduce boilerplate for the code-gen aspect.

### 2. Q8 GEMM Policy Mis-Selection (Correctness Regression)
- **Symptom:** Q8 models (`*-Q8_0.gguf`) under `--engine foundry` produced garbage output (often token `0` / repeated `!`) or appeared to “hang” generating nonsense.
- **Root cause:** After refactoring quantization policies to runtime trait objects, GEMM kernel selection for the **prefill** path (M>1) relied on the serialized `MatMulStep.b_quant` hint. Some specs (e.g. `models/qwen25.json`) do not include `b_quant`, so it defaulted to **F16** even when the bound tensor was **Q8_0**, causing the prefill GEMM to interpret Q8 weights as F16 (silent corruption).
- **Fix (implemented):** Kernel policy selection is now derived from the **actual bound tensor dtype** (the `TensorArg` dtype) instead of serialized hints.
- **Fail-fast guardrails (implemented):**
  - Unsupported GGUF tensor dtypes now **panic** during load/mapping rather than silently falling back.
  - Foundry `prepare_bindings()` now validates that any U8-bound quantized weight has a companion `{name}_scales` binding with expected byte shape (and errors early if missing/malformed).

### 3. Embedding Kernel Polymorphism (Restored)
- **Issue:** Embedding kernel had regressed from template-based `Policy` approach to explicit `run_embedding_core_f16` and `run_embedding_core_q8` functions.
- **Fix (implemented):** Refactored `embedding.metal` to use unified `run_embedding_core<Policy>` template with `Policy::template load_weights<N>()` and conditional `Policy::HAS_SCALE` for quantization handling. `EmbeddingStage` now implements `Stage` trait with dynamic `Arc<dyn MetalPolicy>` selection based on tensor dtype at runtime.

### 4. Compound Kernel Cache Collisions (Resolved)
- **Previous issue:** Dynamic `CompoundKernel` instances could share the same cache key and overwrite each other.
- **Fix (implemented):** Added a global `KernelRegistry` with a content-based `source_hash` on `CompiledCompoundKernel`, and keyed pipeline caching on `(kernel_name_hash, source_hash, device_registry_id)` to prevent cross-kernel collisions.
- **Notes:** The registry is capacity-bounded and time-to-idle evicts unused entries to avoid unbounded growth in long-running sessions.

### 5. Unbounded Kernel Compile Caches (Resolved)
- **Previous issue:** Some kernel getters cached compiled variants by leaking them to satisfy `'static` return types, which could grow without bound.
- **Fix (implemented):** `CompiledCompoundKernel` no longer relies on `'static` leaks, and compiled kernel/pipeline caching is centralized in `KernelRegistry` with bounded capacity and time-to-idle eviction.
- **Follow-ups:** Consider a weighted cache (by estimated bytes) and plumbing config/env knobs for tuning `max_kernels`/`max_pipelines`/TTLs per workload.

### 6. Epsilon Usage in RMSNorm (COMPLETED)
- **Status:** Fixed. Epsilon is now configurable via `RmsNormParams.epsilon` and passed to the Metal kernel.
- **Impact:** RMSNorm kernels currently use a hardcoded `1e-6` epsilon value, which is not configurable.
- **Requirement:** Remove the epsilon parameter from the kernel and use the value from model or from the DSL.

### 7. Fragile Macro Type-Detection (COMPLETED)
- **Status:** Refactored to use `syn::Type` traversal. Handles qualified paths and `Option`/reference wrappers.
- **Issue:** `#[derive(KernelArgs)]` currently uses **string matching** (if `type_str.contains("TensorArg")`) to decide how to bind a buffer.
- **Risk:** This breaks if a developer uses a type alias (e.g., `type MyTensor = TensorArg;`) or a fully qualified path (`metallic::types::TensorArg`).
- **Requirement:** Refactor macros to perform proper `syn::Type` traversal for robust type identification.

### 8. Macro Robustness (COMPLETED)
- **Status:** Fixed see item #7 above.
- **Issue:** `#[derive(KernelArgs)]` uses fragile string matching (`type_str.contains("TensorArg")`) to detect buffer arguments.
- **Risk:** Fails for type aliases, fully qualified paths (`crate::types::TensorArg`), or newtypes.
- **Requirement:** Refactor macros to use `syn`'s type traversal to properly identify types implementing `KernelArg`.

### 9. SDPA Head-Stride Safety (Post-Padding Removal) (COMPLETED)
- **Context:** SDPA scratch (Scores/Probs) is head-major and uses `batch_stride_{c,d} = padded_m * kv_seq_len`.
- **Mitigation (implemented):** `padded_m` is now aligned to the GEMM tile M dimension for `m > 1` (decode keeps `m=1` to preserve throughput), and a regression test covers `m=17` to ensure head isolation.

### 10. Dynamic Context & History Preservation (COMPLETED)
- **Issue:** Foundry was hardcoded to 2048 tokens and would crash or lose history when growing history mid-generation.
- **Fix (implemented):**
    - Implemented geometric doubling (2x) of KV caches.
    - Added high-speed GPU `blit_copy` to preserve token history across reallocations.
    - Synchronized kernel strides with physical buffer capacity via `FastBindings`.
    - Integrated `sysinfo` for memory-aware pre-allocation.
