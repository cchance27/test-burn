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

---

## ⚠️ Risks & Regressions (Immediate Priority)

### 1. Embedding Kernel Polymorphism (Architectural Regression)
- **Issue:** Moved from a template-based `Policy` approach to explicit `run_embedding_core_f16` and `run_embedding_core_q8` functions.
- **Impact:** Breaks the "Single Location Policy" for quantization. Adding new quants (Q4_K, etc.) now requires modifying both Rust `step.rs` and Metal source, rather than just adding a Policy struct.
- **Task:** Refactor `embedding.metal` to restore template-based dispatch.

### 2. Batched Prefill for Multi-Turn (Performance Regression)
- **Issue:** Batched prefill is hard-disabled when `start_pos > 0`.
- **Reason:** Potential offset calculation bugs in kernels when history exists.
- **Impact:** User follow-up questions are processed sequentially (slower) instead of in parallel chunks.
- **Task:** Fix `position_offset` handling in batched kernels and re-enable.

### 3. SDPA Padding Logic
- **Issue:** `padded_m` only aligns to 32 if `m > 1`.
- **Risk:** GEMM kernels might perform out-of-bounds writes on `m=1` if they assume tile alignment without checking boundaries.
- **Task:** Audit GEMM write-back logic for `M < TileSize`.

### 4. Compound Kernel Cache Collisions (Critical)
- **Issue:** Dynamic `CompoundKernel` instances might share the same `TypeId` key in the pipeline cache if the ID mechanism is just `String` or `TypeId::of<String>`.
- **Risk:** Different kernels (e.g. F16 vs Q8) could overwrite each other in the cache, executing wrong code.
- **Task:** Implement robust content-based hashing (Source + Constants) for pipeline cache keys.

### 5. Quantization System Fragmentation (Critical Architectural Debt)
- **Issue:** The codebase currently splits quantization logic between a legacy `MetalPolicy` trait (for static generation) and a modern `QuantizationPolicy` trait (for runtime loading), with some kernels bypassing the trait system entirely to match on the `Quantization` enum directly.
- **Impact:** Bleeds policy infrastructure into kernel stages (e.g. `VectorizedDotStage` knows about `Q8` enum variant), violating open-closed principle and making it hard to add new quants without modifying every kernel stage.
- **Task:** Unify the system. Refactor `QuantizationPolicy` to implement (or provide) `MetalPolicy`. Update kernel stages to consume `Arc<dyn MetalPolicy>` trait objects for code generation instead of the `Quantization` enum. Re-enable `derive(MetalPolicy)` to reduce boilerplate for the code-gen aspect.

---

## 🛠️ Technical Debt (Next Sprint Tasks)

### 1. Dynamic Chat Templates
- **Debt:** Chat tokens (`<|im_start|>`, etc.) are currently hardcoded constants in `tokenizer.rs`.
- **Requirement:** Implement a template parser that respects the `tokenizer.chat_template` field from GGUF metadata.
- **Goal:** Support Llama 3, Phi, and Mistral without code changes.

### 2. Quantization Policy Leakage
- **Debt:** Some kernels are beginning to `match` on `Quantization` types internally.
- **Requirement:** Strengthen the separation of concerns. Kernels should consume generic interfaces (Macros/Templates); the "Policy" should define how to load/unpack data.

### 3. Memory Alignment & Stride
- **Debt:** Padded strides in SDPA scratch buffers are manually calculated in `step.rs`.
- **Requirement:** Centralize stride/padding logic into `metallic-foundry::compound::layout` to prevent silent corruption when tile sizes change.

### 4. Developer Experience (Kernel Macros)
- **Debt:** Manually calculating grid/threadgroup sizes in `step.rs` (e.g. `GridSize::new(...)`).
- **Requirement:** Implement `#[dispatch = "per_row"]` or similar presets in macros to auto-generate dispatch config.
- **Goal:** Reduce boilerplate and off-by-one errors in dispatch logic.
- **Validation:** Implement compile-time validation of Metal `emit()` strings to catch syntax errors early.

### 5. Fragile Macro Type-Detection
- **Issue:** `#[derive(KernelArgs)]` currently uses **string matching** (if `type_str.contains("TensorArg")`) to decide how to bind a buffer.
- **Risk:** This breaks if a developer uses a type alias (e.g., `type MyTensor = TensorArg;`) or a fully qualified path (`metallic::types::TensorArg`).
- **Requirement:** Refactor macros to perform proper `syn::Type` traversal for robust type identification.

### 6. Raw `objc2` Leaks in Public API
- **Issue:** The `Foundry` struct and many public methods still expose raw Metal types (e.g., `Retained<ProtocolObject<dyn MTLBuffer>>`).
- **Impact:** This couples the entire project (including the CLI and external crates) to specific versions of the `objc2` and `objc2-metal` crates.
- **Requirement:** Complete the "Semantic Wrapper" layer (`MetalBuffer`, `MetalDevice`, `MetalCommandQueue`) to encapsulate the raw pointers.

### 7. Memory Safety: Pool Lifetimes (Use-After-Free Risk)
- **Issue:** `Tensor<T, Pooled>` holds a reference to a Metal buffer but is not lifetime-bound to the `MemoryPool` that manages the offsets.
- **Risk:** If `MemoryPool::reset()` is called while a `Pooled` tensor is still alive, the tensor points to reclaimed memory (logical use-after-free).
- **Requirement:** Implement an "Arena" or "Generational" pattern where `Pooled` tensors hold a guard or generation ID that invalidates them if the pool is reset.

### 8. Macro Robustness (Type Detection)
- **Issue:** `#[derive(KernelArgs)]` uses fragile string matching (`type_str.contains("TensorArg")`) to detect buffer arguments.
- **Risk:** Fails for type aliases, fully qualified paths (`crate::types::TensorArg`), or newtypes.
- **Requirement:** Refactor macros to use `syn`'s type traversal to properly identify types implementing `KernelArg`.

### 9. Non-Standard Include Resolution
- **Issue:** `Foundry::load_kernel` uses custom logic to resolve `#include` directives by searching and stripping strings. This is a "virtual pre-processor" that might behave differently than the real Metal compiler.
- **Requirement:** Standardize on `MTLCompileOptions` with explicit header search paths or move toward a more robust Virtual File System (VFS).

### 10. Import Hygiene & Naming
- **Issue:** `MTLSize` tuples `(grid, group)` are used in several places, forcing callers to import Metal types.
- **Issue:** `as_stage()` is used for a method that boxes and allocates. Conventionally, `as_` should be a cheap reference conversion.
- **Requirement:** Use the `DispatchConfig` struct everywhere and rename `as_stage` -> `to_stage` or `into_stage`.

---

## 🔍 Quality Assurance / Debugging (From Phase 2 Status)

### 1. End-to-End Drift Investigation
- **Symptom:** Block-level parity is exact, but full-model generation shows `hidden_state` drift (~1.375 after 24 layers) and `logits` drift (~21.4).
- **Task:** Modify `dsl_vs_context_parity.rs` to perform a layer-by-layer residual stream comparison. Isolate exactly where the drift accumulates (e.g., is it uniform, or does one layer spike?).

### 2. RoPE & Position Index Verification
- **Task:** Verify that `position_offset` and `seq_len` are correctly propagated to the `RoPE` kernel in all layers, especially during multi-turn generation.

---

## 📅 Verified Backlog (Confirmed Valid 2026-01-21)

### 1. Dynamic Context Length (Fix 2048 Cap)
- **Status:** **Confirmed.** `CompiledModel::RUNTIME_MAX_SEQ_LEN_CAP` is hardcoded to `2048` in `executor.rs` (L43).
- **Impact:** Any generation requesting >2048 tokens will crash or be clamped, even if the model supports 32k.
- **Requirement:** Implement "Grow-on-demand" KV caches (e.g., paging or reallocation) to support full model context without allocating worst-case memory at startup.

### 2. SDPA Optimization (Tile Config)
- **Status:** **Confirmed.** `sdpa/step.rs` (L597) hardcodes `let tile_config = TileConfig::default();`.
- **Impact:** Suboptimal performance for prefill. Small `M` or odd-sized `SeqLen` might benefit from different tile shapes (e.g., 32x16 vs 32x32).
- **Task:** Use `TileConfig::auto_select(m, kv_seq_len)` logic (similar to `gemm` module).

### 3. Advanced Instrumentation (DSL Scopes)
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

### 5. Kernel Routing (MLX Parity)
- **Issue:** Native Foundry GEMV is slower than MLX kernels for specific decode shapes (e.g. `M=1, K=4864`).
- **Task:** Implement shape-based routing to select the optimal kernel backend (Foundry Native vs MLX-port) per operation, similar to the legacy engine.
- **Optimization:** Caching `CompoundKernel` construction in `run_with_policy` to avoid hot-path allocation overhead.