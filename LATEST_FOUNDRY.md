# Foundry Backend & CLI Frontend: Sprint Readiness Report

This document tracks the state of the Foundry backend transition, highlighting improvements made in the current sprint, regressions to fix, and architectural debt to address before 100% migration.

## ✅ Improvements & Completed Features

### 1. Interactive Multi-Turn Chat
- **TUI Integration:** Full-featured input box and event loop support for real-time chat.
- **CLI Multi-Turn:** Supports multiple `--prompts` arguments for prompt-driven workflows (`inputs: ["messages", ...]`), reusing the same Foundry session between turns.
- **Session Persistence:** `ModelSession` now correctly tracks `current_pos` and updates global position indices across forward passes.
- **Log Management:** TUI now captures and redirects `tracing` logs to a dedicated pane, preventing terminal corruption.

### 2. Backend Correctness
- **SDPA Reliability:** Removed the incorrect `M=1` GEMV fast-path that was misusing quantized kernels for attention activations. Now defaults to the robust batched GEMM path.
- **Canonical Layout Support:** Optimized kernels (`ParallelProjectStage`, `FusedGemv`) now natively handle GGUF-standard blocked/canonical layouts.
- **Tokenizer Streaming:** Fixed multi-byte UTF-8 decoding across token boundaries and added support for GGUF byte tokens (`<0xXX>`).
- **Chat Template Parity:** Foundry now prefers the GGUF-provided `tokenizer.chat_template` for prompt formatting (avoids hardcoded system prompts that can change model behavior).

### 3. Sampling (Quality + Determinism)
- **Min‑P Support:** Added `min_p` to the Foundry sampler (relative cutoff: `min_p * max_prob`) to better match common sampler stacks.
- **Seed Stepping:** Sampling seed is advanced per generated token to avoid constant-seed repetition failure modes.
- **Penalties:** Repetition/presence/frequency penalties are supported (GPU-side) and default to `repeat_penalty=1.1`, `repeat_last_n=64`, `presence_penalty=0.0`, `frequency_penalty=0.0` in the default workflows/CLI. Throughput benchmarking disables these via the throughput script flags.
- **Correct Logits Row:** After batched prefill, sampling uses the last token’s logits row (fixes “stuck/repetitive” generations caused by sampling row 0).

### 3.5 Workflow Decode Batching (Throughput)
- **Batched Decode in Workflows:** Added `while_batched` to execute decode loops in chunks inside a single Metal capture and emit tokens at the batch boundary.
- **EOS Overshoot Trimming:** When EOS stop is enabled, output stops at EOS at the batch boundary without emitting EOS (overshoot tokens may still be computed but are not emitted).

### 3.6 Workflow Streaming + Async Windows (Latency/Throughput)
- **Channel Framework (v1):** Introduced `ChannelU32` (shared-memory ring buffer) to stream token ids without per-token scalar readback.
- **Streaming Ops:** Added `stream_init` + `stream_write_u32` and a sample workflow `crates/metallic-foundry/workflows/text_generation_stream_u32.json`.
- **Batched Streaming:** `while_batched.stream_channel` can emit tokens via `ChannelU32` instead of reading `token_var` each iteration.
- **Safe Overlap (Pipelined):** `while_batched.stream_async_poll` overlaps CPU token draining for batch *N* while GPU executes batch *N+1* (batch-granularity; not mid-command-buffer polling). Only enabled when EOS stopping is disabled (`METALLIC_IGNORE_EOS_STOP=1`); otherwise Foundry falls back to synchronous draining (warns).
- **Capture Ops:** Added `capture_begin`, `capture_end`, `capture_wait` plus `Value::CommandBuffer` for explicit async windows / command-buffer pipelining.
- **Debugging:** `METALLIC_DEBUG_WORKFLOW_OPS=1` logs workflow op execution; `METALLIC_DEBUG_STREAM_POLL=1` logs per-batch pipelined drain summaries.

### 4. Kernel Optimizations
- **Fused RMSNorm/Project:** Specialized `WarpWriteOutputNoResidualStage` to reduce register pressure in fused paths.
- **Quantized Embeddings:** Dedicated F16 and Q8_0 lookup kernels.
- **Unified Activations:** Activations are now a first-class `Activation` enum option (compiled into kernel variants) across GEMV/GEMM/SwiGLU paths, reducing the need for one-off Metal edits and improving composability.
- **FlashAttention (Metal / Foundry):**
  - Decode (M=1): fused RoPE → FlashAttention with streaming softmax + fused `P·V` accumulation.
  - Prefill (M>1): tiled FA1-style online softmax for `head_dim ∈ {64, 128}` with optional Split‑K prefill (2‑phase).
  - Tuning knobs: `METALLIC_FA_PREFILL_WARPS`, `METALLIC_FA_PREFILL_SPLIT_K`, `METALLIC_DISABLE_FA_PREFILL_SPLITK`, plus decode variant knobs.
  - Parity tests/sweeps exist to validate correctness and support tuning without editing kernels.

### 5. Quantization Policy Architecture (Refactor)
- **Unified Policy System:** Replaced fragmented `Quantization` enum logic with a unified `Arc<dyn MetalPolicy>` trait object system across all kernel stages (`VectorizedDotStage`, `TileLoad`, etc.).
- **Runtime-Driven Selection:** Kernel policies are now strictly derived from runtime `TensorArg` dtypes, eliminating incorrect fallbacks caused by missing model hints.
- **Variant Groundwork (NEW):** Added a centralized policy-variant resolver (`policy/variant.rs`) that records `source/storage/compute` dtype decisions and lossy-cast metadata in one place.
- **Native F32 Path (Gated):** Added `PolicyF32Native` + `policy_f32.metal` so dense F32 can be preserved under a non-legacy variant path, while default behavior remains legacy downcast-to-F16.
- **Future Compute Dtype Hook:** Quant policies now carry explicit compute-target metadata (`F16`/`BF16`/`F32`) for upcoming selector rollout; execution remains legacy until kernel coverage is expanded.
- **Kernel DType Helper Layer (NEW):** Added `metals/dtypes/runtime_types.metal` and migrated `gemv`/`rmsnorm` hot paths to helper-based load/store/cast APIs to reduce per-kernel type boilerplate and prepare compute/storage/accum rollout.
- **Extensible Design:** New quantization schemes (Q4, INT8) can be added by implementing `MetalPolicy` without modifying core kernel generation logic.
- **Zero-Leakage Enforcement**: Successfully removed all explicit `short_name() == "f16"` or `is_q8` branching from kernel stages and execution steps.
- **Mixed-Quant Fused Paths (Implemented):** Fused QKV and fused SwiGLU now support tuple-policy execution (e.g., `(q,k,v)` / `(gate,up)`) without decomposing into unfused fallback paths.
- **Mixed-Quant Kernel Varianting (Implemented):** Kernel cache/variant keys now include policy tuples for mixed quant combinations while preserving uniform fast paths.
- **Mixed-Quant Hardening (Deferred):** Explicit tuple allowlists + per-tuple parity gates are deferred until the per-variant benchmark/perf optimization campaign.

### 6. Backend Reliability & Correctness
- **Q8 Inference Fixed:** Resolved garbage output regression by enforcing strict runtime dtype validation for GEMM prefill kernels (was defaulting to F16).
- **F16 Inference Fixed:** Resolved "Unsupported vector width" panic and added guardrails for scale argument binding in non-quantized policies.
- **EOS Token Source:** `eos_token` is inferred from the tokenizer/model metadata (workflow/CLI no longer needs to hardcode it).

### 7. Grow-on-Demand Infrastructure
Foundry no longer reserves the full model context memory at startup. Instead, it uses a **Geometric Doubling** strategy:
- **Initial Allocation:** Defaults to 2048 tokens (aligned to 128 for AMX performance).
- **On-Demand Growth:** When the sequence (prefill or decode) exceeds current capacity, all context-dependent buffers (KV caches, expanded K/V, scratch) are reallocated and doubled.
- **Max Bound:** Growth respects the `max_context_len` defined in model metadata or overridden by `METALLIC_MAX_CONTEXT_LEN`.

### 8. History Preservation & Stride Alignment
Growth is seamless and "live" during generation:
- **Head-by-Head Blit Copy:** Since KV layout is `[heads, capacity, dim]`, a change in `capacity` shifts the memory offset of every head. The system performs a high-speed GPU `blit_copy` to move existing tokens to their new head-relative offsets.
- **Dynamic Stride Synchronization:** Attention kernels pull the physical `capacity` from `FastBindings` for every dispatch. This ensures that head indexing remains perfectly aligned even if the buffer grew in the previous step.

### 9. Pluggable Eviction
The system uses an `EvictionPolicy` trait. While currently defaulting to `NoEviction` (fail-fast at max context), the architecture is ready for sliding-window or sparse attention policies by implementing the `evict()` method.

### 10. Post-Review Fixes (Implemented)
- **Fixed: capacity-dependent intermediates resize correctly:** context-dependent intermediates (`k_expanded`/`v_expanded`, `k_slice`/`v_slice`) are now explicitly reallocated during context growth (no “already bound” short-circuit).
- **Fixed: max-context override semantics:** `METALLIC_MAX_CONTEXT_LEN` / DSL caps now *cap* runtime max context and are clamped to the model max (never exceed model).
- **Fixed: RoPE scaling + growth:** RoPE cos/sin tables are sized to the *allocated capacity* and grow incrementally on demand, preserving history via GPU blit copies.
- **Fixed: storage mode defaults for performance:** hot intermediates now default to `StorageModePrivate` with `METALLIC_INTERMEDIATES_SHARED=1` as a debug/visibility override.
- **Fixed: test hygiene:** env-mutating context-config tests are now guarded with `serial_test`.
- **Fixed: avoid panic paths:** growth paths now return typed `MetalError` instead of panicking on missing bindings/buffers.

### 11. Safety & Encapsulation
- **Safety & Encapsulation:** We centralize all `objc2` interaction and `unsafe` code within the `types` module system. This ensures that the rest of the Foundry codebase (`model`, `executor`, etc.) remains clean, safe Rust.

### 12. Format-Agnostic Loading & SDK Refactor (COMPLETED)
- **Crate Decoupling:** Created `metallic-sdk` to house generic `ModelLoader`, `LoadedModel`, and `ModelMetadata` traits, isolating file-format specifics from the core engine.
- **Source-Agnostic Foundry:** `metallic-foundry` now depends only on `metallic-sdk` for model ingestion. Direct `GGUFFile` and `GGUFMetadata` dependencies have been removed from the executor, policies, and tests.
- **Generic Tokenizer:** `BPETokenizer` now initializes from generic `ModelMetadata` using trait-based vocabulary and merge extraction (`tokenizer_tokens`, `tokenizer_merges`), supporting any source format that implements these traits.
- **Unified Policy Naming:** Renamed `gguf_tensor_name` to `source_tensor_name` across all policies and `MetalPolicyRuntime`, reflecting the engine's ability to map tensors from any backend format.
- **Agnostic Testing:** Refactored diagnostic and parity tests to use generic `MockModel` and `MapMetadata` implementations, allowing verification without concrete GGUF files.
- **Zero-Warning Workspace:** Performed a project-wide cleanup of unused imports and dead code resulting from the refactor, achieving a 100% clean build.

---

## ⚠️ Risks & Regressions (Immediate Priority)

** None Currently **

---

## 🛠️ Technical Debt (Next Sprint Tasks)

### 1) Typed Tensor Layouts (DX + Correctness)

- **Problem:** Refactors/new kernels repeatedly trip over silent layout mismatches (head-major vs token-major, row-major vs col-major, `m` vs `m_cap`, alignment/contiguity). These failures are often “plausible” (no crash) but cause large downstream drift (e.g. prefill prompt not respected) and are hard to debug.
- **Proposal:** Introduce *zero-cost* layout wrappers around `TensorArg` that encode invariants in the type system and centralize validation:
  - `TokenMajor` vs `HeadMajor` (and an explicit adapter for “token-major metadata but head-major packed contents”).
  - Shape-role views (e.g. `(B,M,DModel)`, `(H,M,D)`, KV cache `(H,capacity,D)`), plus row/col-major roles for GEMM-facing tensors.
  - Explicit contiguity/alignment guarantees (e.g. last-dim contiguous, 16B-aligned) as constructors that fail-fast.
  - Kernel entrypoints take typed views instead of raw `&TensorArg`, eliminating ad-hoc stride math in hot code.
- **Testing:** Add targeted parity/regression tests for each adapter (especially the “metadata != contents” cases) so layout regressions are caught before they become inference drift.

---

### 2) FlashAttention (FA2 + Quantized KV Cache)

- **Current state:** FA1-style attention is implemented for inference with dense `F16/F32` support:
  - Decode (M=1): fused kernel with streaming softmax + fused accumulation.
  - Prefill (M>1): tiled prefill + Split‑K prefill for large KV.
- **Next for FA2:** pipelined MMA / double-buffered KV staging (reduce stalls, increase utilization).
- **Correctness/DX:** keep SDPA routing explicit and fail-fast; expand coverage with targeted parity tests for new variants/paths.
- **Selector work:** auto-select per device + shape (kv_len, m) and expose env overrides for benchmarking.
- **KV-cache quantization (proposal):** add a policy-driven quantized KV-cache format (e.g. int8 + per-block scales) consumable by FlashAttention tile loaders for bandwidth/memory wins, gated by device/heuristics and protected by parity/regression tests.
- **Observed decode reality (Qwen q4_0 throughput runs):**
  - Across WIP iterations, decode `mma` generally leads `scalar` by ~`1-2%` in stable runs, with occasional first-run outliers.
  - Several decode-loop rewrites (extra prefetching/branch reshaping/full-tile transforms) showed noisy or negative absolute decode impact even when `mma-vs-scalar` spread improved.
  - Practical conclusion: decode M=1 path on this device/model is currently closer to memory/latency bound than compute bound; micro-tweaks have diminishing returns.
- **Execution policy for next FA2 work:**
  - Keep the best-known decode baseline pinned and avoid large decode churn without isolated evidence.
  - Prioritize higher-upside areas: prefill pipelining, selector quality, and KV bandwidth reduction.
  - Gate each FA2 change with a focused micro-benchmark first, then validate end-to-end throughput second.
  - Micro-benchmark command (decode-only, scalar vs mma kernel timing):
    - `METALLIC_FA_MICRO_ITERS=200 METALLIC_FA_MICRO_WARMUP=20 METALLIC_FA_MICRO_SCALAR=both cargo test -q --message-format=short -p metallic-foundry --test flashattention_decode_micro_benchmark benchmark_flashattention_decode_scalar_vs_mma_micro -- --ignored --exact --nocapture`
  - Decode diagnostics command (token-level stage timing + hot-step report):
    - `METALLIC_DEBUG_DECODE_STAGE_TIMING=1 RUST_LOG=metallic_foundry::model::executor::forward=info,metallic_foundry::workflow::ops::sample=info cargo run --release -- <model.gguf> "<prompt>" --temperature 0 --top-k 1 --top-p 1 --min-p 0 --repeat-penalty 1 --repeat-last-n 64 --max-tokens 64 --output-format text`
  - Diagnostic interpretation from current runs:
    - `FlashAttention` is typically ~`12-16%` of decode-token cost; biggest single hotspot is usually `MatMul (Unified)` (often `idx=146`), followed by FFN fused steps.
    - This confirms decode-side FA kernel tuning alone cannot produce large end-to-end decode gains without reducing non-FA hotspots.
- **Decode benchmark/test state (current):**
  - Added decode engine parity gate: `tests/flashattention_decode_engine_parity.rs` (scalar vs mma parity coverage on decode path).
  - Added decode micro-benchmark: `tests/flashattention_decode_micro_benchmark.rs` (ignored, env-driven warmup/iters/scalar mode, materialized timing).
  - Added opt-in decode timing env surface in `metallic-env`: `METALLIC_DEBUG_DECODE_STAGE_TIMING`.
- **FA2 decode status:**
  - MMA decode path and selector/bench/diagnostics are landed and usable.
  - Decode-side performance closeout is **in progress** (not marked complete) pending targeted reductions in the top non-FA decode hotspots.

---

## ⚠️ Risks & Regressions (Immediate Priority)

### 1. RoPE & Position Index Verification
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

### 4. M3 ISA Optimization (Qx AMX)
- **Status:** F16 GEMM (`mma.metal`) uses `simdgroup_matrix`.
- **Goal:** Implement dedicated **Q Blockwise** dequantization-to-AMX path.
- **Impact:** Higher throughput for Q prefill batches by leveraging hardware matrix units directly from quantized data.

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

### 6. Zero-Copy Polling (Latency Optimization) (Planned)
- **Problem:** Decode batching improves throughput but hurts streaming latency (tokens appear in chunks). Synchronous GPU execution blocks the CPU from reporting progress mid-batch.
- **Solution:** Implement **zero-copy polling** on Apple Silicon (UMA):
  - Allocate a `StorageModeShared` ring buffer for token output (and optionally timing/metadata).
  - Dispatch large decode windows asynchronously (e.g. 64–512 steps) via workflow batching/capture.
  - CPU polls the ring buffer for new tokens *while the GPU is still executing* (or reacts via command-buffer completion handlers for chunked flush).
  - Because memory is coherent, the CPU can observe tokens as soon as the GPU writes them, enabling **high throughput (batching) AND low latency (streaming)**.
  - Requires a workflow-visible output buffer contract (so non-LLM workflows can stream intermediate artifacts too).

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

### 11. Dynamic Chat Templates (COMPLETED)
- **Status:** Integrated `minijinja` for dynamic template rendering.
- **Feature:** Automatically parses and applies `tokenizer.chat_template` from GGUF metadata.
- **Flexibility:** Supports DSL-level overrides via the `chat_template` field in `ModelSpec`.
- **Correctness:** Correctly injects `bos_token` and `eos_token` into the template context, enabling parity with Llama 3, Mistral, and Qwen 2.5.
- **Preservation:** Configured for strict whitespace and trailing newline preservation to match model-specific requirements.

### 12. Centralized Memory Alignment & Stride (COMPLETED)
- **Status:** Fixed. Centralized all tile-aware stride and padding logic into `metallic-foundry::compound::layout`.
- **Impact:** Ensures consistency between GEMM and Softmax kernels in the SDPA path. Scratch buffers (`Scores`, `Probs`) now use `TiledLayout` for robust head-major indexing, eliminating manual math in `step.rs` and preventing corruption when switching to skinny-M or other tile configurations.

### 13. Developer Experience (Kernel Macros) (COMPLETED)
- **Status:** Implemented `#[kernel(dispatch = "...")]` presets and automatic `CompiledStep` generation.
- **Feature:** Macros now support `per_element`, `per_row`, `warp_per_row`, and `per_element_vec` presets, eliminating manual `dispatch_config` boilerplate for most kernels.
- **Feature:** `#[arg(scale_for = "...")]` adds automatic support for derived scales in quantized kernels.
- **Feature:** `#[derive(Kernel)]` now generates both `Step` and `CompiledStep` implementations, with `Step::execute` automatically routing through the optimized `CompiledStep` path.

### 14. Safety & Type Encapsulation (COMPLETED)
- **Status:** **Completed.** All direct `objc2` and `objc2-metal` dependencies are now strictly isolated within `src/types/mod.rs`.
- **Feature:** **Safe Buffer Wrappers:** Extended `MetalBuffer` with robust safe APIs (`read_to_vec`, `write_via_slice`, `copy_from_slice`), enabling the removal of `unsafe` blocks from high-level application logic.
- **Feature:** **Unsafe Elimination:** application logic (`model`, `executor`, `tokenizer`, `policy`) is now 100% safe code. Raw pointer operations are restricted to the FFI boundary layer.
- **Feature:** **Error Standardization:** All Metal type wrappers now return `Result<T, MetalError>`, providing proper error propagation instead of fragile `Option` unwrapping.
- **Impact:** Decouples the codebase from specific `objc2` versions and prevents memory safety regressions in future feature work.

### 15. Memory Safety: Pool Lifetimes (Generational Invalidation) (COMPLETED)
- **Status:** Fixed. Implemented an "Arena" generational pattern for `MemoryPool`.
- **Feature:** `MemoryPool` tracks a global `current_generation` counter.
- **Feature:** `Tensor<T, Pooled>` and `Tensor<T, View>` hold a generation ID and a reference to the pool's counter.
- **Feature:** `Tensor::buffer()` and other accessors perform a runtime check. If the pool has been reset (incrementing the generation), any attempt to access the stale tensor results in a panic or a `MetalError::UseAfterFree`, preventing logical use-after-free bugs in the Foundry engine.

### 16. Import Hygiene & Naming (COMPLETED)
- **Status:** Fixed.
- **Refactor:** Renamed `Kernel::as_stage` to `to_stage` to reflect that it allocates/boxes.
- **Hygiene:** Removed public `impl From<DispatchConfig> for (MTLSize, MTLSize)` to prevent `objc2_metal` types from leaking into consumer code. Replaced with crate-private `as_mtl_size()` methods.

### 17. Non-Standard Include Resolution (Standardized)
- **Status:** **Fixed.** Standardized include resolution in `compile_pipeline` to strip local `#include` directives and warn, enforcing explicit dependency declaration via `Kernel::includes()`. Refactored `CompoundKernel` to stop emitting includes in source strings, resolving double-handling issues.
- **Issue:** `Foundry::load_kernel` uses custom logic to resolve `#include` directives by searching and stripping strings. This is a "virtual pre-processor" that might behave differently than the real Metal compiler.
- **Requirement:** Standardize on `MTLCompileOptions` with explicit header search paths or move toward a more robust Virtual File System (VFS).

### 18. Quantization Safety & Regression Testing (COMPLETED)
- **Goal:** Prevent future Q8/F16/OTHERS mismatches by strictly enforcing runtime policy.
- **Tasks:**
    - Enforce "runtime dtype is the source of truth" across **all** kernel routing (ignore/verify any DSL quant hints).
    - Add a regression test that loads a spec without `b_quant` and asserts Foundry still selects Q8 kernels when the bound tensor is Q8.
- **Verification:** Added `test_gemm_v2_step_runtime_policy_selection_q8` in `gemm_v2_parity.rs` which successfully verifies that Foundry selects Q8 kernels for Q8-bound tensors even without DSL hints.

### 19. Workflow DX (Foundry)
- **Message-Driven Workflows:** Restored `multiturn_chat*.json` workflows with `format_chat` + `tokenize` inside the workflow graph.
- **Schema Alias:** Workflows accept both `steps` and `phases` keys (to avoid confusion with model DSL “steps”).
- **Debug Knobs:** `METALLIC_DEBUG_TOKENIZE` and `METALLIC_DEBUG_CHAT_TEMPLATE` print formatted prompts + token heads for parity debugging.
- **Multi-Turn Delta Mode:** `format_chat(mode="delta")` + `prefill(mode="delta")` enable KV reuse across turns without replaying the full prompt. Delta mode supports both full-history and delta-only message inputs (TUI uses deltas after the first turn).
- **Runner Cache + Memoization:** `WorkflowRunner` caches compiled workflows/ops and invalidates on workflow spec changes; ops can declare memoization specs for caching pure outputs (e.g. tokenization).

### 20. Mixed-Quant Fused Kernel Performance Recovery (COMPLETED CORE / HARDENING DEFERRED)
- **Completed:** tuple-policy fused execution landed for `FusedQkv` (`(q,k,v)`) and fused SwiGLU/FFN (`(gate,up)`), including mixed-policy kernel generation and dispatch.
- **Completed:** tuple-aware kernel cache keying landed for fused mixed-policy variants with preserved uniform fast-path variants.
- **Completed:** focused mixed-policy fused benchmarks were added (`fused_mixed_policy_benchmark.rs`) with materialized iteration runs and runtime progress logging for `--nocapture` workflows.
- **Deferred (intentional):** explicit tuple allowlists and strict per-tuple parity gate enforcement are postponed until the dedicated per-variant benchmarking and kernel-level perf optimization campaign.

### 21. Dynamic Runtime DType Kernel Support (accum, storage, compute) (COMPLETED)
- **Completed:** model-level mixed-GGUF validation matrix runner landed (`crates/metallic-foundry/tests/mixed_gguf_validation_matrix.rs`) covering prefill/decode transitions, long-context prefill chunking, and GQA group-size variation.
- **Completed:** dense F32 RoPE→FlashAttention parity/perf sweep landed (`crates/metallic-foundry/tests/flashattention_rope_decode_f32_parity_perf.rs`) for `head_dim={64,128}` and short/long KV lengths with parity + perf thresholds.
- **Completed:** authoritative dtype support matrix published in `LATEST_FOUNDRY.md` and `docs/QUANTIZATION.md`.
- **Completed:** CLI/config runtime dtype surface landed:
  - CLI flags: `--compute-dtype`, `--accum-dtype`, `--foundry-env KEY=VALUE`
  - Programmatic API: `FoundryConfig::{with_compute_dtype, with_accum_dtype, with_env_override}` via `Foundry::new_with_config(...)`
- **Completed:** store-helper regression guard coverage landed for strided vs contiguous output stores (`runtime_store_helpers_regression.rs`).
- **Completed:** helper-path micro-opts landed in `runtime_types.metal`:
  - conversion-elision fast path in `metallic_store_output` when `AccumT` and output storage are both FP16
  - contiguous auto-routing in indexed helpers (`metallic_store_output2/4` -> `*_contig` when indices are contiguous)
- **Completed:** local perf non-regression guard landed (`runtime_store_helpers_perf_guard.rs`) for helper-path changes.

Local narrow gates (manual; no CI dependency):

1. `cargo test -q --message-format=short -p metallic-foundry --test mixed_gguf_validation_matrix mixed_gguf_validation_matrix_prefill_decode_long_context_gqa -- --ignored --exact --nocapture`
2. `cargo test -q --message-format=short -p metallic-foundry --test flashattention_rope_decode_f32_parity_perf flashattention_rope_decode_f32_parity_and_perf_sweep -- --ignored --exact --nocapture`
3. `cargo test -q --message-format=short -p metallic-foundry --test runtime_store_helpers_regression runtime_store_helpers_respect_strided_and_contiguous_indices -- --exact`
4. `cargo test -q --message-format=short -p metallic-foundry --test runtime_store_helpers_perf_guard runtime_store_helpers_perf_non_regression -- --ignored --exact --nocapture`
