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
