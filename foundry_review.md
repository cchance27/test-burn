# Foundry Inference Review Notes

This file started as an investigation into the “gibberish output” regression when adding `m>1` prefill. As of **2026-01-09**, the primary correctness issue is fixed and Foundry performance is back to parity (or better) for Qwen2.5-0.5B FP16.

## 1) FIXED: OOB read in GEMM for partial tiles

**Symptom:** gibberish output like `,!!!!!!!!!!!!!!!!!!!%` during inference after enabling `m>1` prefill.

**Root cause:** the MMA GEMM loop was unconditionally using `load_unsafe()` for A/B tile loads, which assumes a full tile. With small `M` (e.g. `M=4`) and `TileConfig::Default` (`BM=32`), this read past the end of the activation buffer.

**Fix implemented:** `crates/metallic/src/metals/mma/stages.rs` now uses `load_safe(...)` when `tgp_bm != GEMM_BM` or `tgp_bn != GEMM_BN`, while keeping `load_unsafe()` for full tiles for performance.

## 2) FIXED: SDPA decode overhead (non-kernel)

The decode regression was not only kernel math. It also came from avoidable per-step overhead:

- **Per-call constant upload + wait**: SDPA was uploading a scalar `half(1.0)` via a blit command buffer and waiting on it every step. This is now replaced by a cached GPU buffer (`crates/metallic/src/foundry/constants.rs`) used by SDPA.
- **Per-call scratch allocations**: SDPA was allocating intermediate `scores`/`probs` buffers repeatedly. This is now replaced by a cached scratch allocator (`crates/metallic/src/metals/sdpa/scratch.rs`).

## 3) FIXED: KV cache “repeat” behavior mismatch vs `context`

`context` writes KV caches already expanded to `n_heads` (repeat-at-write). Foundry previously did repeat-at-read (`RepeatKvHeads`), adding extra dispatches and bandwidth.

Foundry now uses `KvCacheWriteRepeatKvHeads` and stores KV cache tensors in `[n_heads, max_seq_len, head_dim]` so SDPA reads directly from cache without a repeat step.

## 4) FIXED: Missing FFN macro-fusion

Foundry now has a fused decode-path FFN op that mirrors the `context` macro-fusion shape:

- `FusedFfnSwiGluRmsNorm` (RMSNorm + gate/up projections + SwiGLU)
- Parity test: `crates/metallic/src/tests/fused_ffn_swiglu_rmsnorm_parity.rs`

## Remaining TODOs

1) **SDPA tile config selection**: `crates/metallic/src/metals/sdpa/step.rs` still hardcodes `TileConfig::default()` for the batched GEMM path. Using `TileConfig::auto_select(m, kv_seq_len)` is a likely next knob to tune.
2) **SDPA workspace ergonomics in Foundry**: `context` has a clean “cached workspace” abstraction for SDPA. Foundry now has scratch caching, but we should formalize it as a reusable Foundry workspace primitive (for SDPA and future fused ops).
