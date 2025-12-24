# Qwen2.5 Migration Checklist

This document tracks the migration of kernels required for the Qwen2.5 inference pipeline to the new Foundry system.

FOLLOW NEW-KERNELS.md and MACROS.md for migration/implementation details.

## Core Model Kernels

- [x] **EmbeddingLookup** (`EmbeddingLookupOp`)
    - Used in: `qwen25.embed`, `embed_single_token_cached`
    - Logic: Simple gather index lookup.
    - Status: **Ported** - 7 tests passing
- [x] **RMSNorm** (`RMSNormOp`)
    - Used in: `transformer_block`, `final_norm`
    - Logic: Root Mean Square Normalization.
    - Status: **Ported** - 29 tests passing (F16 + Q8)
- [x] **RoPE** (`RoPEOp`)
    - Used in: `transformer_block` (Attention)
    - Logic: Rotary Positional Embeddings with precomputed cos/sin cache.
    - Status: **Ported** - 6 tests passing
- [x] **KvRearrange** (`KvRearrangeOp`)
    - Used in: `transformer_block` (Attention)
    - Logic: Rearranges QKV heads for attention.
    - Status: **Ported** - 6 tests passing
- [x] **RepeatKvHeads** (`RepeatKvHeadsOp`)
    - Used in: `transformer_block` (Attention) for GQA
    - Logic: Repeats K/V from n_kv_heads → n_heads.
    - Status: **Ported** - 5 tests passing
- [x] **SwiGLU** (`SwiGLUFusedActivationOp`)
    - Used in: `transformer_block` (MLP)
    - Logic: SiLU(x) * y fused with bias add.
    - Status: **Ported** - 6 tests passing (already vectorized)
- [x] **ElemwiseAdd** (`BroadcastElemwiseAddInplaceOp`)
    - Used in: Residual connections (`resid_attn`, `resid_mlp`), bias add
    - Logic: Broadcast add: out[i] = a[i] + b[i % b_len]
    - Status: **Ported** - 1 test passing
- [ ] **Matmul** (Foundry Native)
    - Used in: All projections.
    - Status: **Already Ported** (CompoundGemv, but we need to eventually bring GEMM over).

## Attention Kernels

- [x] **SoftmaxVec** (`SoftmaxVecOp`)
    - Used in: SDPA, general softmax (medium sequences 256-2047)
    - Logic: Simdgroup reductions with causal masking
    - Status: **Ported** (Foundry Native) - 8 tests passing (F16/Q8/Edge)
- [x] **SoftmaxBlock** (`SoftmaxBlockOp`)
    - Used in: SDPA with very long sequences (>4096)
    - Logic: Segmented reductions with cross-segment normalization
    - Status: **Ported** (Foundry Native) - 8 tests passing (F16/Q8/Edge)
- [x] **ScaledDotProductAttention** (`scaled_dot_product_attention`)
    - Used in: `transformer_block` (core attention)
    - Logic: softmax(QK^T / sqrt(d)) * V with optional causal mask
    - Status: **Ported** (Foundry Native) - Chained Gemv+Softmax+Gemv dispatch
    - Features: Incremental decode optimization (matches Legacy `seq_len_delta`), `METALLIC_SDPA_BACKEND` override
    - Verified: CPU parity, Legacy parity, Causal masking

## Generation & System Kernels

- [x] **SampleTopKTopP** (`SampleTopKTopPOp`)
    - Used in: `generation` loop.
    - Logic: Fused TopK + TopP + Sampling.
    - Status: **Ported** (Foundry Native via `SampleTopK`) - Verified Argmax & Determinism
- [x] **Arange** (`ArangeOp`)
    - Used in: `Tensor::arange` (creation).
    - Status: **Ported** - 1 test passing
- [x] **Ones** (`OnesOp`)
    - Used in: `Tensor::ones` (creation).
    - Status: **Ported** - 1 test passing (already vectorized)
- [x] **RandomUniform** (`RandomUniformOp`)
    - Used in: `Tensor::random_uniform` (creation).
    - Status: **Ported** - 1 test passing

## Custom Fusion Kernels (Optimization)

> [!NOTE]
> These are high-performance fused kernels that combine Matmul with RMSNorm or other ops.

- [ ] **MatmulGemvQkvFusedRmsnormOp**
    - Used in: `transformer_block` (Q8 quantization path).
    - Logic: Fuses QKV projection with pre-normalization.
- [ ] **MatmulF16CanonicalQkvFusedRmsnormOp**
    - Used in: `transformer_block` (F16 path).
    - Logic: Fuses QKV projection with pre-normalization for F16.

---

## Optimization Backlog

> [!TIP]
> These are optimizations identified during porting. All kernels currently match legacy parity.

### Vectorization & Memory Bandwidth
- [ ] **RMSNorm**: Vectorized `half4` loads in `rmsnorm_compute_inv_rms` (match fused GEMV+RMSNorm perf)
- [ ] **Embedding**: Vectorized `half4`/`half8` loads for better memory bandwidth
- [ ] **RoPE**: Pair processing (1 thread handles both i and j) → 2x fewer threads, no redundant reads
- [ ] **RepeatKvHeads**: Vectorized `half4`/`half8` loads - currently scalar copy
- [ ] **ElemwiseAdd**: Vectorized `half4` loads/stores (currently scalar)
- [ ] **Arange**: Vectorized generation (4/8 indices per thread)
- [ ] **RandomUniform**: Vectorized RNG (generate 4-8 random numbers per thread)
- [ ] **SampleTopK**: Wide vector loads (Half8/Float8) for memory bandwidth
- [ ] **SampleTopK**: Use `simd_sum`/`simd_max` intrinsics instead of manual `quadlane_reduce`

### Threadgroup Optimizations (for Compound Kernel Fusion)
- [ ] **KvRearrange**: Add threadgroup coordination for potential fusion with QKV projection
- [ ] **RoPE**: Threadgroup cos/sin sharing → reduce memory traffic
- [ ] **RepeatKvHeads**: Threadgroup-based coalesced copy for larger batches
- [ ] **SampleTopK**: Multi-threadgroup support (Partials + Merge) for very large vocabularies (>128k)

### SwiGLU Specific
- [ ] **SwiGLU composite**: Full `SwiGLUOp` (matmuls + activation) still in legacy - needs compound kernel migration
- [ ] **SwiGLU fusion**: Consider fusing down_proj matmul with activation output

- [ ] **SoftmaxVec**: 
    - Implement `simd_max` and `simd_sum` intrinsics for cleaner reduction code
    - Use vectorized loads (float4/half4) for higher memory throughput on larger widths
- [ ] **SoftmaxBlock**: 
    - Dynamic segment sizing (tuning SEGMENT_SIZE based on device/sequence length)
    - Optimization of cross-segment reduction phase (currently serial on thread 0)
- [ ] **Softmax**: Use `PolicyQ8` correctly with real packed scale loading if input quantization becomes a requirement (currently mocked/compatible)

### SDPA Specific
- [ ] **Stateful `seq_len_delta` Tracking**: Port full tracking from Legacy (`ctx.sdpa_seq_delta`, `sdpa_workspace_key_for`) for exact parity in edge cases. Currently using `query_offset > 0` as proxy.
- [ ] **MLX vs Incremental Decode Benchmarking**: Once MLX Matmul is ported, benchmark `METALLIC_SDPA_BACKEND=mlx` vs `auto` to determine optimal strategy.
- [ ] **Native Matmul Kernel**: Port MLX Matmul for Prefill (M>1) performance parity with Legacy.

### General
- [ ] Unrolling where possible to hide latency (like legacy kernels)
- [ ] Review all simple kernels for potential fusion opportunities in compound kernel system

---

## Migration Strategy

For each kernel:
1.  **Define Struct**: Create `#[derive(KernelArgs)]` struct in `src/metals/...`.
2.  **Define Logic**: Create `Kernel` trait implementation.
3.  **Port Metal Code**: Ensure `.metal` source is compatible.
4.  **Register**: Add to `src/metals/mod.rs`.
5.  **Verify**: Run parity test against legacy `Context::call`.
6.  **Backlog**: Note any obvious optimization opportunities.
