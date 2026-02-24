# SageAttention-style Quantized Attention (Foundry / Metal)

This doc sketches what a **SageAttention-like** attention backend could look like in Metallic Foundry (Metal), how it differs from our current **FlashAttention (FP16)** path, and a phased plan for implementing it **after FA1/FA2 are complete**.

The guiding principles here match the repo’s architecture goals:
- **Performance/latency first**, memory second.
- Keep quantization logic **componentized** and driven by an external **policy loader** (SLP), while allowing attention kernels to efficiently *consume* policy-defined formats.

## What “SageAttention-like” means (in practice)

SageAttention-style approaches typically aim to speed up attention by:
- Storing and/or computing attention with **low-bit representations** (often K/V quantized; sometimes also QKᵀ / PV math).
- Using **scale/outlier handling** (and sometimes smoothing/calibration) to preserve output quality.

In contrast, our current Foundry attention is:
- **FP16 kernels** with streaming softmax and fused accumulation (FA1-style), plus Split‑K for prefill.
- Quantization today is treated as a separate concern (policies), not baked into SDPA.

## Where the win likely comes from on Metal

On Apple GPUs, the most reliable upside is often:
- **Bandwidth and cache**: smaller KV-cache → fewer bytes moved per token → better latency for long contexts.
- **Memory footprint**: KV-cache size drops substantially → enables longer contexts / reduces memory pressure.

The less-certain upside:
- **Raw compute**: “quantized dot” only wins if Metal + the target Apple GPU provide efficient low-bit dot-product primitives for the chosen format. Otherwise, dequant overhead can eat the gains.

## Architectural shape in Foundry (aligned with SLP/policies)

To keep responsibilities clean:

### Policy loader (external, centralized; SLP)
Owns **format + packing** and produces KV-cache artifacts with stable metadata:
- Quant format: e.g. int8 per-block, int4 packed, group size, signedness.
- Scale / zero-point layout and alignment guarantees.
- Optional outlier metadata (indices/values) and smoothing parameters.
- Versioning (format IDs) and device capability gating.

### Attention kernels (Foundry SDPA backend)
Own **consumption** of the policy-defined formats:
- Efficient tile loads of packed K/V.
- On-the-fly dequant (or mixed dequant + outlier path).
- Streaming softmax and fused `P·V` accumulation.

This does mean the kernel must understand the format’s layout, but the *choice of format* remains policy-driven.

## What it would look like (components)

Think of this as a parallel SDPA backend with two paths:

### 1) Quant Decode (M=1)
Goal: reduce per-token KV bandwidth.

Likely structure:
- `KvQuantTileLoadStage`: loads packed K/V + scales (and outliers) into registers / threadgroup.
- `SdpaQuantDecodeStage`:
  - computes Q·K (dequant K on the fly)
  - streaming softmax
  - accumulates P·V (dequant V on the fly)

Key design decisions:
- Dequant in **registers** whenever possible; avoid threadgroup barriers.
- Keep accumulation in **FP32** for stability.
- Choose formats that minimize unpack cost (e.g. int8 is much easier than int4).

### 2) Quant Prefill (M>1)
Goal: reduce KV bandwidth and/or enable higher parallelism for large KV.

Likely structure:
- Non-Split-K: tiled `M` × tiled `K/V` consumption with online softmax.
- Split‑K: (same as FA1 completion conceptually) per-split partials + reduce.

Quant prefill is harder because:
- You’re dequantizing large volumes while also doing more math per call.
- The reduction path must remain numerically stable and efficient.

## Selector / routing

The SDPA dispatcher would select between:
- `FlashAttention` (FP16, FA1/FA2)
- `QuantAttention` (Sage-like)
- materialized SDPA via explicit `SdpaReferenceStep` routing (no implicit fallback)

Selection inputs:
- Policy availability (quant KV present? format supported?).
- Device capabilities (supported Metal features / performance heuristics).
- Shape thresholds (kv_len, head_dim, m, etc.).
- User override knobs (debug/perf experiments).

## Quality + testing expectations

Quant-attention must be validated separately from FP16 FlashAttention:
- **Parity tests** vs materialized SDPA with appropriate tolerances.
- Stress tests for:
  - extreme KV lengths
  - extreme logits (softmax stability)
  - outlier-heavy distributions
- “Golden prompts” for end-to-end regression checks where feasible.

## Phased plan (lowest risk first)

### Phase 0 — Policy-first groundwork
- Require quant attention to be driven by an **external policy loader**:
  - KV-cache packing + metadata is produced outside SDPA.
  - SDPA receives opaque “format descriptor” + buffers.
- Add format IDs + capability checks (fail-fast on unsupported formats).

### Phase 1 — KV-cache compression without changing attention math
Goal: get bandwidth wins with minimal algorithm risk.
- Quantize **V only** (or V + K storage) but dequant to FP16/FP32 for math.
- Keep Q in FP16.
- Measure:
  - long-context decode latency
  - memory footprint improvements
  - dequant overhead

### Phase 2 — Quant K/V consumption inside attention kernels
- Quantize **K** for QKᵀ and **V** for PV, dequant on the fly.
- Maintain FP32 accumulation and streaming softmax stability.
- Add a robust selector (only enable where it wins).

### Phase 3 — Sage-like enhancements (only if needed)
- Outlier-aware paths:
  - store sparse outlier values separately
  - mix quant + FP16/FP32 for outliers
- Smoothing/calibration knobs in policy loader.

### Phase 4 — More aggressive formats (optional)
- Explore int4 / fp4-ish packing only if Metal + target devices show a real compute or bandwidth advantage after accounting for unpack costs.

## Practical risks / unknowns (Metal-specific)
- Whether low-bit dot-product is a true win depends heavily on:
  - Apple GPU instruction support and compiler behavior
  - memory coalescing for packed formats
  - unpack/dequant cost vs bandwidth savings
- “Sage-like” quality methods (outliers/smoothing) are non-trivial and may need substantial tuning per model family.

## Success criteria (what “worth it” means)
- **Decode:** measurable latency/tok/s wins at large KV (where bandwidth dominates), without quality regressions.
- **Prefill:** improved prefill latency for large KV and medium/large M, or enable longer contexts by reducing KV memory.
- Clear selector thresholds so the quant path is only used where it wins.
