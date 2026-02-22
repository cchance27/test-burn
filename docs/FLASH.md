# FlashAttention (Foundry / Metal)

This doc tracks the current state of the **decode (M=1)** and **prefill (M>1)** FlashAttention paths in Metallic Foundry (Metal backend).

## Current Status (2026-02-22)

- **Decode (M=1):** Fused **RoPE → FlashAttention decode** with streaming softmax (online max/logsumexp) and fused `P·V` accumulation.
- **Prefill (M>1):** Tiled **FA1-style online softmax prefill** for `head_dim ∈ {64, 128}` with optional **Split‑K** for large KV.
- **Default Behavior:** Enabled by default for supported configurations. Unsupported shapes/configurations fail fast with explicit errors.
- **Tuning knobs added:** Prefill `WARPS` selector and Split‑K selector (env overrides) for benchmarking without code changes.
- **No implicit fallback:** `FlashAttentionStep` does not silently route to materialized SDPA. Use `SdpaReferenceStep` explicitly when needed.

## Composition (How it’s built)

The fused decode path is implemented as a **compound kernel**:

- `HeadLayoutStage`:
  - Computes `q_ptr/k_ptr/v_ptr/output_ptr` for `(head_idx, batch_idx)`.
  - Assumes KV is head-major: `[n_heads, capacity, head_dim]`.
- Optional fused RoPE stage:
  - Uses the standard `metals::rope::stage::RopeStage` (no RoPE duplication inside attention).
- `FlashDecodeStage` (Standalone) or `FlashDecodeFusedStage` (Fused):
  - Calls `flash_decode_warp_tiled_m1<WARPS>()` (Metal).
  - `FlashDecodeStage` loads Q into threadgroup once.
  - `FlashDecodeFusedStage` consumes Q from RoPE’s shared buffer.

Key sources:
- Kernels:
  - `crates/metallic-foundry/src/metals/flashattention/decode_common.metal`
  - `crates/metallic-foundry/src/metals/flashattention/decode_layout.metal`
  - `crates/metallic-foundry/src/metals/flashattention/decode_kernels.metal`
- Stage wiring: `crates/metallic-foundry/src/metals/flashattention/stages.rs`
- Dispatch/args: `crates/metallic-foundry/src/metals/flashattention/step.rs`
- DSL dispatch: `crates/metallic-foundry/src/metals/sdpa/step.rs`
- Streaming softmax primitive: `crates/metallic-foundry/src/metals/softmax/streaming.metal`

Prefill is implemented as standalone compound kernels (no RoPE):

- **Non‑Split‑K prefill:** `SdpaPrefillStage` → `flash_prefill_tiled_d{64|128}<WARPS>()`
- **Split‑K prefill (2‑phase):**
  - Part: `SdpaPrefillSplitKPartStage` → `flash_prefill_splitk_part_d{64|128}<WARPS>()`
  - Reduce: `SdpaPrefillSplitKReduceStage` → `flash_prefill_splitk_reduce_d{64|128}<WARPS>()`

Prefill sources:
- Kernels:
  - `crates/metallic-foundry/src/metals/flashattention/prefill_loaders.metal`
  - `crates/metallic-foundry/src/metals/flashattention/prefill_online.metal`
  - `crates/metallic-foundry/src/metals/flashattention/prefill_splitk.metal`
- Stage wiring: `crates/metallic-foundry/src/metals/flashattention/stages.rs`
- Dispatch/selector: `crates/metallic-foundry/src/metals/flashattention/step.rs`

## What this kernel is (and is not)

- **It is:** “FA1‑style” for inference (causal, FP16, fixed head dims)
  - **Decode (M=1):** streaming softmax + fused `P·V`, processing KV in blocks and selecting a decode variant (`WARPS`, `KEYS_PER_WARP`, scalar width).
  - **Prefill (M>1):** tiled online softmax over KV tiles (32 keys/tile) and fused accumulation of `P·V`.
  - **Split‑K prefill:** optional 2‑phase Split‑K to parallelize KV for large contexts, then reduce to final output.
- **It is not (yet):** “FA2”
  - No pipelined MMA (`simdgroup_matrix`) / double‑buffered K/V staging / deep software pipelining.
  - Not fully generic across arbitrary head dims, batch sizes, masks, dropout, bf16, etc.

## Feature matrix (Foundry vs FA1/FA2)

Legend:
- ✅ supported
- ⚠️ supported with limitations
- ❌ not supported

### Decode (M=1)

| Feature | Foundry | FA1 | FA2 |
|---|---:|---:|---:|
| Causal attention | ✅ | ✅ | ✅ |
| Streaming softmax (online max/logsumexp) | ✅ | ✅ | ✅ |
| Fused `P·V` accumulation | ✅ | ✅ | ✅ |
| Variant tuning (warps/keys-per-warp) | ✅ | ⚠️ (impl-dependent) | ⚠️ (impl-dependent) |
| Split‑K | ❌ | ❌ (not typical for decode) | ❌/⚠️ (rare; impl-dependent) |
| Head dims | ⚠️ (`64/128`) | ⚠️ (impl-dependent) | ⚠️ (impl-dependent) |
| Batch>1 | ⚠️ (implementation-dependent) | ⚠️ | ⚠️ |
| Arbitrary masks / padding | ❌ | ⚠️ | ⚠️ |
| bf16 | ❌ | ⚠️ | ⚠️ |

### Prefill (M>1)

| Feature | Foundry | FA1 | FA2 |
|---|---:|---:|---:|
| Causal attention | ✅ | ✅ | ✅ |
| Tiled KV processing | ✅ | ✅ | ✅ |
| Online softmax across KV tiles | ✅ | ✅ | ✅ |
| Split‑K | ✅ (2‑phase) | ✅ | ✅ |
| Double‑buffered K/V staging | ❌ | ❌/⚠️ | ✅ |
| MMA / pipelined compute | ❌ | ❌/⚠️ | ✅ |
| Head dims | ⚠️ (`64/128`) | ⚠️ (impl-dependent) | ⚠️ (impl-dependent) |
| Batch>1 | ❌ (current dispatch is `batch=1`) | ⚠️ | ⚠️ |
| Arbitrary masks / padding | ❌ | ⚠️ | ⚠️ |
| bf16 | ❌ | ⚠️ | ⚠️ |

## Known Limitations / Risks

- **Numerical differences:** output differs slightly from the materialized path due to different accumulation order (streaming softmax). Tests use a loose tolerance.
- **Head dim:** currently supports `head_dim==64` and `head_dim==128` for both decode and prefill. Other dims fail fast.
- **Batch size:** prefill dispatch is currently `batch=1` (grid.z = 1). Decode path may support batch in the fused stack, but Foundry inference is typically batch=1.
- **Split‑K scratch:** Split‑K prefill uses FP32 scratch buffers for partials (performance/memory tradeoff). This is expected for correctness; later optimizations can shrink or pack scratch.
- **Fixed-capacity intermediates:** Foundry commonly uses fixed buffers (e.g. Q: `[1, 32, d_model]`, Out: `[32, d_model]`). The online path assumes decode uses the **first row** (m=1) and requires the last dimension to be contiguous.
- **Quantization boundary (intentional SRP):** FlashAttention/SDPA kernels stay quant-policy-agnostic and consume typed tensor/layout contracts. Quantization policy logic stays in loader/policy components; quantized KV attention should pass a format descriptor + buffers rather than import policy types into FA stages.

## Prefill (M>1) gotchas (important)

These are easy footguns that can produce “prompt ignored” / nonsense generations even when decode (M=1) looks fine.

### 1) Token-major metadata vs head-major contents (Q)

Foundry often stores Q intermediates with **token-major dims/strides metadata** (fixed-capacity), while the **underlying buffer contents** were produced by fused KV-prep as a tightly-packed **head-major** layout over the *true* `m`.

Example observed in the wild:
- `q_dims=[1, 32, d_model]` / `q_strides=[32*d_model, d_model, 1]`
- but `m=26`, and the fused writer packed Q as `[n_heads, m, head_dim]` with `q_stride_h = m*head_dim` (not `32*head_dim`).

If the prefill SDPA kernel derives Q strides from `M_cap` (32) instead of the actual `m`, it reads misaligned Q for most heads/rows and prefill diverges hard (greedy tokens change immediately).

Fix: in the online prefill path, treat `[1, M_cap, d_model]` metadata as “head-major packed over `q_len`” and set `q_stride_h = q_len*head_dim`. See `crates/metallic-foundry/src/metals/flashattention/step.rs`.

### 2) `gid` semantics in compound kernels

In Foundry compound kernels, `gid` is `[[threadgroup_position_in_grid]]` (not `[[thread_position_in_grid]]`). Prefill kernels should treat `gid.x` as the **tile index** (threadgroup index), not as a global thread id. See `crates/metallic-foundry/src/fusion.rs` (`THREAD_ARGS`).

## KV-cache quantization proposal (Policy-driven, future)

This is a proposal for adding a **quantized KV-cache** format that can be consumed by FlashAttention kernels (and potentially other attention backends) without entangling quantization logic into the SDPA dispatcher.

Goal:
- Reduce **bandwidth** and **memory footprint** for long-context attention by storing K/V in a smaller representation.
- Keep quantization logic **policy-driven** (SLP) and loaded externally; attention kernels only need to *consume* a declared format.

### Why this is likely a win on Metal

- Long-context attention is often **bandwidth-bound**: KV cache reads dominate. Smaller KV reduces bytes moved per token.
- Smaller KV improves cache residency and reduces memory pressure, enabling longer contexts and potentially improving tail latency.

### What it would look like (high-level)

1) **External policy loader owns packing + metadata**
- A KV-cache quant policy defines:
  - `format_id` (versioned), `group_size`, packing layout, alignment requirements
  - scale layout (`per-block` / `per-channel-chunk`) and dtype
  - optional outlier side tables (future)
- It produces KV buffers in a stable representation:
  - `kq_values`, `kq_scales` (+ optional `kq_zeros`)
  - `vq_values`, `vq_scales` (+ optional `vq_zeros`)

2) **FlashAttention “tile loader” consumes the format**
- Replace the FP16 K/V tile loads with:
  - load packed bytes + scales
  - dequant into registers (or threadgroup if required)
  - run the same streaming softmax + `P·V` logic

This is still “FlashAttention” conceptually; only the K/V representation changes.

### Suggested initial scope (lowest-risk)

- **Start with int8 (K and/or V)** with per-block scales (block = `32 keys × Dchunk`).
- Keep Q in FP16, accumulate in FP32.
- Focus first on **decode (M=1)** (highest chance of bandwidth win) before quant prefill.

### Proposed memory/layout contracts (sketch)

Constraints to keep fast paths fast:
- KV remains **head-major** at the logical level: `[n_kv_heads, capacity, head_dim]`.
- Quant buffers are packed such that each `(head, key-tile, d-chunk)` is contiguous for coalesced reads.

One reasonable baseline:
- Values: `int8` packed as `[head][capacity][head_dim]` (byte-per-element).
- Scales: `f16` or `f32` per `(head, capacity_block, d_chunk)` where:
  - `capacity_block = 32` (matches our KV tile)
  - `d_chunk` is e.g. 32 (matches `half2`/`half4` lane mapping), tunable by policy.

### Routing / selector requirements

This must be opt-in by capability and auto-selected:
- Enable only when:
  - policy loader provides quant KV for the active model/session
  - device capability/heuristics predict a win (kv_len, head_dim, etc.)
- Always provide fail-fast behavior for unsupported formats.

### Testing requirements

Add parity coverage analogous to current prefill parity tests:
- Compare quant-KV attention outputs vs materialized SDPA (tolerances will be looser than FP16).
- Stress tests:
  - very long KV
  - extreme logits (softmax stability)
  - outlier-heavy distributions (if/when outlier path exists)

## FlashAttention Roadmap (Next Steps)

### Phase 1: Performance Tuning
- **Tune Split‑K selector**: add a sweep for `(kv_len, m)` and pick good defaults per device family.
- **Reduce Split‑K overhead**: minimize scratch bandwidth (packing, fewer passes, better cache locality) and reduce kernel launch overhead where possible.
- **Parameter tuning**: benchmark and tune prefill `WARPS` and decode variants across Apple Silicon generations (M1–M4).
 - **Quantized KV-cache (proposal)**: evaluate an int8 KV-cache policy and add a gated decode fast path if it’s a clear win.

### Phase 2: Generic Prefill
- **Batch>1 support**: add true batch dimension support for prefill dispatch and pointer math.
- **Generic masking**: add padding masks / arbitrary masks (beyond causal) where needed.

### Phase 3: Advanced Features
- **FA2-style pipelining**: double‑buffered KV loading and pipelined MMA for prefill.
- **More head dims**: expand beyond 64/128 via codegen + templated Metal.

## Debugging / Safety Knobs

- **`METALLIC_DISABLE_FA=1`**: `FlashAttentionStep` fails fast (no implicit route to materialized SDPA).
- `METALLIC_SDPA_FORCE_MATERIALIZED=1`: `FlashAttentionStep` fails fast (materialized path must be selected explicitly via `SdpaReferenceStep`).
- `METALLIC_DEBUG_SDPA=1` (+ `METALLIC_DEBUG_SDPA_VERBOSE=1`): Logs SDPA dispatch decisions, shapes, and active path.
- `METALLIC_SDPA_DEBUG_ONLINE_COMPARE=1`: Runs online+materialized once and prints max-abs-diff diagnostics (slow; debug-only).
- `METALLIC_SDPA_DEBUG_ONLINE_COMPARE_PREFILL=1`: currently logs a one-shot prefill compare marker (bounded compare wiring is pending).

### Prefill tuning knobs (M>1)

- `METALLIC_FA_PREFILL_WARPS`: `4` or `8` (controls threadgroup size and TileM = `WARPS*4`).
- `METALLIC_FA_PREFILL_SPLIT_K`: force Split‑K factor `N` (`N<=1` disables). Used for benchmarking and parity testing.
- `METALLIC_DISABLE_FA_PREFILL_SPLITK=1`: disables Split‑K prefill (forces `split_k=1`).

### Decode tuning knobs

These are **decode (M=1) only** and exist for performance tuning. Invalid values will fail-fast.

- `METALLIC_FA_DECODE_WARPS`: override warps (simdgroups) per threadgroup.
- `METALLIC_FA_DECODE_KEYS_PER_WARP`: override keys processed per warp per iteration.
- `METALLIC_FA_DECODE_SCALAR`: override Q/K/V vectorization (`half2` or `half4`).
- `METALLIC_FA_DECODE_TG_OUT`: override threadgroup partial storage (`float`/`half`). `half` can win for very long KV on some devices but may slightly increase numerical error.

## Tests / Bench Sweeps

- Prefill parity: `crates/metallic-foundry/tests/flashattention_prefill_parity.rs`
- Prefill warps sweep (ignored): `crates/metallic-foundry/tests/flashattention_prefill_variant_sweep.rs`
- Decode sweeps (ignored):
  - `crates/metallic-foundry/tests/flashattention_decode_variant_sweep.rs`
  - `crates/metallic-foundry/tests/flashattention_rope_decode_variant_sweep.rs`
