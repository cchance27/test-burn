# Foundry TODOs / Known Gaps

## Context length is currently capped at 2048

**Status:** Known limitation (intentional cap to keep memory bounded and perf stable).

Foundry currently uses a **runtime max sequence length cap of 2048** for KV-related buffers and the `max_seq_len` global.

### Why this exists

Foundry’s current “model-load session” approach materializes and caches several buffers whose sizes scale linearly with
`max_seq_len` (most importantly the **per-layer KV caches**). If we were to respect large spec values (e.g. 32k) by
allocating those buffers eagerly at model load, memory usage would balloon (often into multi-GB) and would be unstable
across different models/devices.

To prevent that, Foundry historically allocated KV-related storage using a hard cap (2048). After the session refactor,
we also had to ensure the **`max_seq_len` global matches the actual allocated shapes** (otherwise indexing/strides can be
wrong and performance can “flap” wildly).

### Where the cap is enforced

- `crates/metallic-foundry/src/model/executor.rs`: `CompiledModel::RUNTIME_MAX_SEQ_LEN_CAP` and
  `CompiledModel::runtime_max_seq_len(...)`.

### What this impacts

- Any prompt + decode that requires KV capacity > 2048 will fail (or be clamped) depending on the call path.
- This is not “true model max context”; it’s a temporary Foundry execution/allocator limitation.

### TODO: implement a real large-context solution for Foundry

Goal: **Respect `arch.max_seq_len` without allocating worst-case memory at model load.**

Preferred direction:

1) **Per-session KV capacity**: allocate KV to `prompt_len + max_new_tokens (+ headroom)`, capped by `arch.max_seq_len`.
2) **Grow-on-demand KV**: support reallocating/growing KV (and any dependent buffers) if generation exceeds headroom.
3) **Avoid full-history expanded buffers**: eliminate or lazy-allocate `k_expanded/v_expanded`-style full-history
   intermediates when possible (derive views, compute per-step slices, or use kernels that read from canonical KV).
4) **Keep globals consistent**: `max_seq_len` used by kernels must always match the actual allocated stride/shape.

## How Context handles this today

`metallic-context` does **not** preallocate the worst-case max context at model load.

Instead, for each generation it computes a **KV capacity** based on the actual run:

- `kv_capacity = (prompt_len + cfg.max_tokens).min(model_seq_len)`

Then it:

- clears caches between generations,
- reserves the KV pool exactly for that capacity,
- allocates per-layer KV caches sized to `kv_capacity`.

This keeps memory bounded by the requested run length while still respecting the model’s max sequence length.

