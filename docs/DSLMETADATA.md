# DSL + Metadata (Foundry) — Sprint Notes

This document captures the changes made in the “DSL/Metadata” sprint to remove Qwen2.5-specific hardcoding from the Foundry executor and move model preparation details into the **DSL** (model JSON) and **GGUF metadata**.

The top-level goal is:

- Make the executor prepare **exactly what the loaded model requires**, driven by DSL + metadata.
- Make missing requirements **fail fast and loudly** (panic) instead of silently producing wrong shapes/layouts.
- Establish precedence: **GGUF baseline < DSL overrides < runtime overrides**.

---

## Precedence model

For architecture numbers and runtime variables:

1. **GGUF metadata baseline** is inferred at load time.
2. **DSL overrides** (values explicitly present in the model JSON) win over baseline.
3. **Runtime overrides** (values set in `TensorBindings` / config) win over both.

---

## What moved into the DSL

### 1) `architecture.prepare`: executor preparation plan

The executor no longer hardcodes intermediate buffers and KV cache buffers for a specific architecture. Instead the DSL declares:

- `prepare.globals`: one-time integer expressions evaluated at session initialization.
- `prepare.derived_globals`: integer expressions evaluated at runtime (prefill chunk / decode step).
- `prepare.tensors`: tensors to allocate + bind before inference (intermediates, KV caches, rope caches).
- `prepare.rope`: optional names for RoPE caches (`cos` / `sin`).

#### Expressions (`IntExpr`)

Tensor shapes and derived globals use integer expressions (`IntExpr`):

- JSON number: `128`
- JSON string: `"d_model / n_heads"` or `"{m} * {d_model}"`

Expressions support:

- `{var}` sugar (equivalent to `var`)
- `+ - * /` and parentheses
- checked arithmetic (overflow/underflow panics)

Missing variables in expressions **panic** with guidance to add them to:

- `architecture.prepare.globals`
- `architecture.prepare.derived_globals`
- or pass a runtime override

### 2) `architecture.weight_bindings`: weight binding + layout is DSL-owned

Weight binding is now driven by the DSL instead of executor-coded “families”:

- Each entry binds a GGUF tensor resolved via `architecture.tensor_names`.
- Supports per-layer binding via `repeat` (`count: "n_layers", var: "i"`) and `{i}` interpolation.
- Optional biases can be declared with `fallback_zero_len` to bind a zero vector if missing.

#### Layout safety (canonical vs row-major)

To avoid NK/KN/canonical regressions, the DSL can declare layout explicitly:

- `row_major` (default)
- `canonical` with **explicit** `expected_k` and `expected_n` expressions (fail-fast validation)

---

## What moved into metadata (and how it’s configured)

### 1) GGUF-derived baseline architecture defaults

Model specs can omit architecture numerics and rely on GGUF metadata inference for:

- `d_model`, `n_heads`, `n_kv_heads`, `n_layers`, `ff_dim`, `vocab_size`, `max_seq_len`
- `rope_base`, `rms_eps` (optional; have defaults if missing)

### 2) `architecture.metadata_keys`: DSL-configurable metadata key mapping

GGUF metadata key names vary across model families. The DSL can now provide:

```json
{
  "architecture": {
    "metadata_keys": {
      "keys": {
        "d_model": ["qwen2.embedding_length", "model.d_model"],
        "n_heads": ["qwen2.attention.head_count"],
        "...": ["..."]
      }
    }
  }
}
```

This removes hardcoded “Qwen vs Llama” assumptions from the loader for models that supply `metadata_keys`.

---

## Runtime variable handling (current state)

### Dynamic values (`DynamicValue<T>`)

Kernels and steps can accept runtime-variable references such as `"{seq_len}"`.

- Resolution pulls from `TensorBindings` at runtime.
- Missing/invalid variables **panic** and point the author to DSL/runtime overrides.
- Integer types (`u32`, `usize`) check `int_globals` first to avoid string parsing.

### Baseline globals seeding (now usage-driven)

The executor seeds baseline globals (from architecture + derived expressions), but only for variables that the DSL actually references (via expression-variable scanning). This avoids always pre-populating the full text-generation key set.

---

## RoPE handling (current state)

- `prepare.rope` is now **optional**.
- If present:
  - The executor computes/uploads RoPE cache values.
  - RoPE cache buffers can be pre-allocated by `prepare.tensors` (the executor will reuse them if compatible).
- If absent:
  - The executor does not allocate/compute RoPE caches.

---

## What is still not fully DSL/metadata-driven (known gaps)

These remain intentionally for now and are expected to be addressed by the upcoming “workflow-based” system:

1. The `generate()` loop is still a **text-generation workflow** baked into the executor.
   - It manages runtime keys like `m`, `seq_len`, `position_offset`, `max_seq_len`, `max_prefill_chunk`.
2. Some “common runtime keys” are still interned for fast `int_globals` updates.
   - This is a performance optimization but will likely be reshaped once workflows own runtime state.

---

## Validation performed during this sprint

We used “change → immediate inference test” to catch layout regressions:

- `cargo build`
- `cargo run --release --package metallic_cli -- models/qwen2.5-coder-0.5b-instruct-fp16.gguf "create very short rust helloworld function" --engine foundry --max-tokens 100 --output-format text`

We also used targeted unit/integration tests for DSL+metadata components (not the full suite due to cost).

