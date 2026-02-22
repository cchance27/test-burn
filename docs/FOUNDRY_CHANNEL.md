# Foundry Stream Channels (Phase Plan)

This document describes the plan to add **generic stream channels** to Foundry so workflows can:

- stream outputs (tokens, bytes, tensors) without per-step CPU `wait_until_completed`
- support both GPU→CPU (D2H) and CPU→GPU (H2D) communication
- remain flexible for future **non‑LLM** workflows (encoders, DiT, diffusion)

Design priorities:

1. **Performance + latency first**: no hot-path allocations; avoid unnecessary syncs.
2. **Clean DX**: keep workflows compositional; avoid requiring workflow authors to hand-roll sync logic.
3. **Safety**: make “dangerous” semantics (KV overshoot, blocking reads) explicit and guarded.

## Status

- Phase 1 is implemented (channel substrate + kernels + tests).
- Phase 2 is partially implemented:
  - workflow ops `stream_init` + `stream_write_u32`
  - a sample workflow `text_generation_stream_u32.json`
  - a focused integration test validating channel emission matches `generated_tokens`
  - `while_batched.stream_channel` can emit tokens via `ChannelU32` instead of per-token scalar readback
  - `while_batched.stream_async_poll` provides safe overlap via *pipelined* command buffers (drain batch N while GPU runs batch N+1)

---

## Non-goals (initially)

- Full async runtime integration (`futures::Stream`) in core library (optional feature later).
- Multi-producer/multi-consumer lock-free correctness on GPU (v1 is single-writer per capture order).
- Streaming detokenized UTF-8 bytes from GPU (tokens first; bytes later).

---

## Core concept

A **channel** is a small shared-memory ring buffer + header used to transfer items between GPU and CPU.

### v1: `Channel<u32>` (tokens / ids)

Backed by two `TensorArg`s (both `StorageModeShared`):

- `header: u32[4]`:
  - `header[0] = write_idx` (monotonic count of items written)
  - `header[1] = read_idx` (monotonic count of items consumed; maintained by CPU for D2H v1)
  - `header[2] = capacity`
  - `header[3] = flags` (reserved)
- `data: u32[capacity]` ring buffer

Ops write using a tiny GPU kernel. CPU reads by observing `write_idx` and mapping ring positions.

### Why not “channels are tensors”?

`TensorArg` is our “any tensor” handle, but a channel has **dynamic validity** (moving cursor, wrap/drop).
So the workflow value should be a distinct variant (`Value::Channel`) to avoid accidental misuse.

---

## Workflow integration (high level)

Channels should be an **engine primitive** that ops can opt into; workflows should not need wrapper blocks to benefit.

Planned interface:

- `Value::Channel(ChannelHandle)` (typed wrapper around `TensorArg` storage)
- `WorkflowExecutionContext` provides access to per-run channels (`ctx.channels`)
- ops can:
  - emit items into a channel (GPU→CPU)
  - consume items from a channel (CPU→GPU; later)

---

## Phase 1 (this sprint): framework + parity tests

Goal: land the “channel substrate” with tests, without changing existing workflows or CLI behavior.

### 1) Types

- Add `Value::Channel(ChannelU32)` (v1 only).
- Add `ChannelU32Reader`:
  - `try_next() -> Option<u32>` (non-blocking)
  - `drain_into(&mut Vec<u32>)` (bulk read)
  - implements `Iterator<Item=u32>` where `next()` is non-blocking and returns `None` when empty.

### 2) Kernels (Foundry way)

Add Metal kernels under `crates/metallic-foundry/src/metals/channel/`:

- `channel_u32_init` (gid==0): initialize header (and optionally zero data)
- `channel_u32_push` (gid==0): write one value using `write_idx % capacity`

No custom Metal structs are authored directly in `.metal`; any structs must be injected via Rust macros.
No `#include` of project headers in `.metal`; use `Kernel::includes()` if needed.

### 3) Tests

Add targeted integration tests (no full suite):

- **Order/no wrap:** capacity large enough; push N values; read N; assert exact order.
- **Wrap/drop-oldest:** small capacity; push > capacity; reader drops overwritten values by clamping start
  to `max(read_idx, write_idx - capacity)`; assert the last `capacity` values are observed.

### 4) Guardrails

Phase 1 does not expose channels to workflows by default; no existing JSON changes required.

---

## Phase 2 (next sprint): first ops + async streaming path

Goal: use channels to stream token ids without per-token waits in throughput mode.

### 1) Ops

- `stream_init` (alloc/register channel) ✅
- `stream_write_u32` (writes a token id to the channel; accepts `u32` or `TensorArg(u32[1])`) ✅
- `stream_flush` (commit capture window; optional wait vs async) ⏳

### 2) Runner/TUI integration

- Throughput: commit large decode windows and overlap CPU work with GPU decode. ✅ (pipelined, batch-granularity)
- Interactive/TUI: small windows or hybrid (completion handler + chunked polling) to keep latency low. ⏳

Notes:

- With `stream_channel` enabled, `while_batched` sources token emission from the channel instead of `token_var`.
- With `stream_async_poll` enabled, overlap is achieved by pipelining command buffers, not by reading the ring
  buffer mid-command-buffer (which would require stronger device→CPU ordering guarantees than Metal provides today).

Debug helpers:

- `METALLIC_DEBUG_WORKFLOW_OPS=1`: logs workflow op execution and `while_batched` stream config.
- `METALLIC_DEBUG_STREAM_POLL=1`: logs one summary line per pipelined drain (tokens drained + inflight count).

### 3) EOS + KV safety

Maintain the existing guardrails:

- multi-turn workflows must not use “overshoot” batching unless explicitly opting in.
- streaming windows must not leak tokens that were computed past EOS into the visible transcript.

Current limitation (important):

- `while_batched.stream_async_poll` is only enabled when EOS stopping is disabled (`METALLIC_IGNORE_EOS_STOP=1`).
  If EOS stopping is enabled, Foundry disables async polling for that run (warns) and falls back to synchronous
  draining. This mode is intended for throughput/single-turn. Multi-turn EOS-safe pipelining needs explicit
  windowing logic.
- For async polling, channel capacity must be at least `2 * batch_size` so draining can lag by one batch without
  overwriting unread tokens.

---

## Stretch goals (future)

- H2D channels (CPU sampling / external control)
- channels for bytes/structured payloads (detokenized bytes, latents)
- optional `futures_core::Stream` impl behind a feature flag (no core deps). Note: stable `std` does not
  currently expose a `Stream` trait; if we want “async iteration” ergonomics we should either:
  - provide our own small `PollNext` trait (no deps), or
  - gate a `futures_core::Stream` adapter behind a Cargo feature.
