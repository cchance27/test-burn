# Resource Cache Nesting Guidance

This document records the rationale behind the recent helper additions that keep a
`ResourceCache` alive across nested kernel invocations. Prior to these changes,
`Context::call` would temporarily take ownership of the active cache, but any
composite kernel that made additional `ctx.call` invocations would drop that
borrow instead of propagating it. Each inner call therefore created a fresh
cache, defeating descriptor reuse for sequences of GEMMs and MPS kernels.

## Helper APIs

To make correct propagation ergonomic, `Context` now exposes two internal
helpers:

- `call_with_cache` mirrors `Context::call` but accepts an explicit
  `&mut ResourceCache` borrow. Composite kernels use this to route the
  outer cache into nested `KernelInvocable` calls.
- `matmul_alpha_beta_with_cache` is a convenience wrapper that forwards the
  cache borrow into `MatMulAlphaBetaOp`, letting attention and FFN kernels keep
  their `MPSMatrixMultiplication` instances warm between dispatches.

Both helpers assume the caller already has an active command buffer and cache,
which the outermost `Context::call` or manually prepared operation guarantees.

## Expectations for Future Kernels

When implementing composite Metal kernels, prefer the `*_with_cache` helpers for
any nested dispatches. This ensures a single `ResourceCache` instance is shared
throughout the call graph, avoiding redundant pipeline creation and reducing
latency.

As a follow up, audit new kernels to confirm they use these helpers when making
nested calls. Doing so maintains consistent cache reuse across the codebase.
