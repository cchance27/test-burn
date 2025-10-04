# metallic_macros

The `metallic_macros` crate exposes the `metal_kernel!` declarative macro that generates the
boilerplate required to register and invoke Metal compute kernels inside the `metallic`
backend.  It eliminates the need to hand-write the `KernelInvocable` and `Operation`
implementations for each kernel, allowing authors to focus on the math and command-encoder
logic.

## Declaring a library

Each macro invocation begins with the Metal library metadata.  Provide a `source` expression
(`include_str!(...)` is common) along with one or more exported functions.  Map every `Dtype`
you support to the corresponding Metal entry point name so runtime dispatch can resolve the
proper symbol.

```rust
metal_kernel! {
    library ElemwiseAdd {
        source: include_str!("kernel.metal"),
        functions: {
            ElemwiseAdd => {
                F32 => "add_kernel_f32",
                F16 => "add_kernel_f16",
            },
        },
        operations: { /* ... */ },
    }
}
```

The macro emits `KernelDescriptor`/`KernelFunctionDescriptor` statics that mirror the existing
`KernelLibrary`/`KernelFunction` enums so that the kernel manager can load pipelines using the
new descriptor flow.

## Defining operations

For each operation exported by the library, specify the argument tuple, whether a Metal
pipeline is required, the state fields to persist between `new` and `encode`, and the bodies of
the `new` and `encode` closures.

```rust
operations: {
    ElemwiseAddOp => {
        function: Some(KernelFunction::ElemwiseAdd),
        args: (Tensor<T>, Tensor<T>),
        pipeline: required,
        state: {
            a: Tensor<T>,
            b: Tensor<T>,
            out: Tensor<T>,
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        },
        new: |ctx, (a, b), pipeline, _cache| {
            ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;
            let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;
            Ok((ElemwiseAddOpState { a, b, out: out.clone(), pipeline }, out))
        },
        encode: |command_buffer, _cache, state| {
            // existing command encoding logic goes here
            Ok(())
        },
    },
}
```

The macro expands the snippet above into:

- A `pub struct ElemwiseAddOp` that implements `KernelInvocable`.
- A concrete `ElemwiseAddOpState<T>` that implements `Operation` and captures the fields listed
  in the `state` block.
- Boilerplate that wires the closure parameters into the generated implementations, performs
  pipeline validation, and boxes the operation for dynamic dispatch.

## Integrating with the existing kernels

To migrate a legacy module:

1. Import `metal_kernel!` and replace the handwritten structs and impls with a single macro
   invocation.
2. Keep any existing tests unchanged aside from their import paths—the generated types retain
   the original operation names.
3. Re-export the operations from the module if they were previously surfaced elsewhere.

With this scaffolding in place you can gradually port each Metal kernel to the macro-based
workflow while the runtime continues to consume the familiar `Context::call::<Op>` API.
