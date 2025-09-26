# Adding New Metal Kernels

This document outlines the standard procedure for adding new custom Metal compute kernels to the project. Following this guide ensures that new kernels are well-integrated, easy to use, and maintainable.

## 1. File Structure

All custom kernels reside in the `src/metallic/kernels/` directory. Each kernel or family of related kernels should be placed in its own module (a subdirectory).

The required structure for a new kernel named `my_kernel` is as follows:

```
src/metallic/kernels/
├── my_kernel/
│   ├── mod.rs                    # Rust logic for MyKernelOp and trait implementations, mod and pub use to export MyKernelOtherOp
│   ├── my_kernel_test.rs         # Tests for the main MyKernelOp
│   └── my_kernel_other.rs        # Additional Ops that exist in the kernel.metal beyond the main MyKernelOp
│   └── my_kernel_other_test.rs   # Tests for the MyKernelOtherOp
│   └── kernel.metal              # Raw Metal Shading Language (MSL) code (for all of this kernels functions)
└── ...
```

## 2. Step-by-Step Guide

Let's add a simple element-wise multiplication kernel as an example.

### Step 1: Create the Kernel Files (Needed for Raw Metal Kernels)

First, create the directory and the two files:

1.  `src/metallic/kernels/elemwise_mul/`
2.  `src/metallic/kernels/elemwise_mul/kernel.metal`
3.  `src/metallic/kernels/elemwise_mul/mod.rs`

### Step 2: Write the Metal Code

Place your MSL function(s) in `kernel.metal`.

**`src/metallic/kernels/elemwise_mul/kernel.metal`**:
```metal
#include <metal_stdlib>
using namespace metal;

kernel void mul_kernel(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* out [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    out[gid] = a[gid] * b[gid];
}
```

### Step 3: Update Kernel Enums (Required for Raw Metal Kernel style kernels, MPS call kernels don't require these enums)

Edit `src/metallic/kernels/mod.rs` to make the kernel management system aware of your new kernel.

1.  **`KernelLibrary` enum**: Add a variant for your new kernel module. This corresponds to the `.metal` file.

    ```rust
    // Export our kernels
    pub mod elemwise_add;
    pub mod elemwise_mul; // <-- Add this

    pub enum KernelLibrary {
        ElemwiseAdd,
        ElemwiseMul, // <-- Add this
    }

    impl KernelLibrary {
        fn source(&self) -> &'static str {
            match self {
                KernelLibrary::ElemwiseAdd => include_str!("elemwise_add/kernel.metal"),
                KernelLibrary::ElemwiseMul => include_str!("elemwise_mul/kernel.metal"), // <-- Add this
            }
        }
    }
    ```

2.  **`KernelFunction` enum**: Add a variant for each `kernel` function in your `.metal` file. (Required for Raw Metal Kernels Only)

    ```rust
    pub enum KernelFunction {
        ElemwiseAdd,
        ElemwiseBroadcastAdd,
        ElemwiseMul, // <-- Add this
    }

    impl KernelFunction {
        fn library(&self) -> KernelLibrary {
            match self {
                Self::ElemwiseAdd | Self::ElemwiseBroadcastAdd => KernelLibrary::ElemwiseAdd,
                Self::ElemwiseMul => KernelLibrary::ElemwiseMul, // <-- Add this
            }
        }

        fn name(&self) -> &'static str {
            match self {
                Self::ElemwiseAdd => "add_kernel",
                Self::ElemwiseBroadcastAdd => "broadcast_add_kernel",
                Self::ElemwiseMul => "mul_kernel", // <-- Add this
            }
        }
    }
    ```

### Step 4: Implement the Rust Logic

In your kernel's `mod.rs`, you will implement the `KernelInvocable` trait. This trait connects the high-level API (`ctx.call<T>()`) to your kernel's specific implementation.

**`src/metallic/kernels/elemwise_mul/mod.rs`**:

```rust
// You can pull in super::* to pull in most imports required for kernel creation to keep kernel rust files small.
use super::*;

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct ElemwiseMulOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct ElemwiseMul {
    a: Tensor,
    b: Tensor,
    out: Tensor,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

// 3. Implement `KernelInvocable` for the public struct.
impl KernelInvocable for ElemwiseMulOp {
    // Input arguments for the call.
    type Args = (Tensor, Tensor);
    // The output type.
    type Output = Tensor;

    // Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseMul)
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new(
        ctx: &mut Context,
        args: Self::Args,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    ) -> Result<(Box<dyn Operation>, Self::Output), MetalError> {
        let (a, b) = args;

        // Create the output tensor.
        let out = Tensor::create_tensor_pooled(a.dims().to_vec(), ctx)?;

        // Create the internal operation struct.
        let op = ElemwiseMul {
            a,
            b,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Module should be supplied from our kernel library"),
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), out))
    }
}

// 4. Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl Operation for ElemwiseMul {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        _cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // ... encoding logic (set buffers, dispatch threads, etc.)
        Ok(())
    }
}
```

### Step 5: Add Tests (Recommended)

To maintain quality, add a test file for your new kernel.

1.  Create `src/metallic/kernels/my_kernel/my_kernel.test.rs`.
2.  In `src/metallic/kernels/my_kernel/mod.rs`, add the following to include the test file:
    ```rust
    mod my_kernel_test;
    ```
3.  Write a test using the `ctx.call` API.

    **`my_kernel.test.rs`**:
    ```rust
    #![cfg(test)]
    use crate::metallic::kernels::my_kernel::MyKernelOp;
    use crate::metallic::{Context, Tensor};

    #[test]
    fn test_my_kernel_logic() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        let a = Tensor::create_tensor_from_slice(&[1., 2.], vec![2], &ctx)?;
        let b = Tensor::create_tensor_from_slice(&[3., 4.], vec![2], &ctx)?;

        // Use the kernel via the generic `call` method.
        let result_tensor = ctx.call::<MyKernelOp>((a, b))?;
        ctx.synchronize();

        assert_eq!(result_tensor.as_slice(), &[3.0, 8.0]);
        Ok(())
    }
    ```

## 3. Usage

Once a kernel is implemented according to this pattern, using it from anywhere in the application is simple, clean, and type-safe.

```rust
// Make sure the Op struct is in scope.
use crate::metallic::kernels::my_kernel::MyKernelOp;

// ...

fn some_function(ctx: &mut Context, tensor_a: Tensor, tensor_b: Tensor) -> Result<Tensor, MetalError> {
    // Simply call the kernel through the generic `Context::call` method.
    let result = ctx.call::<MyKernelOp>((tensor_a, tensor_b))?;
    Ok(result)
}
```
