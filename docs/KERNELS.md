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
        pub fn kernel(&self) -> KernelSource {
            match self {
                KernelLibrary::ElemwiseAdd => kernel_lib!("elemwise_add"),
                KernelLibrary::ElemwiseMul => kernel_lib!("elemwise_mul"), // <-- Add this
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
use crate::{CommandBuffer, TensorElement, operation::{ComputeKernelEncoder}, context::GpuProfilerLabel};

// 1. Public, user-facing, zero-sized struct for the operation.
pub struct ElemwiseMulOp;

// 2. Internal struct that holds data for the `Operation` trait.
struct ElemwiseMul<T: TensorElement> {
    a: Tensor<T>,
    b: Tensor<T>,
    out: Tensor<T>,
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    profiler_label: GpuProfilerLabel,
}

// 3. Implement `DefaultKernelInvocable` for the public struct (updated trait).
impl DefaultKernelInvocable for ElemwiseMulOp {
    // Input arguments for the call with generic tensor element type.
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);
    // The output type.

    // Link to the enum variant in `KernelFunction`.
    fn function_id() -> Option<KernelFunction> {
        Some(KernelFunction::ElemwiseMul)
    }

    // This `new` method is called by `ctx.call()`.
    // It creates the output tensor and the internal `Operation` struct.
    fn new<'a, T: TensorElement>(
        ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        pipeline: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (a, b) = args;

        // Prepare tensors for the active command
        ctx.prepare_tensors_for_active_cmd(&[&a, &b])?;
        
        // Create the output tensor.
        let out = Tensor::new(a.dims().to_vec(), TensorStorage::Pooled(ctx), TensorInit::Uninitialized)?;

        // Create a GPU profiler label for debugging/timing
        let profiler_label = ctx
            .take_gpu_scope()
            .unwrap_or_else(|| GpuProfilerLabel::fallback("elemwise_mul_op"));

        // Create the internal operation struct.
        let op = ElemwiseMul {
            a,
            b,
            out: out.clone(),
            pipeline: pipeline.expect("Kernel Module should be supplied from our kernel library"),
            profiler_label,
        };

        // Return the boxed operation and the output tensor.
        Ok((Box::new(op), out))
    }
}

// 4. Implement `Operation` for the internal struct.
// This contains the low-level logic to encode the kernel onto the command buffer.
impl<T: TensorElement> Operation for ElemwiseMul<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        // Use the new ComputeKernelEncoder for structured encoding
        ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
            .pipeline(&self.pipeline)
            .bind_kernel(self)  // Bind kernel arguments using the new method
            .dispatch_1d(self.a.len() as u32, 256);
        
        Ok(())
    }

    // NEW: Implement bind_kernel_args to handle the kernel argument binding
    fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
        use crate::encoder::{set_buffer, set_bytes};
        
        // Bind input tensors as buffers to the compute encoder
        set_buffer(encoder, 0, &self.a.buf, self.a.offset);
        set_buffer(encoder, 1, &self.b.buf, self.b.offset);
        set_buffer(encoder, 2, &self.out.buf, self.out.offset);
        // Bind additional scalar parameters if needed
        set_bytes(encoder, 3, &(self.a.len() as u32));
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
    use metallic::kernels::my_kernel::MyKernelOp;
    use metallic::{Context, Tensor, TensorInit, TensorStorage};

    #[test]
    fn test_my_kernel_logic() -> Result<(), MetalError> {
        let mut ctx = Context::new()?;
        let a = Tensor::new(vec![2], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&[1., 2.]))?;
        let b = Tensor::new(vec![2], TensorStorage::Dedicated(&ctx), TensorInit::CopyFrom(&[3., 4.]))?;

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
use metallic::kernels::my_kernel::MyKernelOp;

// ...

fn some_function(ctx: &mut Context, tensor_a: Tensor, tensor_b: Tensor) -> Result<Tensor, MetalError> {
    // Simply call the kernel through the generic `Context::call` method.
    let result = ctx.call::<MyKernelOp>((tensor_a, tensor_b))?;
    Ok(result)
}
```

## 4. New ComputeKernelEncoder Pattern

The project now uses a new structured approach for encoding Metal kernels using `ComputeKernelEncoder` and the `bind_kernel_args` method. This modern approach provides several benefits:

### ComputeKernelEncoder Benefits
- **Structured encoding**: Provides a fluent API for kernel encoding with method chaining
- **Built-in profiling**: Automatically handles GPU profiling scope creation and cleanup
- **Standardized dispatch**: Consistent dispatch patterns (1D, 2D, 3D) with proper threadgroup calculations
- **Reduced boilerplate**: Eliminates manual encoder management and error handling

### Key Components

1. **ComputeKernelEncoder**: A helper struct that provides a clean API for encoding compute kernels:
   ```rust
   ComputeKernelEncoder::new(command_buffer, &self.profiler_label)?
       .pipeline(&self.pipeline)                    // Set the compute pipeline
       .bind_kernel(self)                           // Bind kernel arguments using bind_kernel_args
       .dispatch_1d(self.total_elements as u32, 256); // Dispatch with specified threadgroup size
   ```

2. **bind_kernel_args method**: A required method in the `Operation` trait that handles argument binding:
   ```rust
   fn bind_kernel_args(&self, encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>) {
       use crate::encoder::{set_buffer, set_bytes};
       
       // Bind input tensors as buffers
       set_buffer(encoder, 0, &self.input1.buf, self.input1.offset);
       set_buffer(encoder, 1, &self.input2.buf, self.input2.offset);
       set_buffer(encoder, 2, &self.output.buf, self.output.offset);
       
       // Bind scalar parameters if needed
       set_bytes(encoder, 3, &self.scalar_param);
   }
   ```

### Available ComputeKernelEncoder Methods
- `pipeline(&pipeline)`: Set the compute pipeline state
- `bind_kernel(&operation)`: Bind arguments using the operation's `bind_kernel_args` method
- `buffer(index, buffer, offset)`: Bind a buffer directly (fallback option)
- `bytes(index, data)`: Bind bytes directly (fallback option) 
- `dispatch_1d(total_elements, threads_per_threadgroup)`: Dispatch 1D grid
- `dispatch_2d(width, height, threads_per_tg)`: Dispatch 2D grid
- `dispatch_3d(width, height, depth, threads_per_tg)`: Dispatch 3D grid
- `dispatch_custom(groups, threads_per_tg)`: Dispatch with custom parameters
- `dispatch_threads(grid_size, threadgroup_size)`: Direct dispatch control

## Graph-backed Kernels

Several kernels now have MPSGraph-backed implementations that sit alongside the legacy Metal
paths. When promoting a kernel to run through MPSGraph, follow this checklist to keep performance
and telemetry consistent:

- **Implement `GraphKernel`**: create a zero-sized type that implements both `KernelInvocable` and
  the new `GraphKernel` trait. Encode storage and accumulator precision through
  `GraphKernelDtypePolicy` (e.g., f16 storage with fp32 accumulators) so cache keys remain stable.
- **Publish signatures**: override `GraphKernel::signature()` so axis semantics, optional bindings,
  and notes are discoverable by tooling and future kernel ports.
- **Dispatch via the registry**: introduce a `*DispatchOp` that queries `KernelBackendRegistry` and
  selects the appropriate backend. The dispatcher automatically honors
  `METALLIC_FORCE_SDPA_BACKEND=legacy|mpsgraph|auto` so developers can toggle behaviour without
  code changes.
- **Expose overrides**: surface CLI/config toggles (e.g., `--sdpa-backend`) that map onto the
  registry, enabling per-run backend selection without mutating global environment variables.
- **Use shared caches**: request executables through `ResourceCache::get_or_create_mpsgraph_*`.
  The underlying `GraphExecutableCache` and `MaskArena` abstractions handle instrumentation and
  reuse; avoid ad-hoc maps for graph resources.
- **Add parity coverage**: extend `metallic::tests` with fixtures that compare graph vs. legacy
  outputs and assert that env overrides flip the dispatcher as expected.
- **Document invariants**: note any graph-specific constraints (mask semantics, stride
  requirements, accumulator modes) directly alongside the kernel so future ports stay aligned.

Following this pattern keeps DX uniform: every kernel exposes a single entry point, reports backend
selection through `KernelBackendSelected` events, and shares graph executables through the reusable
cache layers introduced in Milestone C.

## Metal Kernel Compilation System (Source vs Precompiled)

The project now supports both source-based and precompiled Metal kernel compilation via a flexible macro-based system that can be configured at build time.

### Two Kernel Source Types

The system supports two different ways of loading Metal kernels:

1. **Source-based Kernels** (`src_kernels` feature):
   - Kernels are loaded as text source code at runtime using `include_str!`
   - Compiled each time the application starts
   - Slower startup time but easier debugging
   - Default in debug builds

2. **Precompiled Binary Kernels** (`built_kernels` feature):
   - Kernels are precompiled to `.metallib` files during build
   - Loaded at runtime as binary using `include_bytes!`
   - Faster startup time with pre-validated compilation
   - Default in release builds

### KernelSource Enum

The system uses a `KernelSource` enum to represent either type:

```rust
pub enum KernelSource {
    Text(&'static str),     // For source-based compilation
    Binary(&'static [u8]),  // For precompiled binary kernels
}
```

### Build Configuration

The build process is controlled by feature flags and build mode:

- `--features src_kernels`: Forces source-based compilation
- `--features built_kernels`: Forces precompiled binary compilation
- In debug builds: defaults to source-based compilation
- In release builds: defaults to precompiled binary compilation

The `build.rs` script automatically compiles `.metal` files to `.metallib` files during the build process when precompiled binaries are enabled.

### kernel_lib! Macro

The `kernel_lib!` macro abstracts the runtime choice between precompiled `.metallib` binaries and inline source kernels. Its most recent form accepts an optional *extra* list of Metal sources:

The standard single kernel file version like below will include_str in debug and build and include_bytes the matmul/metal.kernel...

kernel_lib!("matmul"); 

When you have kernels that rely on shared loader logic, epilog helpers, or otherwise span multiple `.metal` files (for example `common/gemv_core.metal` + `matmul_q8_gemv/kernel_body.metal`), call `kernel_lib!` with the shared sources listed after the main library name:

```rust
kernel_lib!(
    "matmul_q8_gemv",
    "common/gemv_core.metal",
    "matmul_q8_gemv/kernel_body.metal"
)
```

In debug/src-kernel mode the macro concatenates the listed extras (no `kernel.metal`) so the runtime compiler sees a single blob without `#include`s. In release/built-kernel mode these extras are ignored because the `.metallib` already contains everything.

In release/built-kernels mode the macro will include_bytes the generated kernel, but the build.rs needs to know the files to include, as such the above kernel_lib!() will require a sources file list for the build.rs script to use (we haven't created a proc macro to do this automatically as part of kernel_lib!() execution.

metal.sources under matmul_q8_gemv per the above example would have the following text:

```
common/gemv_core.metal
matmul_q8_gemv/kernel_body.metal
```