# Automatic Kernel Fusion Design Proposal

## Overview

This document proposes implementing an automatic kernel fusion system that maintains the existing API while enabling dynamic fusion of compatible operations under the hood. This system would work as a layer on top of the current architecture to improve performance without breaking existing functionality.

## Current Architecture

The existing system has:
- **Modular Kernels**: Each operation follows a standardized pattern with `KernelInvocable` trait
- **Batch Recording**: Already implemented via `CommandBuffer::record_batch` for reducing encoder overhead
- **Immediate Execution**: Operations are executed as soon as `ctx.call()` is invoked

## Proposed Fusion Approach

### 1. Fusion Primitives and Graph Structure

Define core structures for representing operations that can be fused:

```rust
#[derive(Debug, Clone)]
pub struct FusionNode<T: TensorElement> {
    pub operation: FusionOpType<T>,
    pub output_tensors: Vec<Tensor<T>>,
    pub input_tensors: Vec<Tensor<T>>,
}

#[derive(Debug, Clone)]
pub enum FusionOpType<T: TensorElement> {
    ElementWise(ElementWiseOp<T>),
    MemoryOp(MemoryOp<T>),
    // Add more fusion-compatible operations
}

#[derive(Debug, Clone)]
pub enum ElementWiseOp<T: TensorElement> {
    Add { a: Tensor<T>, b: Tensor<T> },
    Mul { a: Tensor<T>, b: Tensor<T> },
    Sigmoid { x: Tensor<T> },
    // etc.
}

#[derive(Debug, Clone)]
pub enum MemoryOp<T: TensorElement> {
    Copy { src: Tensor<T>, dst: Tensor<T> },
    Fill { dst: Tensor<T>, value: f32 },
}
```

### 2. Fusion Graph Manager

Manage sequences of operations that could potentially be fused:

```rust
pub struct FusionGraph<T: TensorElement> {
    nodes: Vec<FusionNode<T>>,
    dependencies: FxHashMap<usize, Vec<usize>>, // node_id -> [dependency_ids]
    tensor_to_node: FxHashMap<*const u8, usize>, // tensor ptr -> producing node_id
}

impl<T: TensorElement> FusionGraph<T> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dependencies: FxHashMap::default(),
            tensor_to_node: FxHashMap::default(),
        }
    }

    pub fn add_node(&mut self, node: FusionNode<T>) -> usize {
        let node_id = self.nodes.len();
        // Track tensor producers
        for tensor in &node.output_tensors {
            let tensor_ptr = Retained::as_ptr(&tensor.buf) as *const u8;
            self.tensor_to_node.insert(tensor_ptr, node_id);
        }
        
        // Track dependencies
        let mut deps = Vec::new();
        for tensor in &node.input_tensors {
            let tensor_ptr = Retained::as_ptr(&tensor.buf) as *const u8;
            if let Some(&producer_id) = self.tensor_to_node.get(&tensor_ptr) {
                deps.push(producer_id);
            }
        }
        self.dependencies.insert(node_id, deps);
        
        self.nodes.push(node);
        node_id
    }

    pub fn can_fuse(&self, node_ids: &[usize]) -> bool {
        // Check if nodes can be fused (no dependencies between them, same compute characteristics)
        true // Simplified for now
    }

    pub fn optimize(&mut self) -> Vec<FusionNode<T>> {
        // Analyze the graph and return optimized fusion groups
        // This is where fusion rules would be applied
        vec![] // Simplified
    }
}
```

### 3. Fusion-Aware Context

Modify the Context to support fusion when appropriate:

```rust
impl<T: TensorElement> Context<T> {
    /// Same public API, but with fusion capability
    pub fn call_fusion_aware<K: KernelInvocable>(&mut self, args: K::Args<'_, T>) -> Result<Tensor<T>, MetalError> {
        // Check if this operation should be added to a fusion graph
        // vs executed immediately (e.g., based on operation type or fusion policy)
        
        // For element-wise operations, add to fusion graph if possible
        if self.should_attempt_fusion::<K>() {
            self.add_to_fusion_graph::<K>(args)
        } else {
            // Execute immediately as before
            self.call(args)
        }
    }

    fn should_attempt_fusion<K: KernelInvocable>(&self) -> bool {
        // Determine based on operation type, tensor sizes, etc.
        // Could be configured via environment variables or context settings
        true // For now, attempt fusion for compatible operations
    }

    fn add_to_fusion_graph<K: KernelInvocable>(&mut self, args: K::Args<'_, T>) -> Result<Tensor<T>, MetalError> {
        // Add operation to the fusion graph
        // Execute the fusion when conditions are met (buffer full, sync point, etc.)
        unimplemented!()
    }
}
```

### 4. Dynamic Kernel Generator

Create a system that generates fused kernels dynamically:

```rust
pub struct KernelGenerator;

impl KernelGenerator {
    pub fn generate_fused_kernel(nodes: &[FusionNode<T>]) -> String {
        // Generate MSL source code for fused operations
        let mut msl = String::from("#include <metal_stdlib>\nusing namespace metal;\n\n");
        
        // Create the fused function
        msl.push_str(&format!(
            "kernel void fused_kernel_{unique_id}(\n    // fused parameters\n)"
        ));
        
        // Combine the logic from multiple operations
        // This is the complex part - generating efficient fused MSL
        
        msl
    }
}
```

## Benefits

1. **Backwards Compatibility**: Existing API remains unchanged
2. **Automatic**: Fusion happens automatically when beneficial
3. **Modular**: Each operation remains modular but can be fused when appropriate
4. **Performance**: Eliminates intermediate buffers and reduces kernel launch overhead
5. **Extensible**: New fusion patterns can be added over time

## Implementation Strategy

### Phase 1: Basic Fusion Graph
- Implement fusion graph tracking for simple element-wise operations
- Add fusion decisions based on operation compatibility

### Phase 2: Dynamic Kernel Generation
- Add code generation for fused kernels
- Create basic fusion templates

### Phase 3: Advanced Fusion Rules
- Implement complex fusion patterns (matmul + add, etc.)
- Add heuristics for when fusion is beneficial

### Phase 4: Profiling and Auto-tuning
- Add performance monitoring to decide when to apply fusion
- Implement auto-tuning based on workload characteristics

## Example Use Case

The current operation sequence:
```rust
let a_plus_b = ctx.call::<AddOp>((a, b))?;      // Separate kernel
let result = ctx.call::<MulOp>((a_plus_b, c))?; // Separate kernel
```

Would become:
```rust
// Same API, but internally fused if beneficial:
let a_plus_b = ctx.call::<AddOp>((a, b))?;      // Added to fusion graph
let result = ctx.call::<MulOp>((a_plus_b, c))?; // Fused with previous operation
// Fusion executed when buffer is full or sync point reached
```

This results in a single kernel executing both operations, eliminating the intermediate tensor and reducing kernel launch overhead.

## Potential Fusion Patterns

- **Element-wise chains**: Multiple element-wise operations (add, multiply, activation functions)
- **Matmul + bias**: Matrix multiplication followed by bias addition
- **Normalization + activation**: LayerNorm/RMSNorm followed by activation functions
- **Memory operations**: Combining multiple small memory operations

## Risks and Considerations

- **Complexity**: Dynamic kernel generation adds complexity
- **Memory**: May increase memory usage for kernel compilation
- **Debugging**: Fused kernels may be harder to debug
- **Compatibility**: Some operations may not be suitable for fusion