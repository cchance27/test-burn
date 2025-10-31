# Dynamic Fusion System (MetallicGraph)

## Overview

This document outlines the new dynamic graph building system that enables runtime composition of multiple kernels into fused executables. The system, named `MetallicGraph`, allows for dynamic graph construction, reducing code duplication and enabling proper fusion of arbitrary kernel sequences.

## Core Concept

The Dynamic Fusion System provides:

1. **Runtime graph composition** - Build graphs with multiple operations at runtime
2. **Single kernel codebase** - No need for separate graph vs. non-graph implementations
3. **Automatic fallback** - Non-fused execution when fusion isn't desired
4. **Caching integration** - Leverages existing resource caching infrastructure

## Architecture

### Core Components

#### 1. `MetallicGraph` - Dynamic Graph Builder
```rust
pub struct MetallicGraph {
    graph: Retained<mpsg::MPSGraph>,        // Underlying MPSGraph object
    nodes: Vec<GraphNode>,                  // Graph operation nodes  
    tensors: Vec<GraphTensor>,              // Graph tensor references
    data_type: MPSDataType,                 // Graph data type
    next_tensor_id: usize,                  // Counter for tensor IDs
    next_node_id: usize,                    // Counter for node IDs
}
```

#### 2. `GraphCompatibleKernel` Trait
```rust
pub trait GraphCompatibleKernel {
    type GraphArgs;
    
    // Add this kernel to the graph and return output tensors
    fn add_to_graph(graph: &mut MetallicGraph, args: Self::GraphArgs) -> Result<Vec<GraphTensor>, MetalError>;
    
    // Create fallback operation for non-graph backend execution to use the standard KernelInvocable kernel ctx.call
    fn create_fallback_operation<T: TensorElement>(
        tensors: Vec<Tensor<T>>,
        ctx: &mut Context<T>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError>;
}
```

#### 3. `FusionOp` - Execution Interface
```rust
pub struct FusionOp;

impl KernelInvocable for FusionOp {
    type Args<'a, T: TensorElement> = (MetallicGraph, Vec<Tensor<T>>, Vec<Tensor<T>>);
    // ...
}
```

### Usage Pattern

```rust
// Create a new dynamic graph
let mut g = MetallicGraph::new(MPSDataType::Float16)?;

// Add operations dynamically
let input1_idx = g.create_input_placeholder(&[1, 8, 128], Some("query"))?;
let input2_idx = g.create_input_placeholder(&[1, 8, 128], Some("key"))?;
let input3_idx = g.create_input_placeholder(&[1, 8, 128], Some("value"))?;

// Dynamic composition - any sequence of operations
let sdpa_result = g.add_kernel_call::<SdpaOp>((input1_idx, input2_idx, input3_idx))?;
let proj_result = g.add_kernel_call::<MatmulOp>((sdpa_result[0], projection_weights_idx))?;

// Execute the fused graph
let result = ctx.call::<FusionOp>((g, input_tensors, output_tensors))?;
```

## Implementation Plan

### Phase 1: Core Infrastructure
- [ ] Create `crates/metallic/src/kernels/fusion/mod.rs`
- [ ] Implement `MetallicGraph` structure and basic operations
- [ ] Define `GraphCompatibleKernel` trait
- [ ] Create `FusionOp` kernel and operation

### Phase 2: Kernel Integration
- [ ] Update existing kernels to optionally implement `GraphCompatibleKernel`
- [ ] Add graph compatibility to core kernels (elemwise operations, matmul, sdpa)
- [ ] Ensure fallback mechanism works properly

### Phase 3: Advanced Features
- [ ] Graph caching based on topology and parameters
- [ ] Automatic graph optimization opportunities
- [ ] Performance monitoring and telemetry for fused operations

### Phase 4: Testing and Validation
- [ ] Unit tests for dynamic graph building
- [ ] Performance comparisons with existing approaches
- [ ] Validation that fused and non-fused execution produce identical results

## Benefits Over Current Approach

### 1. Reduced Code Duplication
- **Before**: Separate graph and non-graph implementations
- **After**: Single kernel implementation supports both modes

### 2. Dynamic Composition
- **Before**: Static fusions (e.g., SDPA+Projection)
- **After**: Arbitrary runtime composition of operations

### 3. Flexibility
- **Before**: Fixed operation sets per fused kernel
- **After**: Any sequence of graph-compatible operations

### 4. Maintainability
- **Before**: Separate code paths increase maintenance burden
- **After**: Single codebase with optional graph support

## Integration with Existing Systems

### Resource Caching
The fusion system integrates with the existing `ResourceCache`:
- New cache entries for fused executables based on graph topology
- Leverages existing cache key generation and eviction policies
- Maintains same performance characteristics

### Context Integration
- `ctx.call::<FusionOp>()` follows existing patterns
- No changes to existing kernel calling interface
- Backward compatibility maintained

### Telemetry and Profiling
- Graph fusion operations report as `FusionOp` with topology information
- Performance metrics track fusion effectiveness
- Existing profiling infrastructure continues to work

## Migration Path

### For Kernel Developers
```rust
// Existing kernel
impl KernelInvocable for MyKernelOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>);
    // existing implementation...
}

// Enhanced for fusion support
impl<T: TensorElement> GraphCompatibleKernel for MyKernelOp {
    type GraphArgs = (usize, usize); // graph tensor indices
    
    fn add_to_graph(
        graph: &mut MetallicGraph, 
        args: Self::GraphArgs
    ) -> Result<Vec<GraphTensor>, MetalError> {
        // Add kernel to graph using MPSGraph APIs
        // Return output tensor references
    }
    
    fn create_fallback_operation<T: TensorElement>(
        tensors: Vec<Tensor<T>>,
        ctx: &mut Context<T>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        // Fall back to regular kernel execution
    }
}
```

### For Users
```rust
// No changes to regular kernel usage
let result = ctx.call::<MyKernelOp>((tensor_a, tensor_b))?;

// New fusion capability
let mut g = MetallicGraph::new(dtype)?;
// ... build graph dynamically
let fused_result = ctx.call::<FusionOp>((g, inputs, outputs))?;
```

---

## Reference: Comparison with Existing Systems

### 1. **Existing SDPA MPS Graph Implementation**

**What it does:**
- **Single-purpose**: Only handles SDPA operation
- **Static compilation**: Each SDPA configuration requires separate compilation
- **Fixed topology**: SDPA operation only, no fusion with other operations
- **Direct tensor binding**: Maps tensors directly to graph placeholders

**Limitations:**
- Not reusable for other operations
- No dynamic composition
- Separate code path from regular kernels
- Cannot compose multiple operations

### 2. **Existing MultiOp Platform**

**What it does:**
- **Pre-built fusions**: SDPA+Projection is a specific fused graph
- **Static compilation**: Built for specific combinations
- **Limited flexibility**: New fusions require new implementations

**Limitations:**
- Not dynamic - all operations must be known at compile time
- Still requires separate implementations per fusion type
- No general-purpose composition framework

### 3. **Proposed Dynamic Graph Building System**

#### Key Advantages:

**A. Dynamic Composition at Runtime:**
```rust
// PROPOSED SYSTEM - can compose dynamically
let mut g = MetallicGraph::new(mps_data_type)?;
let input1_idx = g.create_input_placeholder(&[1, 8, 128], Some("input1"))?;
let input2_idx = g.create_input_placeholder(&[1, 8, 128], Some("input2"))?;

// Add operations dynamically
let add_result = g.add_kernel_call::<ElemwiseAddOp>((input1_idx, input2_idx))?;
let mul_result = g.add_kernel_call::<ElemwiseMulOp>((add_result[0], input2_idx))?;

let result = ctx.call::<FusionOp>((g, input_tensors, output_tensors))?;
```

**B. Reuse Existing Kernels:**
- No need to duplicate kernel logic
- Existing kernels can implement `GraphCompatibleKernel` to support fusion
- Single code path for both fused and non-fused execution

**C. General-Purpose Framework:**
- Works with ANY operation that supports graph fusion
- Not limited to specific combinations like SDPA+Projection
- Can compose arbitrary sequences of operations

**D. Backward Compatibility:**
- Falls back to regular `ctx.call::<Kernel>()` when not fusing
- Same API for users: `ctx.call::<FusionOp>(...)`
- Can be implemented without breaking existing code

### 4. **Detailed Comparison Table:**

| Feature | Existing SDPA Graph | MultiOp | Proposed System |
|---------|-------------------|---------|-------------------|
| **Flexibility** | Fixed (SDPA only) | Fixed (specific fusions) | Dynamic (any combination) |
| **Composition** | No | Pre-built fusions | Dynamic runtime composition |
| **Code Duplication** | High (separate graph version) | High (separate implementations) | Low (reuse existing kernels) |
| **Compile Time** | Per configuration | Per fusion type | Once for framework |
| **API** | `ctx.call::<SdpaMpsGraphOp>()` | `ctx.call::<SpecificFusionOp>()` | `ctx.call::<FusionOp>()` |
| **Caching** | Per SDPA config | Per fusion type | Per graph topology |
| **Maintenance** | Separate code paths | More complex with each new fusion | Single framework to maintain |

### 5. **How the System Solves Original Problems:**

**"A lot of duplicated code":** ✅ **SOLVED**
- No duplicate kernel logic - reuse existing kernels
- Single framework handles all fusion logic

**"Not setup well for building up executable for proper fusion":** ✅ **SOLVED** 
- Dynamic graph building allows any sequence of operations
- Automatic executable compilation when graph is built

**"SDPA MPS kernel isn't properly built for fusion":** ✅ **SOLVED**
- The SDPA kernel doesn't need to change
- If it implements `GraphCompatibleKernel`, it can be part of any fusion

**"Minimize reuse where possible so we don't have many graph files AND normal kernel versions":** ✅ **SOLVED**
- Single kernel implementation that works for both cases
- Graph capability is an extension of existing design

### 6. **Integration Strategy:**

The key insight is that the system works WITH the existing architecture rather than replacing it:

- **Uses existing `KernelInvocable` framework** - no breaking changes
- **Extends existing graph capabilities** - builds on MultiOp concepts
- **Maintains same resource caching** - plugs into existing `ResourceCache`
- **Allows gradual adoption** - kernels can opt-in to fusion support

In essence, the system provides a **unified, dynamic framework** while the existing implementations are **specialized, static solutions**. The proposed system can do everything the existing ones do, plus arbitrary fusion that they cannot achieve.