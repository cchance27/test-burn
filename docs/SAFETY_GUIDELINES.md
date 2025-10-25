# Safety Guidelines for Metal Kernel Development

## Overview

This document outlines the safety guidelines for working with Objective-C and Metal APIs in the Metallic kernel system. The guidelines emphasize the proper encapsulation of unsafe code through safe wrapper APIs.

## Core Principles

### 1. Centralize Unsafe Code
- All `unsafe` blocks must be encapsulated within safe wrapper functions
- Never expose raw `unsafe` calls to kernel implementation modules
- Validate inputs before making unsafe calls
- Handle errors appropriately within wrapper functions

### 2. Validate Before Unsafe Operations
- Always validate object invariants before making unsafe calls
- Check bounds, alignment, and type compatibility
- Provide meaningful error messages for invalid inputs
- Never assume external inputs are valid

### 3. Ergonomic APIs
- Design safe wrappers with clear, concise method names
- Use Rust conventions (snake_case, meaningful parameter names)
- Provide comprehensive error handling
- Maintain clear ownership and lifetime semantics

## Safe Wrapper Design Patterns

### A. Basic Object Wrapper Pattern
```rust
pub struct SafeMpsGraph {
    inner: Retained<mpsg::MPSGraph>,
}

impl SafeMpsGraph {
    pub fn new() -> Result<Self, MetalError> {
        // Safe construction with error handling
        let graph = unsafe { mpsg::MPSGraph::new() };
        Ok(Self { inner: graph })
    }
    
    pub fn create_placeholder(
        &self, 
        shape: &[usize], 
        data_type: mps::MPSDataType, 
        name: Option<&str>
    ) -> Result<SafeGraphTensor, MetalError> {
        // Input validation
        if shape.len() > MAX_DIMENSIONS {
            return Err(MetalError::InvalidShape("Shape exceeds maximum dimensions".into()));
        }
        
        // Safe parameter transformation
        let shape_array = create_shape_array(shape)?;
        let name_obj = name.map(|n| NSString::from_str(n));
        
        // Encapsulated unsafe call
        let tensor = unsafe {
            self.inner.placeholderWithShape_dataType_name(
                Some(&shape_array),
                data_type,
                name_obj.as_ref().map(|n| &**n),
            )
        };
        
        Ok(SafeGraphTensor { inner: tensor })
    }
}
```

### B. Input Validation Pattern
```rust
impl SafeMpsMatrix {
    pub fn from_buffer_descriptor(
        buffer: &ProtocolObject<dyn objc2_metal::MTLBuffer>,
        offset: usize,
        descriptor: &Retained<mps::MPSMatrixDescriptor>,
    ) -> Result<Self, MetalError> {
        // Validate offset alignment
        if offset % std::mem::size_of::<f32>() != 0 {
            return Err(MetalError::InvalidShape("Matrix offset must be aligned to float size".into()));
        }
        
        // Validate offset doesn't exceed buffer bounds
        if offset >= buffer.length() {
            return Err(MetalError::InvalidShape("Matrix offset exceeds buffer bounds".into()));
        }
        
        let matrix = unsafe {
            mps::MPSMatrix::initWithBuffer_offset_descriptor(
                mps::MPSMatrix::alloc(),
                buffer,
                offset as u64,
                descriptor,
            )
        };
        
        Ok(Self { inner: matrix })
    }
}
```

## Migration Guidelines

### 1. Identify Unsafe Code
Search for patterns in the codebase:
- `unsafe { ... }` blocks
- `initWith*` method calls
- `Retained::cast_unchecked` conversions
- Direct Objective-C API calls

### 2. Create Safe Alternatives
For each identified unsafe operation, create a safe wrapper:
- Validate inputs before the unsafe call
- Handle the result appropriately
- Return meaningful errors
- Provide ergonomic APIs

### 3. Replace Unsafe Usage
Gradually replace direct unsafe calls with safe wrapper calls:
- Update kernel implementations to use safe wrappers
- Ensure functionality remains identical
- Add comprehensive tests
- Document the changes

## Common Unsafe Patterns and Safe Alternatives

### A. Object Initialization
**Unsafe:**
```rust
let graph = unsafe { mpsg::MPSGraph::new() };
```

**Safe:**
```rust
let graph = SafeMpsGraph::new()?;
```

### B. Parameterized Object Creation
**Unsafe:**
```rust
let tensor = unsafe {
    graph.placeholderWithShape_dataType_name(Some(&shape), data_type, Some(&name))
};
```

**Safe:**
```rust
let tensor = graph.create_placeholder(shape, data_type, Some("input"))?;
```

### C. Array and Dictionary Conversions
**Unsafe:**
```rust
let array = unsafe { Retained::cast_unchecked(ns_mutable_array) };
```

**Safe:**
```rust
let array = create_shape_array(shape)?;  // Safe wrapper function
```

### D. Command Buffer Operations
**Unsafe:**
```rust
let mps_cmd_buffer = unsafe { 
    mps::MPSCommandBuffer::commandBufferWithCommandBuffer(metal_buffer) 
};
```

**Safe:**
```rust
let mps_cmd_buffer = SafeCommandBuffer::wrap_metal_buffer(metal_buffer)?;
```

## Error Handling Best Practices

### 1. Comprehensive Error Types
Use the existing `MetalError` enum to provide detailed error information:
- `MetalError::InvalidShape` for shape-related errors
- `MetalError::OperationFailed` for runtime failures
- `MetalError::ResourceExhausted` for resource issues

### 2. Input Validation
Always validate inputs before making unsafe calls:
- Check array bounds
- Validate alignment requirements
- Ensure type compatibility
- Verify object invariants

### 3. Meaningful Error Messages
Provide clear, actionable error messages:
- Include specific parameter values
- Describe what went wrong
- Suggest possible solutions

## Code Organization

### A. Safe Wrapper Module Structure
```
crates/metallic/src/mps_graph/
├── safe_wrappers/
│   ├── mod.rs              # Public exports
│   ├── graph.rs            # MPSGraph wrappers
│   ├── tensors.rs          # Tensor wrappers  
│   ├── matrices.rs         # Matrix wrappers
│   ├── commands.rs         # Command buffer wrappers
│   └── utils.rs            # Utility functions
```

### B. Kernel Implementation Guidelines
- Use safe wrappers instead of direct unsafe calls
- Import safe wrapper types
- Handle errors appropriately
- Maintain existing functionality

## Testing and Validation

### 1. Unit Tests for Safe Wrappers
Each safe wrapper should have comprehensive unit tests:
- Test successful operations
- Test error conditions
- Test edge cases
- Test parameter validation

### 2. Integration Tests
Validate that safe wrappers produce the same results as the original unsafe code:
- Compare outputs
- Verify performance characteristics
- Test under various conditions

### 3. Fuzz Testing
For complex operations, implement fuzz testing to identify edge cases that could cause safety issues.

## Migration Checklist

Before making changes to kernel implementations:

- [ ] Identify all `unsafe` calls in the module
- [ ] Verify safe wrappers exist for each unsafe operation
- [ ] Update imports to use safe wrapper types
- [ ] Replace unsafe calls with safe wrapper calls
- [ ] Handle errors appropriately
- [ ] Run existing tests
- [ ] Add new tests if needed
- [ ] Verify performance impact is minimal

## Clean Separation Principle

### Key Indicator of Successful Safety Implementation

A critical success metric for the safety system is the **clean separation** between:
- **Unsafe infrastructure** (type definitions and safe wrappers)
- **Safe kernel implementations** (where developers write kernels)

### Complete Separation Requirement

**ALL `objc2` imports should be restricted to:**
- Type definition files (e.g., `crates/metallic/src/types/mod.rs`)
- Safe wrapper implementation files (e.g., `crates/metallic/src/mps_graph/safe_wrappers/`)
- Core system files that directly interface with Metal APIs

**NO `objc2` imports should exist in:**
- Kernel implementation files (e.g., `crates/metallic/src/kernels/*/mod.rs`)
- Kernel operation files
- User-facing API modules
- Any file where developers write kernel logic

### Verification Method

To verify successful implementation:

1. **Search for `objc2` imports** outside of the designated system files:
   ```bash
   grep -r "use objc2" crates/metallic/src/kernels/
   grep -r "use objc2" crates/metallic/src/context/
   # Should return NO results for kernel implementations
   ```

2. **Confirm clean imports** in kernel files:
   ```rust
   // Kernel files should import from safe wrappers/types only
   use crate::types::{Graph, GraphTensor};  // ✅ Good
   use crate::mps_graph::safe_wrappers::SafeMpsGraph;  // ✅ Good
   use objc2::...;  // ❌ Should not exist in kernel files
   ```

### Benefits of Clean Separation

1. **Developer Safety**: Kernel developers never encounter unsafe code directly
2. **Maintainability**: Unsafe code is centralized and easier to audit
3. **Clarity**: Clear boundaries between system code and kernel logic
4. **Reduced Errors**: Less chance of improper unsafe usage during kernel development

### Migration Verification

After completing the migration:

- **Kernel files** should only import from `crate::types`, `crate::mps_graph::safe_wrappers`, and other safe abstractions
- **All `objc2` types** should be hidden behind safe, well-named wrapper types
- **Unsafe operations** should be completely encapsulated in the wrapper layer
- **Kernel developers** should never need to think about `objc2` or unsafe code

This separation serves as a key architectural principle ensuring that the safety system is truly effective at protecting developers from unsafe code while maintaining all functionality.

## Performance Considerations

### 1. Zero-Cost Abstractions
Safe wrappers should have minimal performance overhead:
- No unnecessary allocations
- Inline simple validation
- Avoid redundant checks when possible
- Maintain existing performance characteristics

### 2. Optimization Hints
- Use `#[inline]` for simple wrapper methods
- Cache frequently used values
- Avoid redundant validation when invariants are guaranteed

## Future Improvements

### 1. Macro-Based Generators
Consider creating macros to reduce boilerplate for similar wrapper types.

### 2. Automatic Migration Tools
Develop tools to automatically replace common unsafe patterns with safe alternatives.

### 3. Static Analysis
Implement static analysis to prevent direct use of unsafe APIs outside of safe wrappers.

### 4. Type Unification and Cleanup

One of the most important improvements is to unify and cleanup the long, unwieldy `objc2` types, especially `Retained` types. The current codebase has many verbose type signatures that hurt readability and make it harder to understand the code.

#### A. Common Type Aliases
Create meaningful type aliases for frequently used `Retained` types:

```rust
// In crates/metallic/src/types/mod.rs
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal_performance_shaders_graph as mpsg;

// Common graph types
pub type Graph = Retained<mpsg::MPSGraph>;
pub type GraphExecutable = Retained<mpsg::MPSGraphExecutable>;
pub type GraphTensor = Retained<mpsg::MPSGraphTensor>;
pub type GraphTensorData = Retained<mpsg::MPSGraphTensorData>;
pub type GraphShapedType = Retained<mpsg::MPSGraphShapedType>;

// Common array types  
pub type TensorDataArray = Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensorData>>;
pub type GraphTensorArray = Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>>;

// Command buffer types
pub type MPSCommandBuffer = Retained<objc2_metal_performance_shaders::MPSCommandBuffer>;
```

#### B. Strongly-Typed Wrappers with Unit Enums
Create unit enums that wrap `Retained` types to provide additional type safety and clearer documentation:

```rust
// In crates/metallic/src/types/wrappers.rs
use crate::types::{GraphExecutable, GraphTensor, GraphTensorData};

/// Wrapper for compiled graph executables with additional metadata
#[derive(Clone)]
pub struct CompiledGraphExecutable(pub GraphExecutable);

impl CompiledGraphExecutable {
    pub fn inner(&self) -> &GraphExecutable {
        &self.0
    }
    
    // Add convenience methods specific to executables
    pub fn encode_to_command_buffer(
        &self,
        command_buffer: &super::MPSCommandBuffer,
        inputs: &[GraphTensorData],
        results: Option<&[GraphTensorData]>,
    ) -> Result<(), MetalError> {
        // Implementation using safe wrapper
        todo!()
    }
}

/// Wrapper for graph input tensors with additional validation
#[derive(Clone)]
pub struct InputGraphTensor {
    pub inner: GraphTensor,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// Wrapper for graph output tensors with additional validation  
#[derive(Clone)]
pub struct OutputGraphTensor {
    pub inner: GraphTensor,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

/// Wrapper for tensor data with buffer information
#[derive(Clone)]
pub struct BoundTensorData {
    pub inner: GraphTensorData,
    pub tensor_shape: Vec<usize>,
    pub buffer_size: usize,
}
```

#### C. Module-Specific Type Groupings
Group related types in module-specific type files:

```rust
// In crates/metallic/src/kernels/sdpa/types.rs
use crate::types::{GraphTensor, GraphShapedType};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaFeedBinding {
    Query,
    Key, 
    Value,
    Mask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpaResultBinding {
    Attention,
}

pub struct SdpaGraphInputs {
    pub query: GraphTensor,
    pub key: GraphTensor, 
    pub value: GraphTensor,
    pub mask: Option<GraphTensor>,
}

pub struct SdpaGraphOutputs {
    pub attention: GraphTensor,
}
```

#### D. Benefits of Type Unification

1. **Improved Readability**:
   - Before: `Retained<ProtocolObject<dyn mpsg::MPSGraphExecutable>>`
   - After: `CompiledGraphExecutable`

2. **Better Type Safety**:
   - Unit enums prevent mixing input/output tensors
   - Additional metadata can be stored with types
   - Method implementations are more focused

3. **Easier Refactoring**:
   - Change underlying type in one place
   - Compiler catches all usages during refactoring
   - Clear documentation of type purpose

4. **Better IDE Experience**:
   - Clearer type names in IDE tooltips
   - Easier to search for type usages
   - Better autocomplete experience

#### E. Migration to Unified Types

When implementing type unification:

1. **Start with common types** that are used across multiple modules
2. **Create gradual migration path** - use both old and new types during transition
3. **Update documentation** to reflect new type names
4. **Provide conversion methods** between old and new types during migration
5. **Gradually replace usage** across the codebase

This type unification approach works excellently with the safe wrapper system, where the wrappers can operate on the cleaner, unified types rather than the verbose `objc2` types directly.

---

## Reference: Unsafe Functions Identified in Current Codebase

### A. MPSGraph Object Creation
- `mpsg::MPSGraph::new()` - Requires unsafe
- `graph.placeholderWithShape_dataType_name()` - Requires unsafe
- `graph.constantWithScalar_shape_dataType()` - Requires unsafe

### B. MPSGraph Type Initialization  
- `MPSGraphShapedType::initWithShape_dataType()` - Requires unsafe
- `MPSGraphTensorData::initWithMTLBuffer_shape_dataType()` - Requires unsafe
- `MPSGraphTensorData::initWithMPSNDArray()` - Requires unsafe

### C. Metal Performance Shaders Matrix Operations
- `MPSMatrixMultiplication::initWithDevice_transposeLeft...` - Requires unsafe
- `MPSMatrix::initWithBuffer_offset_descriptor()` - Requires unsafe
- `MPSMatrixSoftMax::initWithDevice()` - Requires unsafe

### D. Foundation Framework Usage
- `Retained::cast_unchecked()` - Unsafe cast operations
- Various `encodeToCommandBuffer_*` methods - Requires unsafe

### E. Tensor Binding and Shape Operations
- `MPSNDArrayDescriptor::descriptorWithDataType_shape()` - Requires unsafe
- `MPSNDArray::initWithBuffer_offset_descriptor()` - Requires unsafe
- `arrayViewWithShape_strides()` - Requires unsafe

### F. Command Buffer Integration
- `MPSCommandBuffer::commandBufferWithCommandBuffer()` - Requires unsafe
- `encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor()` - Requires unsafe

This comprehensive list serves as a reference for the migration process, ensuring all unsafe operations are properly encapsulated in safe wrappers.