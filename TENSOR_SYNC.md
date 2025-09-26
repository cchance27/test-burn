# Proposal: Implicit Synchronization via Stateful Tensors

## 1. Executive Summary

The current system requires developers to manually call `context.synchronize()` to ensure GPU computations are complete before results are used on the CPU. This is error-prone, leading to bugs from stale data, and often results in suboptimal performance due to excessive, unnecessary synchronization calls.

This document proposes a shift to an **implicit synchronization model** by making our `Tensor` struct "stateful." Each tensor will track the command buffer responsible for its creation or last modification. Synchronization will then be handled automatically and lazily, triggered only when absolutely necessary.

### Key Benefits:
-   **Improve Performance:** By batching multiple operations into a single command buffer and only synchronizing once at the end, we minimize CPU-GPU stalls.
-   **Enhance Developer Experience:** Remove the mental overhead of tracking synchronization. The API becomes more intuitive and less verbose.
-   **Reduce Errors:** Make it impossible to accidentally read stale or incomplete data from the GPU.
-   **Improve Resource Management:** Enable better reuse of memory from the `MemoryPool` for intermediate tensors within a single, uncommitted command buffer.

## 2. Proposed Technical Solution

### Step 1: Make `Tensor` Stateful

We will add a field to the `Tensor` struct to hold a reference to the command buffer that will produce its data. This field makes the tensor "aware" of its own readiness.

**File:** `src/metallic/tensor.rs`
```rust
// We will need a shared, clonable CommandBuffer wrapper.
// Let's assume it's defined in `src/metallic/operation.rs` or similar.
// pub struct CommandBuffer(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

#[derive(Clone)]
pub struct Tensor {
    pub buf: RetainedBuffer,
    pub dims: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: Dtype,
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub offset: usize,
    
    // --- NEW FIELD ---
    /// The command buffer that must complete before this tensor's data is valid.
    /// If `None`, the data is already synchronized and ready for CPU access.
    pub(crate) defining_cmd_buffer: Option<CommandBuffer>,
}
```

### Step 2: Update `Context` to Manage State

The `Context` will manage a single "active" command buffer. New operations are encoded onto it.

**File:** `src/metallic/context.rs` (or equivalent)
```rust
pub struct Context {
    // ... existing fields: device, command_queue, kernel_manager, pool ...
    
    // --- NEW FIELD ---
    /// The current command buffer for encoding new operations.
    active_cmd_buffer: Option<CommandBuffer>,
}

impl Context {
    // `call` will be updated to use and manage the active buffer
    pub fn call<K: KernelInvocable>(/*...*/) -> Result<K::Output, MetalError> {
        // ... (detailed logic in TODO list) ...
    }

    // `synchronize` will now flush the active buffer
    pub fn synchronize(&mut self) {
        if let Some(cmd_buf) = self.active_cmd_buffer.take() {
            cmd_buf.commit();
            cmd_buf.wait();
        }
    }
}
```

### Step 3: Implement "Pull" Synchronization on CPU Access

The primary trigger for synchronization will be any attempt to read or write tensor data from the CPU.

**File:** `src/metallic/tensor.rs`
```rust
impl Tensor {
    /// Immutable host view of the buffer. This will block and wait for the GPU
    /// if the tensor's data is not yet synchronized.
    pub fn as_slice(&mut self) -> &[f32] { // Note: now takes &mut self
        if let Some(cmd_buf) = self.defining_cmd_buffer.take() {
            // Commit the buffer if it hasn't been already (idempotent)
            cmd_buf.commit(); 
            // Wait for it to complete.
            cmd_buf.wait();
        }
        
        // Now the data is guaranteed to be ready.
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.len()) }
    }

    /// Mutable host view of the buffer. Also synchronizes.
    pub fn as_mut_slice(&mut self) -> &mut [f32] { // Note: takes &mut self
        if let Some(cmd_buf) = self.defining_cmd_buffer.take() {
            cmd_buf.commit();
            cmd_buf.wait();
        }
        
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) }
    }

    pub fn to_vec(&mut self) -> Vec<f32> {
        self.as_slice().to_vec()
    }
}
```

### Step 4: Refactor High-Level Tensor Ops

All `ctx.synchronize()` calls must be removed from the element-wise and creation methods in `tensor.rs`.

**File:** `src/metallic/tensor.rs`
```rust
// BEFORE
pub fn add_elem(&self, other: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
    // ... checks ...
    let out = ctx.call::<ElemwiseAddOp>((self.clone(), other.clone()))?;
    ctx.synchronize(); // <-- REMOVE THIS
    Ok(out)
}

// AFTER
pub fn add_elem(&self, other: &Tensor, ctx: &mut Context) -> Result<Tensor, MetalError> {
    // ... checks ...
    // The call now returns a "pending" tensor. No sync.
    ctx.call::<ElemwiseAddOp>((self.clone(), other.clone()))
}
```

## 3. Example: Before vs. After

**Before (Current System):**
```rust
// Three separate dispatches, three blocking waits. Very slow.
let x = a.add_elem(&b, &mut ctx)?;      // Dispatches, waits.
let y = x.mul_elem(&c, &mut ctx)?;      // Dispatches, waits.
let mut z = y.add_scalar(1.0, &mut ctx)?;   // Dispatches, waits.

// Now we can read z
let result = z.to_vec(); 
```

**After (Proposed System):**
```rust
// These calls just encode kernels onto the same command buffer.
// They return instantly without waiting for the GPU.
let x = a.add_elem(&b, &mut ctx)?;
let y = x.mul_elem(&c, &mut ctx)?;
let mut z = y.add_scalar(1.0, &mut ctx)?; // z is now a "pending" tensor

// The moment of truth: we ask for the data.
// The `to_vec` call on `z` triggers the commit and wait for the whole chain.
let result = z.to_vec(); 
```

## 4. Implementation TODO List

- [ ] **1. Define `CommandBuffer` Wrapper**
    -   **File:** `src/metallic/operation.rs` (or a new `src/metallic/command_buffer.rs`)
    -   **Action:** Create a struct `CommandBuffer(pub Retained<ProtocolObject<dyn MTLCommandBuffer>>)` that is `Clone`, and implements `commit()` and `wait()` methods. Ensure these methods handle being called multiple times safely (e.g., use a flag or `Once`).

- [ ] **2. Update `Tensor` Struct**
    -   **File:** `src/metallic/tensor.rs`
    -   **Action:** Add the new field: `pub(crate) defining_cmd_buffer: Option<CommandBuffer>`.
    -   **Action:** Update `Tensor` creation functions (`create_tensor_from_slice`, etc.) to initialize the new field to `None` as they are CPU-synchronized.

- [ ] **3. Update `Context` Struct**
    -   **File:** `src/metallic/context.rs`
    -   **Action:** Add the new field: `active_cmd_buffer: Option<CommandBuffer>`.
    -   **Action:** Implement the `synchronize` method as described above.

- [ ] **4. Implement Core Logic in `Context::call`**
    -   **File:** `src/metallic/context.rs`
    -   **Action:**
        1.  Check the `defining_cmd_buffer` of all input `Tensor` arguments.
        2.  If any input tensor has a dependency, `commit()` and `wait()` on its command buffer. Clear the tensor's dependency field to `None`.
        3.  Ensure `self.active_cmd_buffer` exists, creating one if it's `None`.
        4.  Encode the new operation onto the `active_cmd_buffer`.
        5.  For the output `Tensor`(s), `clone()` the `active_cmd_buffer` and set it as their `defining_cmd_buffer`.
        6.  Return the "pending" output tensor(s).

- [ ] **5. Implement Synchronization Triggers**
    -   **File:** `src/metallic/tensor.rs`
    -   **Action:** Modify `as_slice`, `as_mut_slice`, and `to_vec` to take `&mut self`.
    -   **Action:** Implement the logic to check `defining_cmd_buffer`, `take()` it, `commit()` and `wait()` on the buffer, and then return the slice.

- [ ] **6. Refactor High-Level Tensor Methods**
    -   **File:** `src/metallic/tensor.rs`
    -   **Action:** Go through every method that calls `ctx.synchronize()` (e.g., `add_elem`, `mul_elem`, `ones`, `zeros`, `random_uniform`, `permute`) and remove the call.

- [ ] **7. Review and Update Tests**
    -   **Files:** All `*_test.rs` files.
    -   **Action:** Tests that use `as_slice()` will now need to use a mutable reference to the tensor (e.g., `let mut result_tensor = ...`). Update all tests to reflect the new API and ensure they still pass. Remove explicit `ctx.synchronize()` calls where they are no longer needed.
