#![allow(dead_code)]
use super::{CommandBuffer, Context, MetalError};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;
use std::ops::{Add, Div, Mul, Sub};

pub type RetainedBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;

/// A small zero-copy Tensor backed by a retained Metal buffer.
/// The buffer is retained while this struct is alive. `as_slice()` provides
/// an immutable view into the underlying f32 contents; callers must ensure
/// GPU work has completed (e.g. via command buffer wait) before reading.
#[derive(Clone)]
pub struct Tensor {
    pub buf: RetainedBuffer,
    /// Shape of the tensor in elements (e.g. [batch, seq_q, dim])
    pub dims: Vec<usize>,
    /// The Metal device used to create this tensor's buffer.
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Byte offset into the buffer.
    pub offset: usize,
}

impl Tensor {
    #[inline]
    pub fn size_bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * std::mem::size_of::<f32>()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.dims.iter().product::<usize>()
    }

    /// Create a tensor by copying from a host slice.
    pub fn create_tensor_from_slice(
        items: &[f32],
        dims: Vec<usize>,
        context: &Context,
    ) -> Result<Tensor, MetalError> {
        let num_elements = items.len();
        let byte_len = std::mem::size_of_val(items);
        let item_ptr =
            std::ptr::NonNull::new(items.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;

        let buf = unsafe {
            context
                .device
                .newBufferWithBytes_length_options(
                    item_ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or(MetalError::BufferFromBytesCreationFailed)?
        };

        let expected_elements = dims.iter().product::<usize>();
        if expected_elements != num_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: num_elements,
            });
        }

        Ok(Tensor {
            buf,
            dims,
            device: context.device.clone(),
            offset: 0,
        })
    }

    /// Create an uninitialized tensor of given shape. Contents are unspecified.
    pub fn create_tensor(
        num_elements: usize,
        dims: Vec<usize>,
        context: &Context,
    ) -> Result<Tensor, MetalError> {
        let byte_len = num_elements * std::mem::size_of::<f32>();
        let buf = context
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;

        let expected_elements = dims.iter().product::<usize>();
        if expected_elements != num_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: num_elements,
            });
        }
        Ok(Tensor {
            buf,
            dims,
            device: context.device.clone(),
            offset: 0,
        })
    }

    /// Create a tensor view from an existing Metal buffer without copying.
    pub fn from_existing_buffer(
        buffer: RetainedBuffer,
        dims: Vec<usize>,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        offset: usize,
    ) -> Result<Tensor, MetalError> {
        let expected_bytes = dims.iter().product::<usize>() * std::mem::size_of::<f32>();
        if offset + expected_bytes > buffer.length() {
            return Err(MetalError::InvalidShape(
                "buffer too small for dims/offset".into(),
            ));
        }
        Ok(Tensor {
            buf: buffer,
            dims,
            device: device.clone(),
            offset,
        })
    }

    /// Immutable host view of the buffer. Ensure GPU work has completed before reading.
    pub fn as_slice(&self) -> &[f32] {
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.len()) }
    }

    /// Copy the tensor contents to a host Vec.
    pub fn to_vec(&self) -> Vec<f32> {
        self.as_slice().to_vec()
    }

    /// Synchronize given command buffers before host read. Convenience to make the read contract explicit.
    pub fn sync_before_read(buffers: &[CommandBuffer]) {
        for cb in buffers {
            cb.wait();
        }
    }

    /// Mutable host view of the buffer. Ensure no concurrent GPU access.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        let ptr = unsafe { self.buf.contents().as_ptr().add(self.offset) } as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            buf: self.buf.clone(),
            dims: vec![self.len()],
            device: self.device.clone(),
            offset: self.offset,
        }
    }

    pub fn reshape(&self, new_dims: Vec<usize>) -> Result<Tensor, MetalError> {
        let expected_elements: usize = new_dims.iter().product();
        let actual_elements = self.len();
        if expected_elements != actual_elements {
            return Err(MetalError::DimensionMismatch {
                expected: expected_elements,
                actual: actual_elements,
            });
        }
        Ok(Tensor {
            buf: self.buf.clone(),
            dims: new_dims,
            device: self.device.clone(),
            offset: self.offset,
        })
    }

    /// Allocate and zero-initialize a tensor of the given shape.
    pub fn zeros(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let num_elements = dims.iter().product::<usize>();
        let byte_len = num_elements * std::mem::size_of::<f32>();
        let buf = context
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;

        let ptr = buf.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::write_bytes(ptr, 0, num_elements);
        }

        Ok(Tensor {
            buf,
            dims,
            device: context.device.clone(),
            offset: 0,
        })
    }

    /// Allocate and fill a tensor with ones.
    pub fn ones(dims: Vec<usize>, context: &Context) -> Result<Tensor, MetalError> {
        let mut t = Self::zeros(dims, context)?;
        for x in t.as_mut_slice().iter_mut() {
            *x = 1.0;
        }
        Ok(t)
    }

    /// Allocate and fill a tensor from a Vec (host).
    pub fn from_vec(
        values: Vec<f32>,
        dims: Vec<usize>,
        context: &Context,
    ) -> Result<Tensor, MetalError> {
        let expected = dims.iter().product::<usize>();
        if values.len() != expected {
            return Err(MetalError::DimensionMismatch {
                expected,
                actual: values.len(),
            });
        }
        let mut t = Self::create_tensor(values.len(), dims, context)?;
        t.as_mut_slice().copy_from_slice(&values);
        Ok(t)
    }

    /// Create an arange tensor (0..n as f32) with the given shape.
    pub fn arange(
        num_elements: usize,
        dims: Vec<usize>,
        context: &Context,
    ) -> Result<Tensor, MetalError> {
        let v: Vec<f32> = (0..num_elements).map(|x| x as f32).collect();
        Self::from_vec(v, dims, context)
    }

    /// Create a zeros tensor with the same shape.
    pub fn zeros_like(&self, context: &Context) -> Result<Tensor, MetalError> {
        Self::zeros(self.dims.clone(), context)
    }

    /// Create a ones tensor with the same shape.
    pub fn ones_like(&self, context: &Context) -> Result<Tensor, MetalError> {
        Self::ones(self.dims.clone(), context)
    }

    /// Fill the tensor in-place with a scalar value.
    pub fn fill(&mut self, value: f32) {
        let slice = self.as_mut_slice();
        for x in slice.iter_mut() {
            *x = value;
        }
    }

    /// Element-wise add, returns a new tensor on the same device.
    pub fn add_elem(&self, other: &Tensor) -> Result<Tensor, MetalError> {
        Self::binary_elementwise(self, other, |a, b| a + b)
    }

    /// Element-wise sub, returns a new tensor on the same device.
    pub fn sub_elem(&self, other: &Tensor) -> Result<Tensor, MetalError> {
        Self::binary_elementwise(self, other, |a, b| a - b)
    }

    /// Element-wise mul, returns a new tensor on the same device.
    pub fn mul_elem(&self, other: &Tensor) -> Result<Tensor, MetalError> {
        Self::binary_elementwise(self, other, |a, b| a * b)
    }

    /// Element-wise div, returns a new tensor on the same device.
    pub fn div_elem(&self, other: &Tensor) -> Result<Tensor, MetalError> {
        Self::binary_elementwise(self, other, |a, b| a / b)
    }

    /// Element-wise scalar add.
    pub fn add_scalar(&self, value: f32) -> Result<Tensor, MetalError> {
        Self::unary_elementwise(self, |a| a + value)
    }

    /// Element-wise scalar mul.
    pub fn mul_scalar(&self, value: f32) -> Result<Tensor, MetalError> {
        Self::unary_elementwise(self, |a| a * value)
    }

    fn binary_elementwise<F: Fn(f32, f32) -> f32>(
        a: &Tensor,
        b: &Tensor,
        f: F,
    ) -> Result<Tensor, MetalError> {
        if a.dims != b.dims {
            return Err(MetalError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }
        let byte_len = a.size_bytes();
        let buf = a
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;
        let mut out = Tensor {
            buf,
            dims: a.dims.clone(),
            device: a.device.clone(),
            offset: 0,
        };
        let aslice = a.as_slice();
        let bslice = b.as_slice();
        let oslice = out.as_mut_slice();
        for i in 0..a.len() {
            oslice[i] = f(aslice[i], bslice[i]);
        }
        Ok(out)
    }

    fn unary_elementwise<F: Fn(f32) -> f32>(a: &Tensor, f: F) -> Result<Tensor, MetalError> {
        let byte_len = a.size_bytes();
        let buf = a
            .device
            .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferCreationFailed(byte_len))?;
        let mut out = Tensor {
            buf,
            dims: a.dims.clone(),
            device: a.device.clone(),
            offset: 0,
        };
        let aslice = a.as_slice();
        let oslice = out.as_mut_slice();
        for i in 0..a.len() {
            oslice[i] = f(aslice[i]);
        }
        Ok(out)
    }

    pub fn get_batch(&self, batch_index: usize) -> Result<Tensor, MetalError> {
        if self.dims.len() < 3 {
            return Err(MetalError::InvalidShape(
                "get_batch requires at least 3 dimensions".to_string(),
            ));
        }
        if batch_index >= self.dims[0] {
            return Err(MetalError::InvalidShape(
                "batch_index out of bounds".to_string(),
            ));
        }

        let batch_size_bytes =
            self.dims[1..].iter().product::<usize>() * std::mem::size_of::<f32>();
        let new_offset = self.offset + batch_index * batch_size_bytes;

        Ok(Tensor {
            buf: self.buf.clone(),
            dims: self.dims[1..].to_vec(),
            device: self.device.clone(),
            offset: new_offset,
        })
    }
}

// Operator overloading for convenience. These panic on dimension mismatch.
impl<'b> Add<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &'b Tensor) -> Tensor {
        self.add_elem(rhs)
            .expect("Tensor Add: dimension mismatch or allocation failure")
    }
}
impl<'b> Sub<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &'b Tensor) -> Tensor {
        self.sub_elem(rhs)
            .expect("Tensor Sub: dimension mismatch or allocation failure")
    }
}
impl<'b> Mul<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Tensor {
        self.mul_elem(rhs)
            .expect("Tensor Mul: dimension mismatch or allocation failure")
    }
}
impl<'b> Div<&'b Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &'b Tensor) -> Tensor {
        self.div_elem(rhs)
            .expect("Tensor Div: dimension mismatch or allocation failure")
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::metallic::Context;

    #[test]
    fn zeros_and_ones() {
        let ctx = Context::new().unwrap();
        let t0 = Tensor::zeros(vec![2, 3, 4], &ctx).unwrap();
        assert!(t0.as_slice().iter().all(|&x| x == 0.0));
        let t1 = Tensor::ones(vec![2, 3, 4], &ctx).unwrap();
        assert!(t1.as_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn zeros_like_and_ones_like() {
        let ctx = Context::new().unwrap();
        let base =
            Tensor::create_tensor_from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], &ctx).unwrap();
        let z = base.zeros_like(&ctx).unwrap();
        assert_eq!(z.dims(), base.dims());
        assert!(z.as_slice().iter().all(|&x| x == 0.0));
        let o = base.ones_like(&ctx).unwrap();
        assert_eq!(o.dims(), base.dims());
        assert!(o.as_slice().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn elementwise_ops_and_fill() {
        let ctx = Context::new().unwrap();
        let a = Tensor::create_tensor_from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], &ctx).unwrap();
        let b = Tensor::create_tensor_from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2], &ctx).unwrap();
        let c = &a + &b;
        assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
        let d = &a * &b;
        assert_eq!(d.as_slice(), &[5.0, 12.0, 21.0, 32.0]);
        let mut e = a.add_scalar(10.0).unwrap();
        assert_eq!(e.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
        e.fill(2.5);
        assert!(e.as_slice().iter().all(|&x| (x - 2.5).abs() < 1e-12));
    }

    #[test]
    fn get_batch_and_from_existing_buffer() {
        let ctx = Context::new().unwrap();
        // base tensor: shape [2,3,4], values 0..24
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let mut base = Tensor::create_tensor(24, vec![2, 3, 4], &ctx).unwrap();
        base.as_mut_slice().copy_from_slice(&data);

        // get second batch
        let b1 = base.get_batch(1).unwrap();
        assert_eq!(b1.dims(), &[3, 4]);
        // expected slice
        let expected: Vec<f32> = data[12..].to_vec();
        assert_eq!(b1.as_slice(), expected.as_slice());

        // wrap the second batch region using from_existing_buffer
        let offset_bytes = 12 * std::mem::size_of::<f32>();
        let view =
            Tensor::from_existing_buffer(base.buf.clone(), vec![3, 4], &base.device, offset_bytes)
                .unwrap();
        assert_eq!(view.as_slice(), expected.as_slice());
    }

    #[test]
    fn arange_helper() {
        let ctx = Context::new().unwrap();
        let t = Tensor::arange(6, vec![2, 3], &ctx).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.as_slice(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn from_vec_helper() {
        let ctx = Context::new().unwrap();
        let v = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        let t = Tensor::from_vec(v.clone(), vec![2, 3], &ctx).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.to_vec(), v);
    }
}
