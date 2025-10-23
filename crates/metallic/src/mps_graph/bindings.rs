use std::ptr::NonNull;

use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSMutableArray, NSNumber, NSUInteger};
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders::{MPSDataType, MPSNDArray, MPSNDArrayDescriptor, MPSShape};
use objc2_metal_performance_shaders_graph as mpsg;

use crate::{MetalError, Tensor, TensorElement};

/// Describes a zero-copy binding of an MTLBuffer slice to an MPSGraph placeholder.
/// Invariants:
/// - The bound Tensor/Buffer must match the executable's expected shape and data type.
/// - The buffer must be large enough to cover all elements at the given data type size.
pub struct GraphTensorBinding<'a> {
    buffer: &'a ProtocolObject<dyn MTLBuffer>,
    offset_bytes: usize,
    shape: &'a objc2_foundation::NSArray<NSNumber>,
    data_type: MPSDataType,
    layout: GraphTensorLayout,
}

enum GraphTensorLayout {
    Contiguous,
    Strided { dims: Vec<usize>, strides: Vec<usize> },
}

impl<'a> GraphTensorBinding<'a> {
    pub fn from_tensor<T: TensorElement>(
        tensor: &'a Tensor<T>,
        expected_shape: &'a objc2_foundation::NSArray<NSNumber>,
        data_type: MPSDataType,
    ) -> Result<Self, MetalError> {
        let tensor_dtype = crate::cacheable_resources::mps_data_type_for_dtype(tensor.dtype);
        if tensor_dtype != data_type {
            return Err(MetalError::OperationFailed(format!(
                "Tensor dtype ({tensor_dtype:?}) does not match MPSGraph executable dtype ({data_type:?})"
            )));
        }

        let element_size = bytes_per_element(data_type)?;
        if !tensor.offset.is_multiple_of(element_size) {
            return Err(MetalError::OperationFailed(
                "Tensor offset not aligned to element size for MPSGraph binding".into(),
            ));
        }

        let expected_elements = shape_element_count(expected_shape)?;
        if expected_elements != tensor.len() {
            return Err(MetalError::OperationFailed(
                "Tensor shape does not match cached MPSGraph placeholder dimensions".into(),
            ));
        }
        let layout = if tensor.strides == Tensor::<T>::compute_strides(tensor.dims()) {
            GraphTensorLayout::Contiguous
        } else {
            GraphTensorLayout::Strided {
                dims: tensor.dims.clone(),
                strides: tensor.strides.clone(),
            }
        };

        let available = tensor.buf.length();
        let required = required_span_bytes(tensor.offset, tensor.dims(), &tensor.strides, element_size)?;
        if required > available {
            return Err(MetalError::OperationFailed(
                "MTLBuffer too small for requested MPSGraph binding".into(),
            ));
        }

        Ok(Self {
            buffer: &*tensor.buf,
            offset_bytes: tensor.offset,
            shape: expected_shape,
            data_type,
            layout,
        })
    }

    pub fn from_buffer(
        buffer: &'a ProtocolObject<dyn MTLBuffer>,
        offset_bytes: usize,
        expected_shape: &'a objc2_foundation::NSArray<NSNumber>,
        data_type: MPSDataType,
    ) -> Result<Self, MetalError> {
        let element_size = bytes_per_element(data_type)?;
        if !offset_bytes.is_multiple_of(element_size) {
            return Err(MetalError::OperationFailed(
                "Buffer offset not aligned to element size for MPSGraph binding".into(),
            ));
        }

        let elements = shape_element_count(expected_shape)?;
        let available = buffer.length();
        let required = offset_bytes
            .checked_add(elements * element_size)
            .ok_or_else(|| MetalError::OperationFailed("MPSGraph binding byte size overflow".into()))?;
        if required > available {
            return Err(MetalError::OperationFailed(
                "MTLBuffer too small for requested MPSGraph binding".into(),
            ));
        }

        Ok(Self {
            buffer,
            offset_bytes,
            shape: expected_shape,
            data_type,
            layout: GraphTensorLayout::Contiguous,
        })
    }

    pub fn into_tensor_data(&self) -> Result<Retained<mpsg::MPSGraphTensorData>, MetalError> {
        let element_size = bytes_per_element(self.data_type)?;
        match &self.layout {
            GraphTensorLayout::Contiguous if self.offset_bytes == 0 => {
                // SAFETY: MPSGraphTensorData::initWithMTLBuffer requires that the buffer and shape
                // are valid for the given data type and that the buffer offset is zero. These
                // invariants are enforced by from_tensor/from_buffer.
                unsafe {
                    Ok(mpsg::MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                        mpsg::MPSGraphTensorData::alloc(),
                        self.buffer,
                        self.shape,
                        self.data_type,
                    ))
                }
            }
            GraphTensorLayout::Contiguous => {
                let offset = NSUInteger::try_from(self.offset_bytes)
                    .map_err(|_| MetalError::OperationFailed("Tensor offset exceeds NSUInteger when binding to MPSGraph".into()))?;

                // SAFETY: descriptorWithDataType_shape produces a packed descriptor compatible with the tensor shape.
                let descriptor = unsafe { MPSNDArrayDescriptor::descriptorWithDataType_shape(self.data_type, self.shape) };

                // SAFETY: initWithBuffer_offset_descriptor aliases the underlying Metal buffer at the requested offset.
                let ndarray =
                    unsafe { MPSNDArray::initWithBuffer_offset_descriptor(MPSNDArray::alloc(), self.buffer, offset, &descriptor) };

                // SAFETY: initWithMPSNDArray binds the NDArray view without additional copies.
                let tensor_data = unsafe { mpsg::MPSGraphTensorData::initWithMPSNDArray(mpsg::MPSGraphTensorData::alloc(), &ndarray) };
                Ok(tensor_data)
            }
            GraphTensorLayout::Strided { dims, strides } => {
                let available_bytes = self.buffer.length();
                let available_after_offset = available_bytes
                    .checked_sub(self.offset_bytes)
                    .ok_or_else(|| MetalError::OperationFailed("Tensor offset exceeds buffer length".into()))?;
                let elements_after_offset = available_after_offset / element_size;
                if elements_after_offset == 0 {
                    return Err(MetalError::OperationFailed("Tensor view exceeds available buffer range".into()));
                }

                let mut base_dims = vec![
                    NSUInteger::try_from(elements_after_offset)
                        .map_err(|_| MetalError::OperationFailed("Buffer length exceeds NSUInteger".into()))?,
                ];
                let base_ptr = NonNull::new(base_dims.as_mut_ptr())
                    .ok_or_else(|| MetalError::OperationFailed("Failed to build MPSNDArray base dimensions".into()))?;

                let base_descriptor =
                    unsafe { MPSNDArrayDescriptor::descriptorWithDataType_dimensionCount_dimensionSizes(self.data_type, 1, base_ptr) };

                let offset = NSUInteger::try_from(self.offset_bytes)
                    .map_err(|_| MetalError::OperationFailed("Tensor offset exceeds NSUInteger when binding to MPSGraph".into()))?;

                let base_array =
                    unsafe { MPSNDArray::initWithBuffer_offset_descriptor(MPSNDArray::alloc(), self.buffer, offset, &base_descriptor) };

                let shape_array = make_shape_array(dims)?;
                let stride_array = make_shape_array(strides)?;

                let view = unsafe {
                    base_array
                        .arrayViewWithShape_strides(Some(&shape_array), &stride_array)
                        .ok_or_else(|| MetalError::OperationFailed("Failed to create strided MPSNDArray view for graph binding".into()))?
                };

                let tensor_data = unsafe { mpsg::MPSGraphTensorData::initWithMPSNDArray(mpsg::MPSGraphTensorData::alloc(), &view) };
                Ok(tensor_data)
            }
        }
    }
}

/// A typed spec describing the expected shape and data type for a binding.
/// Use this to ergonomically construct GraphTensorBinding without repeating
/// shape/data_type at each call site.
pub struct GraphBindingSpec<'a> {
    pub expected_shape: &'a objc2_foundation::NSArray<NSNumber>,
    pub data_type: MPSDataType,
}

impl<'a> GraphBindingSpec<'a> {
    pub fn try_from_tensor<T: TensorElement>(&self, tensor: &'a Tensor<T>) -> Result<GraphTensorBinding<'a>, MetalError> {
        GraphTensorBinding::from_tensor(tensor, self.expected_shape, self.data_type)
    }

    pub fn try_from_buffer(
        &self,
        buffer: &'a ProtocolObject<dyn MTLBuffer>,
        offset_bytes: usize,
    ) -> Result<GraphTensorBinding<'a>, MetalError> {
        GraphTensorBinding::from_buffer(buffer, offset_bytes, self.expected_shape, self.data_type)
    }
}

pub struct GraphTensorDataArrayBuilder {
    array: Retained<NSMutableArray<mpsg::MPSGraphTensorData>>,
}

impl Default for GraphTensorDataArrayBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphTensorDataArrayBuilder {
    pub fn new() -> Self {
        Self {
            array: NSMutableArray::array(),
        }
    }

    pub fn push(&self, binding: &GraphTensorBinding<'_>) -> Result<(), MetalError> {
        let tensor_data = binding.into_tensor_data()?;
        self.array.addObject(&*tensor_data);
        Ok(())
    }

    pub fn finish(self) -> Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensorData>> {
        unsafe { Retained::cast_unchecked(self.array) }
    }
}

fn bytes_per_element(data_type: MPSDataType) -> Result<usize, MetalError> {
    match data_type {
        MPSDataType::Float16 => Ok(core::mem::size_of::<half::f16>()),
        MPSDataType::Float32 => Ok(core::mem::size_of::<f32>()),
        other => Err(MetalError::OperationFailed(format!(
            "Unsupported MPSDataType for tensor binding: {other:?}"
        ))),
    }
}

fn shape_element_count(shape: &objc2_foundation::NSArray<NSNumber>) -> Result<usize, MetalError> {
    let mut total = 1usize;
    for idx in 0..shape.count() {
        let value = shape.objectAtIndex(idx).unsignedIntegerValue();
        total = total
            .checked_mul(value)
            .ok_or_else(|| MetalError::OperationFailed("Shape element count overflow".into()))?;
    }
    Ok(total)
}

fn make_shape_array(values: &[usize]) -> Result<Retained<MPSShape>, MetalError> {
    let arr: Retained<NSMutableArray<NSNumber>> = NSMutableArray::array();
    for &value in values {
        let uint = NSUInteger::try_from(value)
            .map_err(|_| MetalError::OperationFailed("Dimension exceeds NSUInteger when constructing MPSShape".into()))?;
        let number = NSNumber::numberWithUnsignedInteger(uint);
        arr.addObject(&*number);
    }
    // SAFETY: NSMutableArray and NSArray share representations.
    Ok(unsafe { Retained::cast_unchecked(arr) })
}

fn required_span_bytes(offset_bytes: usize, dims: &[usize], strides: &[usize], element_size: usize) -> Result<usize, MetalError> {
    if dims.len() != strides.len() {
        return Err(MetalError::OperationFailed(
            "Tensor dims/strides mismatch for MPSGraph binding".into(),
        ));
    }
    let mut last_index = 0usize;
    for (&dim, &stride) in dims.iter().zip(strides.iter()) {
        if dim == 0 {
            return Err(MetalError::InvalidShape(
                "Tensor dimensions must be non-zero when binding to MPSGraph".into(),
            ));
        }
        let term = (dim - 1)
            .checked_mul(stride)
            .ok_or_else(|| MetalError::OperationFailed("Stride computation overflow".into()))?;
        last_index = last_index
            .checked_add(term)
            .ok_or_else(|| MetalError::OperationFailed("Tensor span computation overflow".into()))?;
    }
    let span_elements = last_index
        .checked_add(1)
        .ok_or_else(|| MetalError::OperationFailed("Tensor span computation overflow".into()))?;
    let span_bytes = span_elements
        .checked_mul(element_size)
        .ok_or_else(|| MetalError::OperationFailed("Tensor byte span overflow".into()))?;
    offset_bytes
        .checked_add(span_bytes)
        .ok_or_else(|| MetalError::OperationFailed("Tensor buffer requirement overflow".into()))
}
