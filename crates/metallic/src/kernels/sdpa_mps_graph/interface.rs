use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSMutableArray, NSNumber};
use objc2_metal_performance_shaders_graph as mpsg;

use crate::{
    MetalError, Tensor, TensorElement, cacheable_resources::{CacheableMpsGraphSdpa, CacheableMpsGraphSdpaMask, MpsGraphSdpaFeedBinding, MpsGraphSdpaOutputBinding}, mps_graph::bindings::{GraphTensorBinding, GraphTensorDataArrayBuilder}
};

pub struct SdpaGraphInterface<'a> {
    cache_entry: &'a CacheableMpsGraphSdpa,
    mask_cache_entry: Option<&'a CacheableMpsGraphSdpaMask>,
    custom_mask_buffer: Option<&'a ProtocolObject<dyn objc2_metal::MTLBuffer>>,
}

impl<'a> SdpaGraphInterface<'a> {
    pub fn new(
        cache_entry: &'a CacheableMpsGraphSdpa,
        mask_cache_entry: Option<&'a CacheableMpsGraphSdpaMask>,
        custom_mask_buffer: Option<&'a ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) -> Self {
        Self {
            cache_entry,
            mask_cache_entry,
            custom_mask_buffer,
        }
    }
}

pub struct SdpaGraphInputs<'a, T: TensorElement> {
    pub query: &'a Tensor<T>,
    pub key: &'a Tensor<T>,
    pub value: &'a Tensor<T>,
}

impl<'a, T: TensorElement> From<(&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>)> for SdpaGraphInputs<'a, T> {
    fn from(value: (&'a Tensor<T>, &'a Tensor<T>, &'a Tensor<T>)) -> Self {
        Self {
            query: value.0,
            key: value.1,
            value: value.2,
        }
    }
}

pub struct SdpaGraphOutput<'a, T: TensorElement> {
    pub attention: &'a Tensor<T>,
}

impl<'a, T: TensorElement> From<&'a Tensor<T>> for SdpaGraphOutput<'a, T> {
    fn from(attention: &'a Tensor<T>) -> Self {
        Self { attention }
    }
}

impl<'a> SdpaGraphInterface<'a> {
    pub fn bind_inputs<T: TensorElement>(
        &self,
        inputs: SdpaGraphInputs<'_, T>,
    ) -> Result<Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensorData>>, MetalError> {
        let builder = GraphTensorDataArrayBuilder::default();

        let query_dims = inputs.query.dims();
        let key_dims = inputs.key.dims();
        let value_dims = inputs.value.dims();

        if query_dims.len() != 3 || key_dims.len() != 3 || value_dims.len() != 3 {
            return Err(MetalError::InvalidShape("SDPA MPSGraph expects 3D tensors [B, S, D]".into()));
        }

        let batch = query_dims[0];
        let seq_q = query_dims[1];
        let dim = query_dims[2];
        let seq_k = key_dims[1];

        // Pre-compute all required shapes once to avoid repeated allocations
        let query_shape = make_qkv_shape(batch, seq_q, dim);
        let key_shape = make_qkv_shape(batch, seq_k, dim);
        let value_shape = make_qkv_shape(batch, seq_k, dim);
        let mask_shape = make_mask_shape(seq_q, seq_k);

        for binding in &self.cache_entry.feed_layout {
            match binding {
                MpsGraphSdpaFeedBinding::Query => {
                    let descriptor = GraphTensorBinding::from_tensor(inputs.query, &query_shape, self.cache_entry.data_type)?;
                    builder.push(&descriptor)?;
                }
                MpsGraphSdpaFeedBinding::Key => {
                    let descriptor = GraphTensorBinding::from_tensor(inputs.key, &key_shape, self.cache_entry.data_type)?;
                    builder.push(&descriptor)?;
                }
                MpsGraphSdpaFeedBinding::Value => {
                    let descriptor = GraphTensorBinding::from_tensor(inputs.value, &value_shape, self.cache_entry.data_type)?;
                    builder.push(&descriptor)?;
                }
                MpsGraphSdpaFeedBinding::Mask => {
                    let descriptor = if let Some(custom_buffer) = self.custom_mask_buffer {
                        GraphTensorBinding::from_buffer(custom_buffer, 0, &mask_shape, self.cache_entry.data_type)?
                    } else {
                        let mask = self
                            .mask_cache_entry
                            .as_ref()
                            .ok_or_else(|| MetalError::OperationFailed("SDPA mask buffer missing from cache entry".into()))?;
                        if seq_q > mask.seq_q_size || seq_k > mask.seq_k_size {
                            return Err(MetalError::OperationFailed(format!(
                                "Mask cache entry too small for requested shape ({seq_q}x{seq_k})",
                            )));
                        }
                        GraphTensorBinding::from_buffer(&mask.buffer, 0, &mask_shape, self.cache_entry.data_type)?
                    };
                    builder.push(&descriptor)?;
                }
            }
        }

        Ok(builder.finish())
    }

    pub fn bind_outputs<T: TensorElement>(
        &self,
        output: SdpaGraphOutput<'_, T>,
    ) -> Result<Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensorData>>, MetalError> {
        if self.cache_entry.result_layout.len() != 1 || self.cache_entry.result_layout[0] != MpsGraphSdpaOutputBinding::Attention {
            return Err(MetalError::OperationFailed(
                "Unexpected SDPA result layout in cached MPSGraph executable".into(),
            ));
        }

        let builder = GraphTensorDataArrayBuilder::default();
        let mut _shapes: Vec<Retained<NSArray<NSNumber>>> = Vec::new();

        let dims = output.attention.dims();
        if dims.len() != 3 {
            return Err(MetalError::InvalidShape("SDPA output tensor must be 3D [B, S, D]".into()));
        }

        let shape = make_qkv_shape(dims[0], dims[1], dims[2]);
        let descriptor = GraphTensorBinding::from_tensor(output.attention, &shape, self.cache_entry.data_type)?;
        builder.push(&descriptor)?;
        _shapes.push(shape);

        Ok(builder.finish())
    }
}

fn make_shape(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let arr: Retained<NSMutableArray<NSNumber>> = NSMutableArray::array();
    for &dim in dims {
        let value = NSNumber::numberWithUnsignedInteger(dim);
        arr.addObject(&*value);
    }
    unsafe { Retained::cast_unchecked(arr) }
}

fn make_qkv_shape(batch: usize, seq: usize, dim: usize) -> Retained<NSArray<NSNumber>> {
    make_shape(&[batch, 1, seq, dim])
}

fn make_mask_shape(seq_q: usize, seq_k: usize) -> Retained<NSArray<NSNumber>> {
    make_shape(&[1, 1, seq_q, seq_k])
}
