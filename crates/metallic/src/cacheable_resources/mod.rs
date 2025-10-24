use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSObjectProtocol;
use objc2_metal::{MTLBuffer, MTLDevice};
use objc2_metal_performance_shaders::{MPSMatrixDescriptor, MPSMatrixMultiplication, MPSMatrixSoftMax};
use objc2_metal_performance_shaders_graph as mpsg;
use rustc_hash::FxHashMap;

use crate::{
    cache_keys::{
        MaskSizeBucket, MpsGemmKey, MpsGraphFusedKey, MpsGraphSdpaKey, MpsGraphSdpaMaskKey, MpsMatrixDescriptorKey, MpsSoftMaxKey
    }, cacheable::Cacheable, error::MetalError, tensor::dtypes::Dtype
};

mod kv_write;

pub use kv_write::CacheableMpsGraphKvWrite;

pub(crate) fn mps_data_type_for_dtype(dtype: Dtype) -> objc2_metal_performance_shaders::MPSDataType {
    use objc2_metal_performance_shaders::MPSDataType;

    match dtype {
        Dtype::F32 => MPSDataType::Float32,
        Dtype::F16 => MPSDataType::Float16,
    }
}

/// A cacheable wrapper for `MPSMatrixSoftMax` operations.
#[derive(Clone)]
pub struct CacheableMpsSoftMax {
    pub softmax: Retained<MPSMatrixSoftMax>,
    pub key: MpsSoftMaxKey,
}

impl Cacheable for CacheableMpsSoftMax {
    type Key = MpsSoftMaxKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let device = device.ok_or(MetalError::DeviceNotFound)?;
        let softmax = unsafe { MPSMatrixSoftMax::initWithDevice(MPSMatrixSoftMax::alloc(), device) };
        Ok(Self { softmax, key: key.clone() })
    }
}

/// A cacheable wrapper for MPSGraph SDPA operations.
///
/// This struct holds the compiled MPSGraph and associated resources for SDPA operations,
/// allowing them to be cached and reused based on their parameters to avoid recompilation overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpsGraphSdpaFeedBinding {
    Query,
    Key,
    Value,
    Mask,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpsGraphSdpaOutputBinding {
    Attention,
}

#[derive(Clone)]
pub struct CacheableMpsGraphSdpa {
    pub key: MpsGraphSdpaKey,
    pub graph: Retained<mpsg::MPSGraph>,
    pub executable: Retained<mpsg::MPSGraphExecutable>,
    pub q_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub k_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub v_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub mask_placeholder: Option<Retained<mpsg::MPSGraphTensor>>,
    pub output_tensor: Retained<mpsg::MPSGraphTensor>,
    pub feed_tensors: Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>>,
    pub target_tensors: Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>>,
    pub feed_layout: Vec<MpsGraphSdpaFeedBinding>,
    pub result_layout: Vec<MpsGraphSdpaOutputBinding>,
    pub data_type: objc2_metal_performance_shaders::MPSDataType,
    pub accumulator_data_type: Option<objc2_metal_performance_shaders::MPSDataType>,
}

impl Cacheable for CacheableMpsGraphSdpa {
    type Key = MpsGraphSdpaKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        use objc2_foundation::{NSMutableArray, NSMutableDictionary, NSString};

        let data_type = mps_data_type_for_dtype(key.dtype);
        let accumulator_type = key.accumulator_dtype.map(mps_data_type_for_dtype);

        // Create MPSGraph
        // SAFETY: MPSGraph::new returns a retained graph object; no raw pointers are exposed
        // and lifetime is tied to the CacheableMpsGraphSdpa entry holding it.
        let graph = unsafe { mpsg::MPSGraph::new() };

        // Create placeholders
        let q_ph = unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("query"))) };
        let k_ph = unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("key"))) };
        let v_ph = unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("value"))) };

        // Create mask shape and placeholder if causal
        let mask_ph = if key.causal {
            Some(unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("mask"))) })
        } else {
            None
        };

        // Create SDPA operation
        let scale = 1.0f32 / (key.dim as f32).sqrt();
        let attn = if key.causal {
            // For causal, use masked SDPA operation
            unsafe {
                graph.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name(
                    &q_ph,
                    &k_ph,
                    &v_ph,
                    Some(mask_ph.as_ref().unwrap()),
                    scale,
                    Some(&NSString::from_str("sdpa_causal")),
                )
            }
        } else {
            // No mask case
            unsafe {
                graph.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name(
                    &q_ph,
                    &k_ph,
                    &v_ph,
                    scale,
                    Some(&NSString::from_str("sdpa")),
                )
            }
        };

        // Build shaped-type metadata for compilation
        let feed_types: Retained<NSMutableDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> = NSMutableDictionary::dictionary();
        let q_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
        let q_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*q_ph);
        unsafe {
            feed_types.setObject_forKey(&q_type, q_key);
        }
        let k_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
        let k_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*k_ph);
        unsafe {
            feed_types.setObject_forKey(&k_type, k_key);
        }
        let v_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
        let v_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*v_ph);
        unsafe {
            feed_types.setObject_forKey(&v_type, v_key);
        }
        if let Some(mask_placeholder) = mask_ph.as_ref() {
            let mask_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
            let mask_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&**mask_placeholder);
            unsafe {
                feed_types.setObject_forKey(&mask_type, mask_key);
            }
        }

        let target_tensor_list: Retained<NSMutableArray<mpsg::MPSGraphTensor>> = NSMutableArray::array();
        target_tensor_list.addObject(&*attn);

        let target_tensor_array: Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>> =
            unsafe { Retained::cast_unchecked(target_tensor_list) };
        let feed_types_dict: Retained<objc2_foundation::NSDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> =
            unsafe { Retained::cast_unchecked(feed_types) };

        // SAFETY: compileWithDevice uses the provided feed types and target tensors. We
        // ensure feed_types_dict/target_tensor_array contain the placeholders we created
        // and that shaped types match. Returned executable retains the graph's internals.
        let compile_start = std::time::Instant::now();
        let executable = unsafe {
            graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                None,
                &feed_types_dict,
                &target_tensor_array,
                None,
                None,
            )
        };

        let compile_elapsed = compile_start.elapsed();
        metallic_instrumentation::record_metric_async!(metallic_instrumentation::MetricEvent::InternalKernelCompleted {
            parent_op_name: "mpsgraph".to_string(),
            internal_kernel_name: "sdpa_compile".to_string(),
            duration_us: (compile_elapsed.as_secs_f64() * 1e6) as u64,
        });

        let feed_tensors = unsafe {
            executable
                .feedTensors()
                .ok_or_else(|| MetalError::OperationFailed("MPSGraphExecutable did not expose feed tensor order".into()))?
        };
        let target_tensors = unsafe {
            executable
                .targetTensors()
                .ok_or_else(|| MetalError::OperationFailed("MPSGraphExecutable did not expose target tensor order".into()))?
        };

        let mut feed_layout = Vec::with_capacity(feed_tensors.count());
        for idx in 0..feed_tensors.count() {
            let tensor = feed_tensors.objectAtIndex(idx);
            if tensor.isEqual(Some(&*q_ph)) {
                feed_layout.push(MpsGraphSdpaFeedBinding::Query);
            } else if tensor.isEqual(Some(&*k_ph)) {
                feed_layout.push(MpsGraphSdpaFeedBinding::Key);
            } else if tensor.isEqual(Some(&*v_ph)) {
                feed_layout.push(MpsGraphSdpaFeedBinding::Value);
            } else if let Some(mask_placeholder) = mask_ph.as_ref()
                && tensor.isEqual(Some(&**mask_placeholder))
            {
                feed_layout.push(MpsGraphSdpaFeedBinding::Mask);
            } else {
                return Err(MetalError::OperationFailed(
                    "Unexpected tensor encountered in SDPA feed tensor order".into(),
                ));
            }
        }

        let mut result_layout = Vec::with_capacity(target_tensors.count());
        for idx in 0..target_tensors.count() {
            let tensor = target_tensors.objectAtIndex(idx);
            if tensor.isEqual(Some(&*attn)) {
                result_layout.push(MpsGraphSdpaOutputBinding::Attention);
            } else {
                return Err(MetalError::OperationFailed(
                    "Unexpected tensor encountered in SDPA target tensor order".into(),
                ));
            }
        }

        Ok(Self {
            key: key.clone(),
            graph,
            executable,
            q_placeholder: q_ph,
            k_placeholder: k_ph,
            v_placeholder: v_ph,
            mask_placeholder: mask_ph,
            output_tensor: attn,
            feed_tensors,
            target_tensors,
            feed_layout,
            result_layout,
            data_type,
            accumulator_data_type: accumulator_type,
        })
    }
}

pub type CacheableMpsGraphSdpaMaskViews =
    std::cell::RefCell<rustc_hash::FxHashMap<(usize, usize), Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>>>;

/// A cacheable wrapper for reusable MPSGraph mask buffers.
/// This enables mask reuse across different sequence lengths that fit within the same bucket.
#[derive(Clone)]
pub struct CacheableMpsGraphSdpaMask {
    pub key: MpsGraphSdpaMaskKey,
    pub buffer: Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    pub data_type: objc2_metal_performance_shaders::MPSDataType,
    pub seq_q_size: usize,
    pub seq_k_size: usize,
    pub views: CacheableMpsGraphSdpaMaskViews,
}

impl Cacheable for CacheableMpsGraphSdpaMask {
    type Key = MpsGraphSdpaMaskKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        use std::{ffi::c_void, ptr::NonNull};

        use half::f16;

        // Device is not actually needed for this cacheable resource as it's only for creating buffers
        let data_type = mps_data_type_for_dtype(key.dtype);

        // Calculate actual buffer sizes based on the bucketed sequence lengths
        let mut seq_q_size = match key.seq_q_bucket {
            MaskSizeBucket::XSmall => 32,
            MaskSizeBucket::Small => 128,
            MaskSizeBucket::Medium => 512,
            MaskSizeBucket::Large => 1024,
            MaskSizeBucket::XLarge => 2048,
            MaskSizeBucket::XXLarge => 4096,
            MaskSizeBucket::XXXLarge => 8192, // or a larger default
        };

        let seq_k_size = match key.seq_k_bucket {
            MaskSizeBucket::XSmall => 32,
            MaskSizeBucket::Small => 128,
            MaskSizeBucket::Medium => 512,
            MaskSizeBucket::Large => 1024,
            MaskSizeBucket::XLarge => 2048,
            MaskSizeBucket::XXLarge => 4096,
            MaskSizeBucket::XXXLarge => 8192, // or a larger default
        };

        // Ensure the mask covers all possible causal rows for incremental decode scenarios by
        // allocating at least as many rows as we have key positions.
        if seq_q_size < seq_k_size {
            seq_q_size = seq_k_size;
        }

        let total = seq_q_size * seq_k_size;
        let options = objc2_metal::MTLResourceOptions::StorageModeShared;

        // Create a buffer filled with zeros (allowed positions) and -inf (disallowed positions)
        let buffer = match key.dtype {
            Dtype::F16 => {
                let mut host = vec![f16::ZERO; total];
                // For causal mask: upper triangular part (including diagonal) is 0, lower triangular is -inf
                for q_idx in 0..seq_q_size.min(seq_k_size) {
                    // Only fill up to min to avoid out-of-bounds
                    let row_start = q_idx * seq_k_size;
                    for k_idx in (q_idx + 1)..seq_k_size {
                        if row_start + k_idx < total {
                            host[row_start + k_idx] = f16::NEG_INFINITY;
                        }
                    }
                }
                let byte_len = host.len() * core::mem::size_of::<f16>();
                let ptr = NonNull::new(host.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;
                unsafe {
                    // Use the default device for buffer creation
                    let default_device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
                    default_device
                        .newBufferWithBytes_length_options(ptr, byte_len, options)
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?
                }
            }
            Dtype::F32 => {
                let mut host = vec![0.0f32; total];
                // For causal mask: upper triangular part (including diagonal) is 0, lower triangular is -inf
                for q_idx in 0..seq_q_size.min(seq_k_size) {
                    // Only fill up to min to avoid out-of-bounds
                    let row_start = q_idx * seq_k_size;
                    for k_idx in (q_idx + 1)..seq_k_size {
                        if row_start + k_idx < total {
                            host[row_start + k_idx] = f32::NEG_INFINITY;
                        }
                    }
                }
                let byte_len = host.len() * core::mem::size_of::<f32>();
                let ptr = NonNull::new(host.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;
                unsafe {
                    // Use the default device for buffer creation
                    let default_device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
                    default_device
                        .newBufferWithBytes_length_options(ptr, byte_len, options)
                        .ok_or(MetalError::BufferFromBytesCreationFailed)?
                }
            }
        };

        Ok(Self {
            key: key.clone(),
            buffer,
            data_type,
            seq_q_size,
            seq_k_size,
            views: std::cell::RefCell::new(FxHashMap::default()),
        })
    }
}

impl CacheableMpsGraphSdpaMask {
    pub fn view_for(
        &self,
        device: &Retained<ProtocolObject<dyn objc2_metal::MTLDevice>>,
        offset_bytes: usize,
        length_bytes: usize,
    ) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLBuffer>>, MetalError> {
        let mut guard = self.views.borrow_mut();
        if let Some(existing) = guard.get(&(offset_bytes, length_bytes)) {
            return Ok(existing.clone());
        }

        let base_ptr = self.buffer.contents().as_ptr() as *mut u8;
        if base_ptr.is_null() {
            return Err(MetalError::OperationFailed("SDPA mask buffer has null contents pointer".into()));
        }

        let masked_ptr = unsafe { base_ptr.add(offset_bytes) } as *mut std::ffi::c_void;
        let alias_ptr = std::ptr::NonNull::new(masked_ptr).ok_or(MetalError::NullPointer)?;

        let alias = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(
                    alias_ptr,
                    length_bytes,
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                    None,
                )
                .ok_or(MetalError::BufferFromBytesCreationFailed)?
        };

        guard.insert((offset_bytes, length_bytes), alias.clone());
        Ok(alias)
    }
}

/// A cacheable wrapper for MPSGraph fused operations (like SDPA + projection).
#[derive(Clone)]
pub struct CacheableMpsGraphFused {
    pub key: MpsGraphFusedKey,
    pub executable: Retained<mpsg::MPSGraphExecutable>,
    pub feed_layout: Vec<crate::mps_graph::multi_op::GraphFeedBinding>,
    pub result_layout: Vec<crate::mps_graph::multi_op::GraphResultBinding>,
    pub data_type: objc2_metal_performance_shaders::MPSDataType,
    pub accumulator_data_type: Option<objc2_metal_performance_shaders::MPSDataType>,
}

impl Cacheable for CacheableMpsGraphFused {
    type Key = MpsGraphFusedKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        // This implementation will delegate to the multi_op module
        use crate::{cacheable_resources::mps_data_type_for_dtype, mps_graph::multi_op::MultiOpGraphBuilder};

        let data_type = mps_data_type_for_dtype(key.dtype);
        let accumulator_type = key.accumulator_dtype.map(mps_data_type_for_dtype);

        // TODO: Why arent we properly handling multiop here?
        let _builder = MultiOpGraphBuilder::new(data_type)?;

        // For now, we'll use simplified empty layouts, but in a real implementation
        // this would use the multi_op module functions
        let feed_layout = vec![]; // Would come from multi_op module
        let result_layout = vec![]; // Would come from multi_op module

        // Create a placeholder executable - in a real implementation, the builder would be used
        // For now, we just create an empty executable to satisfy the type
        use objc2_foundation::{NSArray, NSMutableArray, NSMutableDictionary};
        use objc2_metal_performance_shaders_graph as mpsg;

        // Create an empty graph and executable for now
        let graph = unsafe { mpsg::MPSGraph::new() };
        let feed_types: Retained<NSMutableDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> = NSMutableDictionary::dictionary();
        let target_tensor_list: Retained<NSMutableArray<mpsg::MPSGraphTensor>> = NSMutableArray::array();
        let target_tensor_array: Retained<NSArray<mpsg::MPSGraphTensor>> = unsafe { Retained::cast_unchecked(target_tensor_list) };
        let feed_types_dict: Retained<objc2_foundation::NSDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> =
            unsafe { Retained::cast_unchecked(feed_types) };

        let executable = unsafe {
            graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                None,
                &feed_types_dict,
                &target_tensor_array,
                None,
                None,
            )
        };

        Ok(Self {
            key: key.clone(),
            executable,
            feed_layout,
            result_layout,
            data_type,
            accumulator_data_type: accumulator_type,
        })
    }
}

/// A cacheable wrapper for MPSMatrixMultiplication.
///
/// This struct implements the Cacheable trait, allowing MPS matrix multiplication
/// operations to be cached and reused based on their parameters.
#[derive(Clone)]
pub struct CacheableMpsGemm {
    pub gemm: Retained<MPSMatrixMultiplication>,
    pub key: MpsGemmKey,
}

impl Cacheable for CacheableMpsGemm {
    type Key = MpsGemmKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let device = device.ok_or(MetalError::DeviceNotFound)?;
        let gemm = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                device,
                key.transpose_left,
                key.transpose_right,
                key.result_rows,
                key.result_columns,
                key.interior_columns,
                key.alpha as f64,
                key.beta as f64,
            )
        };
        Ok(Self { gemm, key: key.clone() })
    }
}

/// A cacheable wrapper for MPSMatrixDescriptor.
///
/// This struct implements the Cacheable trait, allowing MPS matrix descriptors
/// to be cached and reused based on their dimensions.
#[derive(Clone)]
pub struct CacheableMpsMatrixDescriptor {
    pub descriptor: Retained<MPSMatrixDescriptor>,
    pub key: MpsMatrixDescriptorKey,
}

impl Cacheable for CacheableMpsMatrixDescriptor {
    type Key = MpsMatrixDescriptorKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let data_type = mps_data_type_for_dtype(key.dtype);
        let descriptor = unsafe {
            if key.matrices > 1 || key.matrix_bytes != key.row_bytes * key.rows {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_matrices_rowBytes_matrixBytes_dataType(
                    key.rows,
                    key.columns,
                    key.matrices,
                    key.row_bytes,
                    key.matrix_bytes,
                    data_type,
                )
            } else {
                MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(key.rows, key.columns, key.row_bytes, data_type)
            }
        };
        Ok(Self {
            descriptor,
            key: key.clone(),
        })
    }
}
