use std::{
    cell::RefCell, ffi::c_void, hash::{Hash, Hasher}, ptr::NonNull
};

use half::f16;
use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSMutableArray, NSMutableDictionary, NSObjectProtocol, NSString};
use objc2_metal::{MTLBuffer, MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph as mpsg;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{caching::CacheableKernel, error::MetalError, tensor::dtypes::Dtype};

/// Bucketing for mask sizes to enable reuse across different sequence lengths.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MaskSizeBucket {
    XSmall,   // 1-32
    Small,    // 33-128
    Medium,   // 129-512
    Large,    // 513-1024
    XLarge,   // 1025-2048
    XXLarge,  // 2049-4096
    XXXLarge, // >4096
}

impl From<usize> for MaskSizeBucket {
    fn from(seq_len: usize) -> Self {
        match seq_len {
            0..=32 => MaskSizeBucket::XSmall,
            33..=128 => MaskSizeBucket::Small,
            129..=512 => MaskSizeBucket::Medium,
            513..=1024 => MaskSizeBucket::Large,
            1025..=2048 => MaskSizeBucket::XLarge,
            2049..=4096 => MaskSizeBucket::XXLarge,
            _ => MaskSizeBucket::XXXLarge,
        }
    }
}

/// Key for MPSGraph SDPA operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGraphSdpaKey {
    pub batch: usize,
    pub dim: usize,
    pub causal: bool,
    pub dtype: Dtype,
    pub accumulator_dtype: Option<Dtype>,
}

impl PartialEq for MpsGraphSdpaKey {
    fn eq(&self, other: &Self) -> bool {
        self.batch == other.batch
            && self.dim == other.dim
            && self.causal == other.causal
            && self.dtype == other.dtype
            && self.accumulator_dtype == other.accumulator_dtype
    }
}

impl Eq for MpsGraphSdpaKey {}

impl Hash for MpsGraphSdpaKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.batch.hash(state);
        self.dim.hash(state);
        self.causal.hash(state);
        self.dtype.hash(state);
        self.accumulator_dtype.hash(state);
    }
}

/// Key for reusable mask buffers in MPSGraph SDPA.
/// This enables mask reuse across different sequence lengths that fit within the same bucket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MpsGraphSdpaMaskKey {
    pub causal: bool,
    pub dtype: Dtype,
    pub head_dim: usize,
    pub seq_q_bucket: MaskSizeBucket,
    pub seq_k_bucket: MaskSizeBucket,
}

impl PartialEq for MpsGraphSdpaMaskKey {
    fn eq(&self, other: &Self) -> bool {
        self.causal == other.causal
            && self.dtype == other.dtype
            && self.head_dim == other.head_dim
            && self.seq_q_bucket == other.seq_q_bucket
            && self.seq_k_bucket == other.seq_k_bucket
    }
}

impl Eq for MpsGraphSdpaMaskKey {}

impl Hash for MpsGraphSdpaMaskKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.causal.hash(state);
        self.dtype.hash(state);
        self.head_dim.hash(state);
        self.seq_q_bucket.hash(state);
        self.seq_k_bucket.hash(state);
    }
}

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
    pub feed_tensors: Retained<NSArray<mpsg::MPSGraphTensor>>,
    pub target_tensors: Retained<NSArray<mpsg::MPSGraphTensor>>,
    pub feed_layout: Vec<MpsGraphSdpaFeedBinding>,
    pub result_layout: Vec<MpsGraphSdpaOutputBinding>,
    pub data_type: MPSDataType,
    pub accumulator_data_type: Option<MPSDataType>,
}

impl CacheableMpsGraphSdpa {
    pub fn key(&self) -> &MpsGraphSdpaKey {
        &self.key
    }

    pub fn from_key(key: &MpsGraphSdpaKey, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let data_type = key.dtype.into();
        let accumulator_type = key.accumulator_dtype.map(Into::into);

        let graph = unsafe { mpsg::MPSGraph::new() };

        let q_ph = unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("query"))) };
        let k_ph = unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("key"))) };
        let v_ph = unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("value"))) };

        let mask_ph = if key.causal {
            Some(unsafe { graph.placeholderWithShape_dataType_name(None, data_type, Some(&NSString::from_str("mask"))) })
        } else {
            None
        };

        let scale = 1.0f32 / (key.dim as f32).sqrt();
        let attn = if let Some(mask_placeholder) = mask_ph.as_ref() {
            unsafe {
                graph.scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name(
                    &q_ph,
                    &k_ph,
                    &v_ph,
                    Some(mask_placeholder),
                    scale,
                    Some(&NSString::from_str("sdpa_causal")),
                )
            }
        } else {
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

        let feed_types: Retained<NSMutableDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> = NSMutableDictionary::dictionary();

        let q_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
        let q_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*q_ph);
        unsafe { feed_types.setObject_forKey(&q_type, q_key) };

        let k_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
        let k_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*k_ph);
        unsafe { feed_types.setObject_forKey(&k_type, k_key) };

        let v_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
        let v_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*v_ph);
        unsafe { feed_types.setObject_forKey(&v_type, v_key) };

        if let Some(mask_placeholder) = mask_ph.as_ref() {
            let mask_type = unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), None, data_type) };
            let mask_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&**mask_placeholder);
            unsafe { feed_types.setObject_forKey(&mask_type, mask_key) };
        }

        let target_tensor_list: Retained<NSMutableArray<mpsg::MPSGraphTensor>> = NSMutableArray::array();
        target_tensor_list.addObject(&*attn);

        let feed_types_dict: Retained<objc2_foundation::NSDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> =
            unsafe { Retained::cast_unchecked(feed_types) };
        let target_tensor_array: Retained<NSArray<mpsg::MPSGraphTensor>> = unsafe { Retained::cast_unchecked(target_tensor_list) };

        let executable = unsafe {
            graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                None,
                &feed_types_dict,
                &target_tensor_array,
                None,
                None,
            )
        };

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
            } else if mask_ph.as_ref().is_some_and(|mask| tensor.isEqual(Some(&**mask))) {
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

pub type CacheableMpsGraphSdpaMaskViews = RefCell<FxHashMap<(usize, usize), Retained<ProtocolObject<dyn MTLBuffer>>>>;

#[derive(Clone)]
pub struct CacheableMpsGraphSdpaMask {
    pub key: MpsGraphSdpaMaskKey,
    pub buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub data_type: MPSDataType,
    pub seq_q_size: usize,
    pub seq_k_size: usize,
    pub views: CacheableMpsGraphSdpaMaskViews,
}

impl CacheableMpsGraphSdpaMask {
    pub fn key(&self) -> &MpsGraphSdpaMaskKey {
        &self.key
    }

    pub fn from_key(key: &MpsGraphSdpaMaskKey, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let data_type = key.dtype.into();

        let mut seq_q_size = mask_bucket_size(key.seq_q_bucket);
        let seq_k_size = mask_bucket_size(key.seq_k_bucket);
        if seq_q_size < seq_k_size {
            seq_q_size = seq_k_size;
        }

        let total = seq_q_size * seq_k_size;
        let buffer = match key.dtype {
            Dtype::F16 => build_mask_buffer_f16(total, seq_q_size, seq_k_size)?,
            Dtype::F32 => build_mask_buffer_f32(total, seq_q_size, seq_k_size)?,
            Dtype::U8 | Dtype::U32 => {
                return Err(MetalError::OperationFailed(
                    "U8/U32 dtype not supported for causal mask buffers".into(),
                ));
            }
        };

        Ok(Self {
            key: key.clone(),
            buffer,
            data_type,
            seq_q_size,
            seq_k_size,
            views: RefCell::new(FxHashMap::default()),
        })
    }

    pub fn view_for(
        &self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        offset_bytes: usize,
        length_bytes: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
        if let Some(existing) = self.views.borrow().get(&(offset_bytes, length_bytes)) {
            return Ok(existing.clone());
        }

        let base_ptr = self.buffer.contents().as_ptr() as *mut u8;
        if base_ptr.is_null() {
            return Err(MetalError::OperationFailed("SDPA mask buffer has null contents pointer".into()));
        }

        let masked_ptr = unsafe { base_ptr.add(offset_bytes) } as *mut c_void;
        let alias_ptr = NonNull::new(masked_ptr).ok_or(MetalError::NullPointer)?;

        let alias = unsafe {
            device
                .newBufferWithBytesNoCopy_length_options_deallocator(alias_ptr, length_bytes, MTLResourceOptions::StorageModeShared, None)
                .ok_or(MetalError::BufferFromBytesCreationFailed)?
        };

        self.views.borrow_mut().insert((offset_bytes, length_bytes), alias.clone());
        Ok(alias)
    }
}

/// Cache adapter for compiled MPSGraph SDPA executables.
pub struct MpsGraphSdpaKernel;

impl CacheableKernel for MpsGraphSdpaKernel {
    type Key = MpsGraphSdpaKey;
    type CachedResource = CacheableMpsGraphSdpa;
    type Params = MpsGraphSdpaKey;

    const CACHE_NAME: &'static str = "mpsgraph_sdpa";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableMpsGraphSdpa::from_key(key, device)
    }
}

/// Cache adapter for SDPA mask arena resources.
pub struct MpsGraphSdpaMaskKernel;

impl CacheableKernel for MpsGraphSdpaMaskKernel {
    type Key = MpsGraphSdpaMaskKey;
    type CachedResource = CacheableMpsGraphSdpaMask;
    type Params = MpsGraphSdpaMaskKey;

    const CACHE_NAME: &'static str = "mpsgraph_mask";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableMpsGraphSdpaMask::from_key(key, device)
    }
}

fn mask_bucket_size(bucket: MaskSizeBucket) -> usize {
    match bucket {
        MaskSizeBucket::XSmall => 32,
        MaskSizeBucket::Small => 128,
        MaskSizeBucket::Medium => 512,
        MaskSizeBucket::Large => 1024,
        MaskSizeBucket::XLarge => 2048,
        MaskSizeBucket::XXLarge => 4096,
        MaskSizeBucket::XXXLarge => 8192,
    }
}

fn build_mask_buffer_f16(
    total: usize,
    seq_q_size: usize,
    seq_k_size: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut host = vec![f16::ZERO; total];
    for q_idx in 0..seq_q_size.min(seq_k_size) {
        let row_start = q_idx * seq_k_size;
        for k_idx in (q_idx + 1)..seq_k_size {
            let idx = row_start + k_idx;
            if idx < total {
                host[idx] = f16::NEG_INFINITY;
            }
        }
    }
    let byte_len = host.len() * core::mem::size_of::<f16>();
    let ptr = NonNull::new(host.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;
    let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
    unsafe {
        device
            .newBufferWithBytes_length_options(ptr, byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferFromBytesCreationFailed)
    }
}

fn build_mask_buffer_f32(
    total: usize,
    seq_q_size: usize,
    seq_k_size: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, MetalError> {
    let mut host = vec![0.0f32; total];
    for q_idx in 0..seq_q_size.min(seq_k_size) {
        let row_start = q_idx * seq_k_size;
        for k_idx in (q_idx + 1)..seq_k_size {
            let idx = row_start + k_idx;
            if idx < total {
                host[idx] = f32::NEG_INFINITY;
            }
        }
    }
    let byte_len = host.len() * core::mem::size_of::<f32>();
    let ptr = NonNull::new(host.as_ptr() as *mut c_void).ok_or(MetalError::NullPointer)?;
    let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceNotFound)?;
    unsafe {
        device
            .newBufferWithBytes_length_options(ptr, byte_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferFromBytesCreationFailed)
    }
}
