use objc2::{
    AnyThread, rc::Retained, runtime::{NSObjectProtocol, ProtocolObject}
};
use objc2_foundation::{NSMutableArray, NSMutableDictionary, NSNumber, NSString};
use objc2_metal::MTLDevice;
use objc2_metal_performance_shaders_graph as mpsg;

use crate::{cache_keys::MpsGraphKvWriteKey, cacheable::Cacheable, caching::CacheableKernel, error::MetalError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpsGraphKvWriteFeedBinding {
    KIn,
    VIn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpsGraphKvWriteResultBinding {
    KOut,
    VOut,
}

#[derive(Clone)]
pub struct CacheableMpsGraphKvWrite {
    pub key: MpsGraphKvWriteKey,
    pub graph: Retained<mpsg::MPSGraph>,
    pub executable: Retained<mpsg::MPSGraphExecutable>,
    pub k_in_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub v_in_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub k_out_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub v_out_placeholder: Retained<mpsg::MPSGraphTensor>,
    pub feed_tensors: Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>>,
    pub target_tensors: Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>>,
    pub feed_layout: Vec<MpsGraphKvWriteFeedBinding>,
    pub result_layout: Vec<MpsGraphKvWriteResultBinding>,
    pub data_type: objc2_metal_performance_shaders::MPSDataType,
}

impl Cacheable for CacheableMpsGraphKvWrite {
    type Key = MpsGraphKvWriteKey;

    fn cache_key(&self) -> Self::Key {
        self.key.clone()
    }

    fn from_key(key: &Self::Key, _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>) -> Result<Self, MetalError> {
        let shape_array: Retained<objc2_foundation::NSArray<NSNumber>> = {
            let arr = NSMutableArray::array();
            arr.addObject(&*NSNumber::numberWithUnsignedInteger(key.heads));
            arr.addObject(&*NSNumber::numberWithUnsignedInteger(key.seq_bucket));
            arr.addObject(&*NSNumber::numberWithUnsignedInteger(key.head_dim));
            unsafe { Retained::cast_unchecked(arr) }
        };

        let data_type = key.dtype.into();
        let graph = unsafe { mpsg::MPSGraph::new() };

        let k_in = unsafe { graph.placeholderWithShape_dataType_name(Some(&shape_array), data_type, Some(&NSString::from_str("k_in"))) };
        let v_in = unsafe { graph.placeholderWithShape_dataType_name(Some(&shape_array), data_type, Some(&NSString::from_str("v_in"))) };

        let zero = unsafe { graph.constantWithScalar_shape_dataType(0.0f32 as f64, shape_array.as_ref(), data_type) };

        let k_out = unsafe { graph.additionWithPrimaryTensor_secondaryTensor_name(&k_in, &zero, Some(&NSString::from_str("k_out"))) };
        let v_out = unsafe { graph.additionWithPrimaryTensor_secondaryTensor_name(&v_in, &zero, Some(&NSString::from_str("v_out"))) };

        let feed_types_dict: Retained<objc2_foundation::NSDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> = {
            let feed_types: Retained<NSMutableDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> =
                NSMutableDictionary::dictionary();

            let k_in_shape_type = unsafe {
                mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&shape_array), data_type)
            };
            let k_in_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*k_in);
            unsafe { feed_types.setObject_forKey(&k_in_shape_type, k_in_key) };

            let v_in_shape_type = unsafe {
                mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&shape_array), data_type)
            };
            let v_in_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*v_in);
            unsafe { feed_types.setObject_forKey(&v_in_shape_type, v_in_key) };

            unsafe { Retained::cast_unchecked(feed_types) }
        };

        let target_tensor_array: Retained<objc2_foundation::NSArray<mpsg::MPSGraphTensor>> = {
            let target_tensor_list: Retained<NSMutableArray<mpsg::MPSGraphTensor>> = NSMutableArray::array();
            target_tensor_list.addObject(&*k_out);
            target_tensor_list.addObject(&*v_out);
            unsafe { Retained::cast_unchecked(target_tensor_list) }
        };

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
            if tensor.isEqual(Some(&*k_in)) {
                feed_layout.push(MpsGraphKvWriteFeedBinding::KIn);
            } else if tensor.isEqual(Some(&*v_in)) {
                feed_layout.push(MpsGraphKvWriteFeedBinding::VIn);
            } else {
                return Err(MetalError::OperationFailed(
                    "Unexpected tensor encountered in KV Write feed tensor order".into(),
                ));
            }
        }

        let mut result_layout = Vec::with_capacity(target_tensors.count());
        for idx in 0..target_tensors.count() {
            let tensor = target_tensors.objectAtIndex(idx);
            if tensor.isEqual(Some(&*k_out)) {
                result_layout.push(MpsGraphKvWriteResultBinding::KOut);
            } else if tensor.isEqual(Some(&*v_out)) {
                result_layout.push(MpsGraphKvWriteResultBinding::VOut);
            } else {
                return Err(MetalError::OperationFailed(
                    "Unexpected tensor encountered in KV Write target tensor order".into(),
                ));
            }
        }

        Ok(Self {
            key: key.clone(),
            graph,
            executable,
            k_in_placeholder: k_in,
            v_in_placeholder: v_in,
            k_out_placeholder: k_out,
            v_out_placeholder: v_out,
            feed_tensors,
            target_tensors,
            feed_layout,
            result_layout,
            data_type,
        })
    }
}

/// Cache adapter for the MPSGraph KV write executable used by fast KV updates.
pub struct KvWriteGraphKernel;

impl CacheableKernel for KvWriteGraphKernel {
    type Key = MpsGraphKvWriteKey;
    type CachedResource = CacheableMpsGraphKvWrite;
    type Params = MpsGraphKvWriteKey;

    const CACHE_NAME: &'static str = "mpsgraph_kv_write";

    #[inline]
    fn create_cache_key(params: &Self::Params) -> Self::Key {
        params.clone()
    }

    #[inline]
    fn create_cached_resource(
        key: &Self::Key,
        _device: Option<&Retained<ProtocolObject<dyn MTLDevice>>>,
    ) -> Result<Self::CachedResource, MetalError> {
        CacheableMpsGraphKvWrite::from_key(key, None)
    }
}
