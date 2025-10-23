use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSMutableArray, NSNumber};
use objc2_metal_performance_shaders::MPSCommandBuffer;
use objc2_metal_performance_shaders_graph as mpsg;

use crate::{MetalError, Tensor, TensorElement};

// Helper function to convert dims to NSNumber array
fn dims_to_nsnumbers(dims: &[usize]) -> Result<Retained<NSArray<NSNumber>>, MetalError> {
    let array = NSMutableArray::array();

    for &dim in dims {
        let nsnumber = NSNumber::numberWithUnsignedInteger(dim);
        array.addObject(&*nsnumber);
    }

    Ok(unsafe { Retained::cast_unchecked(array) })
}

/// A builder for multi-op MPSGraph executables
pub struct MultiOpGraphBuilder {
    graph: Retained<mpsg::MPSGraph>,
    data_type: objc2_metal_performance_shaders::MPSDataType,
}

impl MultiOpGraphBuilder {
    pub fn new(data_type: objc2_metal_performance_shaders::MPSDataType) -> Result<Self, MetalError> {
        // SAFETY: MPSGraph::new returns a retained graph object
        let graph = unsafe { mpsg::MPSGraph::new() };

        Ok(Self { graph, data_type })
    }

    /// Build an SDPA + projection fused graph
    pub fn build_sdpa_projection(
        &self,
        batch: usize,
        seq_q: usize,
        seq_k: usize,
        dim: usize,
        output_dim: usize,
        causal: bool,
    ) -> Result<Retained<mpsg::MPSGraphExecutable>, MetalError> {
        use objc2_foundation::{NSMutableArray, NSMutableDictionary};

        // Define tensor shapes
        let q_shape_4d = dims_to_nsnumbers(&[batch, 1, seq_q, dim])?;
        let k_shape_4d = dims_to_nsnumbers(&[batch, 1, seq_k, dim])?;
        let v_shape_4d = dims_to_nsnumbers(&[batch, 1, seq_k, dim])?;
        let proj_weight_shape = dims_to_nsnumbers(&[dim, output_dim])?;
        let _output_shape = dims_to_nsnumbers(&[batch, seq_q, output_dim])?;

        // Create placeholders
        let q_ph = unsafe {
            self.graph.placeholderWithShape_dataType_name(
                Some(&*q_shape_4d),
                self.data_type,
                Some(&objc2_foundation::NSString::from_str("query")),
            )
        };
        let k_ph = unsafe {
            self.graph.placeholderWithShape_dataType_name(
                Some(&*k_shape_4d),
                self.data_type,
                Some(&objc2_foundation::NSString::from_str("key")),
            )
        };
        let v_ph = unsafe {
            self.graph.placeholderWithShape_dataType_name(
                Some(&*v_shape_4d),
                self.data_type,
                Some(&objc2_foundation::NSString::from_str("value")),
            )
        };

        // Create mask placeholder if causal
        let mask_ph = if causal {
            let mask_shape_4d = dims_to_nsnumbers(&[1, 1, seq_q, seq_k])?;
            let mask_ph = unsafe {
                self.graph.placeholderWithShape_dataType_name(
                    Some(&*mask_shape_4d),
                    self.data_type,
                    Some(&objc2_foundation::NSString::from_str("mask")),
                )
            };
            Some(mask_ph)
        } else {
            None
        };

        // Create projection weight placeholder
        let proj_weight_ph = unsafe {
            self.graph.placeholderWithShape_dataType_name(
                Some(&*proj_weight_shape),
                self.data_type,
                Some(&objc2_foundation::NSString::from_str("proj_weight")),
            )
        };

        // Create SDPA operation
        let scale = 1.0f32 / (dim as f32).sqrt();
        let sdpa_output = if let Some(mask_placeholder) = mask_ph.as_ref() {
            unsafe {
                self.graph
                    .scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_maskTensor_scale_name(
                        &q_ph,
                        &k_ph,
                        &v_ph,
                        Some(mask_placeholder),
                        scale,
                        Some(&objc2_foundation::NSString::from_str("sdpa")),
                    )
            }
        } else {
            unsafe {
                self.graph
                    .scaledDotProductAttentionWithQueryTensor_keyTensor_valueTensor_scale_name(
                        &q_ph,
                        &k_ph,
                        &v_ph,
                        scale,
                        Some(&objc2_foundation::NSString::from_str("sdpa")),
                    )
            }
        };

        // Reshape SDPA output from [B, 1, S, D] to [B, S, D] for matrix multiplication
        let sdpa_reshaped = unsafe {
            self.graph.reshapeTensor_withShape_name(
                &sdpa_output,
                &*dims_to_nsnumbers(&[batch, seq_q, dim])?,
                Some(&objc2_foundation::NSString::from_str("sdpa_reshape")),
            )
        };

        // Create matrix multiplication for projection
        let proj_output = unsafe {
            self.graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(
                &sdpa_reshaped,
                &proj_weight_ph,
                Some(&objc2_foundation::NSString::from_str("projection")),
            )
        };

        // Build feed types dictionary
        let feed_types: Retained<NSMutableDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> = NSMutableDictionary::dictionary();

        // Add Q tensor type
        let q_type = unsafe {
            mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&*q_shape_4d), self.data_type)
        };
        let q_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*q_ph);
        unsafe {
            feed_types.setObject_forKey(&q_type, q_key);
        }

        // Add K tensor type
        let k_type = unsafe {
            mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&*k_shape_4d), self.data_type)
        };
        let k_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*k_ph);
        unsafe {
            feed_types.setObject_forKey(&k_type, k_key);
        }

        // Add V tensor type
        let v_type = unsafe {
            mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&*v_shape_4d), self.data_type)
        };
        let v_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*v_ph);
        unsafe {
            feed_types.setObject_forKey(&v_type, v_key);
        }

        // Add mask tensor type if causal
        if let Some(mask_placeholder) = mask_ph.as_ref() {
            let mask_type = unsafe {
                mpsg::MPSGraphShapedType::initWithShape_dataType(
                    mpsg::MPSGraphShapedType::alloc(),
                    Some(&*dims_to_nsnumbers(&[1, 1, seq_q, seq_k])?),
                    self.data_type,
                )
            };
            let mask_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&**mask_placeholder);
            unsafe {
                feed_types.setObject_forKey(&mask_type, mask_key);
            }
        }

        // Add projection weight type
        let proj_weight_type = unsafe {
            mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&*proj_weight_shape), self.data_type)
        };
        let proj_weight_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*proj_weight_ph);
        unsafe {
            feed_types.setObject_forKey(&proj_weight_type, proj_weight_key);
        }

        // Create target tensors array
        let target_tensor_list: Retained<NSMutableArray<mpsg::MPSGraphTensor>> = NSMutableArray::array();
        target_tensor_list.addObject(&*proj_output);

        let target_tensor_array: Retained<NSArray<mpsg::MPSGraphTensor>> = unsafe { Retained::cast_unchecked(target_tensor_list) };

        let feed_types_dict: Retained<objc2_foundation::NSDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> =
            unsafe { Retained::cast_unchecked(feed_types) };

        // Compile the executable
        // SAFETY: We provide valid feed types and target tensors
        let executable = unsafe {
            self.graph
                .compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                    None,
                    &feed_types_dict,
                    &target_tensor_array,
                    None,
                    None,
                )
        };

        Ok(executable)
    }
}

/// Helper functions to extract layout information from MPSGraphExecutable
pub fn extract_feed_layout(executable: &mpsg::MPSGraphExecutable) -> Result<Vec<GraphFeedBinding>, MetalError> {
    let feed_tensors = unsafe {
        executable
            .feedTensors()
            .ok_or_else(|| MetalError::OperationFailed("MPSGraphExecutable did not expose feed tensor order".into()))?
    };

    let mut feed_layout = Vec::with_capacity(feed_tensors.count());
    for idx in 0..feed_tensors.count() {
        let _tensor = feed_tensors.objectAtIndex(idx);
        // In a real implementation, we would identify tensors by name or other metadata
        // For now, we'll create generic bindings
        feed_layout.push(GraphFeedBinding::Generic(idx));
    }

    Ok(feed_layout)
}

pub fn extract_result_layout(executable: &mpsg::MPSGraphExecutable) -> Result<Vec<GraphResultBinding>, MetalError> {
    let target_tensors = unsafe {
        executable
            .targetTensors()
            .ok_or_else(|| MetalError::OperationFailed("MPSGraphExecutable did not expose target tensor order".into()))?
    };

    let mut result_layout = Vec::with_capacity(target_tensors.count());
    for idx in 0..target_tensors.count() {
        let _tensor = target_tensors.objectAtIndex(idx);
        // In a real implementation, we would identify result tensors by name or other metadata
        // For now, we'll create generic bindings
        result_layout.push(GraphResultBinding::Generic(idx));
    }

    Ok(result_layout)
}

/// A cacheable wrapper for MPSGraph fused operations (like SDPA + projection).
#[derive(Clone)]
pub struct CacheableMpsGraphFused {
    pub key: MpsGraphFusedKey,
    pub executable: Retained<mpsg::MPSGraphExecutable>,
    pub feed_layout: Vec<GraphFeedBinding>,
    pub result_layout: Vec<GraphResultBinding>,
    pub data_type: objc2_metal_performance_shaders::MPSDataType,
}

/// Key for caching fused MPSGraph operations
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MpsGraphFusedKey {
    pub batch: usize,
    pub seq_q: usize,
    pub seq_k: usize,
    pub dim: usize,
    pub output_dim: usize,
    pub causal: bool,
    pub operation_type: FusedOperationType,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FusedOperationType {
    SdpaProjection,
    // Additional fused operations can be added here
}

/// A trait for strongly-typed executable layouts that can be derived from MPSGraphExecutable metadata.
pub trait ExecutableLayout {
    /// Get the ordered list of feed tensor types for this executable
    fn feed_bindings(&self) -> &[GraphFeedBinding];

    /// Get the ordered list of result tensor types for this executable
    fn result_bindings(&self) -> &[GraphResultBinding];
}

/// Represents the layout of an executable that can be extended with additional nodes
#[derive(Clone, Debug)]
pub struct ExtendableExecutableLayout {
    pub feed_bindings: Vec<GraphFeedBinding>,
    pub result_bindings: Vec<GraphResultBinding>,
}

impl ExecutableLayout for ExtendableExecutableLayout {
    fn feed_bindings(&self) -> &[GraphFeedBinding] {
        &self.feed_bindings
    }

    fn result_bindings(&self) -> &[GraphResultBinding] {
        &self.result_bindings
    }
}

impl TryFrom<&mpsg::MPSGraphExecutable> for ExtendableExecutableLayout {
    type Error = MetalError;

    fn try_from(executable: &mpsg::MPSGraphExecutable) -> Result<Self, Self::Error> {
        let feed_layout = extract_feed_layout(executable)?;
        let result_layout = extract_result_layout(executable)?;

        Ok(Self {
            feed_bindings: feed_layout,
            result_bindings: result_layout,
        })
    }
}

impl TryFrom<&Retained<mpsg::MPSGraphExecutable>> for ExtendableExecutableLayout {
    type Error = MetalError;

    fn try_from(executable: &Retained<mpsg::MPSGraphExecutable>) -> Result<Self, Self::Error> {
        let feed_layout = extract_feed_layout(executable)?;
        let result_layout = extract_result_layout(executable)?;

        Ok(Self {
            feed_bindings: feed_layout,
            result_bindings: result_layout,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphFeedBinding {
    /// A generic feed binding with an index
    Generic(usize),

    /// SDPA-specific bindings
    SdpaQuery,
    SdpaKey,
    SdpaValue,
    SdpaMask,

    /// Projection-specific bindings
    ProjectionInput,
    ProjectionWeight,
    ProjectionBias,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphResultBinding {
    /// A generic result binding with an index
    Generic(usize),

    /// SDPA-specific result
    SdpaAttention,

    /// Projection-specific result
    ProjectionOutput,

    /// General attention output
    AttentionOutput,
}

/// A generic graph interface that can handle multi-node executables
pub struct GenericGraphInterface<'a> {
    executable: &'a mpsg::MPSGraphExecutable,
    layout: &'a dyn ExecutableLayout,
}

impl<'a> GenericGraphInterface<'a> {
    pub fn new(executable: &'a mpsg::MPSGraphExecutable, layout: &'a dyn ExecutableLayout) -> Self {
        Self { executable, layout }
    }

    pub fn bind_inputs<T: TensorElement>(&self, inputs: &[&Tensor<T>]) -> Result<Retained<NSArray<mpsg::MPSGraphTensorData>>, MetalError> {
        use crate::{
            cacheable_resources::mps_data_type_for_dtype, mps_graph::bindings::{GraphBindingSpec, GraphTensorDataArrayBuilder}
        };

        let builder = GraphTensorDataArrayBuilder::new();

        for (i, _binding) in self.layout.feed_bindings().iter().enumerate() {
            if i >= inputs.len() {
                return Err(MetalError::OperationFailed(format!(
                    "Not enough input tensors provided for graph feed binding {}",
                    i
                )));
            }

            let tensor = inputs[i];
            let data_type = mps_data_type_for_dtype(tensor.dtype);
            let shape_nsnumbers = dims_to_nsnumbers(&tensor.dims)?;
            let spec = GraphBindingSpec {
                expected_shape: &shape_nsnumbers,
                data_type,
            };
            let descriptor = spec.try_from_tensor(tensor)?;
            builder.push(&descriptor)?;
        }

        Ok(builder.finish())
    }

    pub fn bind_outputs<T: TensorElement>(
        &self,
        outputs: &[&Tensor<T>],
    ) -> Result<Retained<NSArray<mpsg::MPSGraphTensorData>>, MetalError> {
        use crate::{
            cacheable_resources::mps_data_type_for_dtype, mps_graph::bindings::{GraphBindingSpec, GraphTensorDataArrayBuilder}
        };

        let builder = GraphTensorDataArrayBuilder::new();

        for (i, _binding) in self.layout.result_bindings().iter().enumerate() {
            if i >= outputs.len() {
                return Err(MetalError::OperationFailed(format!(
                    "Not enough output tensors provided for graph result binding {}",
                    i
                )));
            }

            let tensor = outputs[i];
            let data_type = mps_data_type_for_dtype(tensor.dtype);
            let shape_nsnumbers = dims_to_nsnumbers(&tensor.dims)?;
            let spec = GraphBindingSpec {
                expected_shape: &shape_nsnumbers,
                data_type,
            };
            let descriptor = spec.try_from_tensor(tensor)?;
            builder.push(&descriptor)?;
        }

        Ok(builder.finish())
    }

    pub fn encode(
        &self,
        mps_command_buffer: &MPSCommandBuffer,
        input_bindings: &NSArray<mpsg::MPSGraphTensorData>,
        result_bindings: Option<&NSArray<mpsg::MPSGraphTensorData>>,
    ) -> Result<(), MetalError> {
        // SAFETY: We're using the provided command buffer and our verified tensor bindings
        unsafe {
            self.executable.encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                mps_command_buffer,
                input_bindings,
                result_bindings,
                None,
            );
        }

        Ok(())
    }
}
