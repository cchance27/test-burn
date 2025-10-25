use objc2::{AnyThread, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSArray, NSMutableArray, NSNumber, NSString};
use objc2_metal_performance_shaders_graph as mpsg;

use crate::{
    CommandBuffer, Context, MetalError, Operation, Tensor, TensorElement, kernels::{DefaultKernelInvocable, GraphKernel, GraphKernelAccumulator, GraphKernelDtypePolicy}, mps_graph::bindings::{GraphBindingSpec, GraphTensorDataArrayBuilder}, resource_cache::ResourceCache
};

pub struct RepeatKvHeadsGraphOp;

struct RepeatKvHeadsGraph<T: TensorElement> {
    executable: Retained<mpsg::MPSGraphExecutable>,
    input: Tensor<T>,
    output: Tensor<T>,
    data_type: objc2_metal_performance_shaders::MPSDataType,
}

impl RepeatKvHeadsGraphOp {
    fn build_graph(
        data_type: objc2_metal_performance_shaders::MPSDataType,
        canonical_heads: usize,
        group_size: usize,
        seq: usize,
        head_dim: usize,
    ) -> Result<Retained<mpsg::MPSGraphExecutable>, MetalError> {
        use objc2_foundation::{NSMutableArray, NSMutableDictionary};

        let graph = unsafe { mpsg::MPSGraph::new() };

        // Shapes
        let in_shape = dims_to_nsnumbers(&[canonical_heads, seq, head_dim])?;
        let expanded_shape = dims_to_nsnumbers(&[canonical_heads, 1, seq, head_dim])?;
        let broadcast_shape = dims_to_nsnumbers(&[canonical_heads, group_size, seq, head_dim])?;
        let out_shape = dims_to_nsnumbers(&[canonical_heads * group_size, seq, head_dim])?;

        // Placeholders
        let input_ph = unsafe { graph.placeholderWithShape_dataType_name(Some(&*in_shape), data_type, Some(&*ns("input"))) };

        // input [C,S,D] -> reshape to [C,1,S,D]
        let reshaped = unsafe { graph.reshapeTensor_withShape_name(&input_ph, &expanded_shape, Some(&*ns("expand"))) };
        // broadcast to [C,G,S,D]
        let broadcasted = unsafe { graph.broadcastTensor_toShape_name(&reshaped, &broadcast_shape, Some(&*ns("broadcast"))) };
        // reshape to [C*G,S,D]
        let output = unsafe { graph.reshapeTensor_withShape_name(&broadcasted, &out_shape, Some(&*ns("collapse"))) };

        // Build feed types and targets
        let feed_types: Retained<NSMutableDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> = NSMutableDictionary::dictionary();
        let in_type =
            unsafe { mpsg::MPSGraphShapedType::initWithShape_dataType(mpsg::MPSGraphShapedType::alloc(), Some(&*in_shape), data_type) };
        let in_key: &ProtocolObject<dyn objc2_foundation::NSCopying> = ProtocolObject::from_ref(&*input_ph);
        unsafe { feed_types.setObject_forKey(&in_type, in_key) };

        let target_list: Retained<NSMutableArray<mpsg::MPSGraphTensor>> = NSMutableArray::array();
        target_list.addObject(&*output);
        let target_array: Retained<NSArray<mpsg::MPSGraphTensor>> = unsafe { Retained::cast_unchecked(target_list) };
        let feed_types_dict: Retained<objc2_foundation::NSDictionary<mpsg::MPSGraphTensor, mpsg::MPSGraphShapedType>> =
            unsafe { Retained::cast_unchecked(feed_types) };

        let exec = unsafe {
            graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
                None,
                &feed_types_dict,
                &target_array,
                None,
                None,
            )
        };
        Ok(exec)
    }
}

impl DefaultKernelInvocable for RepeatKvHeadsGraphOp {
    type Args<'a, T: TensorElement> = (Tensor<T>, Tensor<T>, u32, u32, u32, u32, u32, u32, u32);

    fn function_id() -> Option<super::KernelFunction> {
        None
    }

    fn new<'a, T: TensorElement>(
        _ctx: &mut Context<T>,
        args: Self::Args<'a, T>,
        _pipeline: Option<Retained<ProtocolObject<dyn objc2_metal::MTLComputePipelineState>>>,
        _cache: Option<&mut ResourceCache>,
    ) -> Result<(Box<dyn Operation>, Tensor<T>), MetalError> {
        let (input, output, group_size, batch, n_kv_heads, n_heads, seq, head_dim, _cache_stride) = args;

        let group_size = group_size as usize;
        let batch = batch as usize;
        let n_kv_heads = n_kv_heads as usize;
        let n_heads = n_heads as usize;
        let seq = seq as usize;
        let head_dim = head_dim as usize;
        let canonical_heads = batch * n_kv_heads;
        let repeated_heads = batch * n_heads;
        if group_size == 0 || repeated_heads != canonical_heads * group_size {
            return Err(MetalError::InvalidShape("repeat_kv_heads_graph invalid group sizing".into()));
        }
        // Expect input [canonical_heads, seq, head_dim]
        let in_dims = input.dims();
        if in_dims != [canonical_heads, seq, head_dim] {
            return Err(MetalError::InvalidShape("repeat_kv_heads_graph input dims mismatch".into()));
        }
        let data_type = crate::cacheable_resources::mps_data_type_for_dtype(T::DTYPE);
        let executable = Self::build_graph(data_type, canonical_heads, group_size, seq, head_dim)?;

        Ok((
            Box::new(RepeatKvHeadsGraph {
                executable,
                input,
                output: output.clone(),
                data_type,
            }),
            output,
        ))
    }
}

impl<T: TensorElement> Operation for RepeatKvHeadsGraph<T> {
    fn encode(&self, command_buffer: &CommandBuffer, _cache: &mut ResourceCache) -> Result<(), MetalError> {
        use objc2_metal_performance_shaders::MPSCommandBuffer;

        use crate::operation::EncoderType;

        command_buffer.prepare_encoder_for_operation(EncoderType::MpsGraph)?;

        let input_shape = dims_to_nsnumbers(&self.input.dims)?;
        let spec_in = GraphBindingSpec {
            expected_shape: &input_shape,
            data_type: self.data_type,
        };
        let in_desc = spec_in.try_from_tensor(&self.input)?;
        let input_arr = {
            let builder = GraphTensorDataArrayBuilder::new();
            builder.push(&in_desc)?;
            builder.finish()
        };

        let output_shape = dims_to_nsnumbers(&self.output.dims)?;
        let spec_out = GraphBindingSpec {
            expected_shape: &output_shape,
            data_type: self.data_type,
        };
        let out_desc = spec_out.try_from_tensor(&self.output)?;
        let result_arr = {
            let builder = GraphTensorDataArrayBuilder::new();
            builder.push(&out_desc)?;
            builder.finish()
        };

        let mps_cb = unsafe { MPSCommandBuffer::commandBufferWithCommandBuffer(command_buffer.raw()) };
        unsafe {
            self.executable.encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
                &mps_cb,
                &input_arr,
                Some(&result_arr),
                None,
            );
        }
        Ok(())
    }
}

impl GraphKernel for RepeatKvHeadsGraphOp {
    const OP_NAME: &'static str = "repeat_kv_heads_graph";

    fn dtype_policy() -> GraphKernelDtypePolicy {
        GraphKernelDtypePolicy::new(
            crate::tensor::dtypes::Dtype::F16,
            GraphKernelAccumulator::Explicit(crate::tensor::dtypes::Dtype::F32),
        )
    }
}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

fn dims_to_nsnumbers(dims: &[usize]) -> Result<Retained<NSArray<NSNumber>>, MetalError> {
    let list: Retained<NSMutableArray<NSNumber>> = NSMutableArray::array();
    for &d in dims {
        let n = NSNumber::numberWithUnsignedInteger(d);
        list.addObject(&*n);
    }
    Ok(unsafe { Retained::cast_unchecked(list) })
}
