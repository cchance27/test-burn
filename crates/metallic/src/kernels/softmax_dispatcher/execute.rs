
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLCommandBuffer;

use crate::{
    error::MetalError,
    kernels::Operation,
    resource_cache::ResourceCache,
};

use super::types::SoftmaxPolicy;

/// The internal operation struct for the softmax dispatcher.
/// It holds the policy and the actual operation to be executed.
pub struct SoftmaxDispatch {
    _policy: SoftmaxPolicy,
    /// The underlying operation created based on the policy.
    op: Box<dyn Operation>,
}

impl SoftmaxDispatch {
    pub fn new(policy: SoftmaxPolicy, op: Box<dyn Operation>) -> Self {
        Self { _policy: policy, op }
    }
}

impl Operation for SoftmaxDispatch {
    fn encode(
        &self,
        command_buffer: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        cache: &mut ResourceCache,
    ) -> Result<(), MetalError> {
        // TODO: Add with_gpu_scope! with labels for backend, variant, etc.
        self.op.encode(command_buffer, cache)
    }
}
