use super::types::SoftmaxPolicy;
use crate::{CommandBuffer, caching::ResourceCache, context::GpuProfilerLabel, error::MetalError, kernels::Operation};

/// The internal operation struct for the softmax dispatcher.
/// It holds the policy and the actual operation to be executed.
pub struct SoftmaxDispatch {
    _policy: SoftmaxPolicy,
    /// The underlying operation created based on the policy.
    op: Box<dyn Operation>,
    _profiler_label: GpuProfilerLabel,
}

impl SoftmaxDispatch {
    pub fn new_with_label(policy: SoftmaxPolicy, op: Box<dyn Operation>, profiler_label: GpuProfilerLabel) -> Self {
        Self {
            _policy: policy,
            op,
            _profiler_label: profiler_label,
        }
    }
}

impl Operation for SoftmaxDispatch {
    fn encode(&self, command_buffer: &CommandBuffer, cache: &mut ResourceCache) -> Result<(), MetalError> {
        // Simply execute the underlying operation without creating an encoder here
        // The underlying operation will handle its own encoder and profiling
        self.op.encode(command_buffer, cache)
    }
}
