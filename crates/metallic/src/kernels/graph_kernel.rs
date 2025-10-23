use crate::{cacheable_resources::mps_data_type_for_dtype, error::MetalError, tensor::dtypes::Dtype};

/// Describes how a graph-backed kernel expects its storage and accumulator
/// precision to be configured. The storage dtype applies to the raw tensors
/// bound into the graph, while the accumulator dtype captures any widened
/// precision the graph requires internally (for example fp16 inputs with
/// fp32 accumulators).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphKernelDtypePolicy {
    storage: Dtype,
    accumulator: GraphKernelAccumulator,
}

impl GraphKernelDtypePolicy {
    /// Construct a policy with explicit storage and optional accumulator dtype.
    pub const fn new(storage: Dtype, accumulator: GraphKernelAccumulator) -> Self {
        Self { storage, accumulator }
    }

    /// Convenience constructor for kernels that do not widen precision.
    pub const fn storage_only(storage: Dtype) -> Self {
        Self {
            storage,
            accumulator: GraphKernelAccumulator::None,
        }
    }

    /// The dtype that bound tensors must use.
    pub const fn storage(&self) -> Dtype {
        self.storage
    }

    /// Returns the accumulator dtype when widening is required.
    pub const fn accumulator(&self) -> Option<Dtype> {
        match self.accumulator {
            GraphKernelAccumulator::None => None,
            GraphKernelAccumulator::Explicit(dtype) => Some(dtype),
        }
    }

    /// Convert the storage dtype to an `MPSDataType`.
    pub fn storage_mps_type(&self) -> objc2_metal_performance_shaders::MPSDataType {
        mps_data_type_for_dtype(self.storage)
    }

    /// Convert the accumulator dtype (when present) to an `MPSDataType`.
    pub fn accumulator_mps_type(&self) -> Option<objc2_metal_performance_shaders::MPSDataType> {
        self.accumulator().map(mps_data_type_for_dtype)
    }

    /// Validate that the provided storage dtype matches the policy.
    pub fn validate_storage(&self, dtype: Dtype, op_name: &'static str) -> Result<(), MetalError> {
        if dtype != self.storage {
            return Err(MetalError::UnsupportedDtype { operation: op_name, dtype });
        }
        Ok(())
    }
}

/// Describes whether a graph-backed kernel widens precision internally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphKernelAccumulator {
    None,
    Explicit(Dtype),
}

/// Semantic description for an axis within a tensor descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphKernelAxis {
    /// Batch dimension. Typically dynamic and shared across Q/K/V.
    Batch,
    /// Number of attention heads.
    Heads,
    /// Query sequence length.
    SequenceQ,
    /// Key sequence length.
    SequenceK,
    /// Value sequence length. Frequently identical to `SequenceK`.
    SequenceV,
    /// Feature / channel dimension for attention inputs.
    ModelDim,
    /// Projection output dimension.
    OutputDim,
    /// Static dimension with a compile-time constant extent.
    Static(usize),
    /// Custom, developer-supplied semantic label.
    Custom(&'static str),
}

impl GraphKernelAxis {
    /// Returns a short, human-readable label for diagnostics and telemetry.
    pub const fn label(self) -> &'static str {
        match self {
            GraphKernelAxis::Batch => "batch",
            GraphKernelAxis::Heads => "heads",
            GraphKernelAxis::SequenceQ => "seq_q",
            GraphKernelAxis::SequenceK => "seq_k",
            GraphKernelAxis::SequenceV => "seq_v",
            GraphKernelAxis::ModelDim => "model_dim",
            GraphKernelAxis::OutputDim => "output_dim",
            GraphKernelAxis::Static(_value) => "static",
            GraphKernelAxis::Custom(label) => label,
        }
    }

    /// Extracts the constant size when the axis is statically sized.
    pub const fn static_extent(self) -> Option<usize> {
        match self {
            GraphKernelAxis::Static(value) => Some(value),
            _ => None,
        }
    }
}

/// Describes a single tensor binding that participates in a graph kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphKernelTensorDescriptor {
    /// Stable binding identifier shared across caches and telemetry.
    pub binding: &'static str,
    /// Ordered semantic axis descriptions for the tensor shape.
    pub axes: &'static [GraphKernelAxis],
    /// Optional free-form notes (mask semantics, storage expectations, etc.).
    pub notes: Option<&'static str>,
}

impl GraphKernelTensorDescriptor {
    /// Construct a descriptor with the provided axes and optional notes.
    pub const fn new(binding: &'static str, axes: &'static [GraphKernelAxis], notes: Option<&'static str>) -> Self {
        Self { binding, axes, notes }
    }

    /// Convenience helper for descriptors without additional notes.
    pub const fn without_notes(binding: &'static str, axes: &'static [GraphKernelAxis]) -> Self {
        Self {
            binding,
            axes,
            notes: None,
        }
    }

    /// Returns the tensor rank implied by the axis description.
    pub fn rank(&self) -> usize {
        self.axes.len()
    }
}

/// Captures the full input/output signature for a graph kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphKernelSignature {
    /// Ordered input tensor descriptors.
    pub inputs: &'static [GraphKernelTensorDescriptor],
    /// Ordered output tensor descriptors.
    pub outputs: &'static [GraphKernelTensorDescriptor],
}

impl GraphKernelSignature {
    /// Construct a signature from input/output descriptor slices.
    pub const fn new(inputs: &'static [GraphKernelTensorDescriptor], outputs: &'static [GraphKernelTensorDescriptor]) -> Self {
        Self { inputs, outputs }
    }

    /// Empty signature used when a kernel has not populated descriptor metadata yet.
    pub const fn empty() -> Self {
        Self { inputs: &[], outputs: &[] }
    }

    /// Number of tensors described across inputs and outputs.
    pub fn total_bindings(&self) -> usize {
        self.inputs.len() + self.outputs.len()
    }
}

/// A trait implemented by kernels that can execute through the graph backend.
/// The trait is intentionally lightweight; it primarily exposes dtype policy
/// metadata so dispatchers and cache layers can plan zero-copy bindings and
/// accumulator requirements up front.
pub trait GraphKernel {
    /// Human-readable operation name used for error reporting and telemetry.
    const OP_NAME: &'static str;

    /// Returns the dtype policy for the kernel.
    fn dtype_policy() -> GraphKernelDtypePolicy;

    /// Helper for callers that need to validate storage dtype before binding.
    fn validate_storage_dtype(dtype: Dtype) -> Result<(), MetalError> {
        Self::dtype_policy().validate_storage(dtype, Self::OP_NAME)
    }

    /// Describes the tensor binding contract for the kernel. Kernels can override
    /// this to surface reusable metadata (axis semantics, notes, etc.) for tooling.
    fn signature() -> GraphKernelSignature {
        GraphKernelSignature::empty()
    }
}
