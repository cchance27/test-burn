use crate::metallic::{Dtype, MetalError};

/// Describes a Metal kernel library that can be compiled at runtime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelDescriptor {
    pub id: &'static str,
    pub source: &'static str,
}

impl KernelDescriptor {
    /// Returns the unique identifier associated with this library descriptor.
    pub const fn id(&self) -> &'static str {
        self.id
    }

    /// Returns the Metal source for this library descriptor.
    pub const fn source(&self) -> &'static str {
        self.source
    }
}

/// Describes a function exported by a Metal kernel library.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelFunctionDescriptor {
    pub id: &'static str,
    pub library: &'static KernelDescriptor,
    pub name_for_dtype: fn(Dtype) -> Option<&'static str>,
}

impl KernelFunctionDescriptor {
    /// Returns the unique identifier for this function descriptor.
    pub const fn id(&self) -> &'static str {
        self.id
    }

    /// Returns the library descriptor associated with this function.
    pub const fn library(&self) -> &'static KernelDescriptor {
        self.library
    }

    /// Resolves the Metal function name for the provided [`Dtype`].
    pub fn resolve_name(&self, dtype: Dtype) -> Result<&'static str, MetalError> {
        (self.name_for_dtype)(dtype).ok_or(MetalError::UnsupportedDtype { operation: self.id, dtype })
    }
}
