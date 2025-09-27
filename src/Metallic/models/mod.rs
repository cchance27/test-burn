use super::{CommandBuffer, MetalError, Operation, Tensor, resource_cache::ResourceCache};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLCommandQueue;

pub mod qwen25;
pub use qwen25::{Qwen25, Qwen25Config};

mod tests;

/// Trait for Metallic models that can be instantiated from a GGUF-loaded model.
/// Implement this trait for any model that should support loading via GGUF.
pub trait LoadableModel: Sized {
    fn load_from_gguf(gguf_model: &crate::gguf::model_loader::GGUFModel, ctx: &mut crate::metallic::Context) -> Result<Self, MetalError>;
}

/// Generic loader helper: load a concrete model type that implements LoadableModel.
/// Example usage: let qwen: Qwen25 = Model::load(&gguf_model, &mut ctx)?;
pub fn load<T: LoadableModel>(
    gguf_model: &crate::gguf::model_loader::GGUFModel,
    ctx: &mut crate::metallic::Context,
) -> Result<T, MetalError> {
    T::load_from_gguf(gguf_model, ctx)
}
