//! PolicyStage - Adapts MetalPolicy into a prologue Stage.

use std::marker::PhantomData;

use crate::{
    compound::{BufferArg, Stage}, fusion::MetalPolicy
};

/// Adapts a MetalPolicy into a prologue Stage.
#[derive(Clone)]
pub struct PolicyStage<P: MetalPolicy> {
    _policy: PhantomData<P>,
}

impl<P: MetalPolicy + Default> PolicyStage<P> {
    pub fn new() -> Self {
        Self { _policy: PhantomData }
    }
}

impl<P: MetalPolicy + Default> Default for PolicyStage<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: MetalPolicy + Default + 'static> Stage for PolicyStage<P> {
    fn includes(&self) -> Vec<&'static str> {
        vec!["policies/base.metal", P::default().header()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        P::default()
            .buffer_types()
            .iter()
            .enumerate()
            .map(|(i, (name, metal_type))| BufferArg {
                name,
                metal_type,
                buffer_index: i as u32,
            })
            .collect()
    }

    fn struct_defs(&self) -> String {
        P::default().define_loader().unwrap_or_default()
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let policy = P::default();
        let (output_var, dequant_code) = policy.load_and_dequant("data_ptr", "scale_ptr", "offset");
        (output_var.to_string(), format!("    {}", dequant_code))
    }
}
