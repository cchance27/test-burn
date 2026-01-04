use serde::{Deserialize, Serialize};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{Ref, Step, TensorBindings}
    }, metals::softmax::Softmax
};

/// DSL Step for Softmax (conditional dispatch).
#[derive(Debug, Serialize, Deserialize)]
pub struct SoftmaxStep {
    pub input: Ref,
    pub output: Ref,
    pub rows_total: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub causal: bool,
    pub query_offset: u32,
}

#[typetag::serde(name = "Softmax")]
impl Step for SoftmaxStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let input = bindings.resolve(&self.input)?;
        let output = bindings.resolve(&self.output)?;

        let kernel = Softmax::new(
            &input,
            &output,
            self.rows_total,
            self.seq_q,
            self.seq_k,
            self.causal,
            self.query_offset,
        );

        foundry.run(&kernel)
    }

    fn name(&self) -> &'static str {
        "Softmax"
    }
}
