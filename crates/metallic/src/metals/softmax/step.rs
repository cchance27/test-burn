use serde::{Deserialize, Serialize};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{CompiledStep, FastBindings, Ref, Step, SymbolTable, TensorBindings}
    }, metals::softmax::Softmax, types::TensorArg
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

/// Shared Softmax execution logic.
fn execute_softmax(
    foundry: &mut Foundry,
    input: &TensorArg,
    output: &TensorArg,
    rows_total: u32,
    seq_q: u32,
    seq_k: u32,
    causal: bool,
    query_offset: u32,
) -> Result<(), MetalError> {
    let kernel = Softmax::new(input, output, rows_total, seq_q, seq_k, causal, query_offset);
    foundry.run(&kernel)
}

#[typetag::serde(name = "Softmax")]
impl Step for SoftmaxStep {
    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let input = bindings.resolve(&self.input)?;
        let output = bindings.resolve(&self.output)?;
        execute_softmax(
            foundry,
            &input,
            &output,
            self.rows_total,
            self.seq_q,
            self.seq_k,
            self.causal,
            self.query_offset,
        )
    }

    fn compile(&self, resolver: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        vec![Box::new(CompiledSoftmaxStep {
            input_idx: symbols.get_or_create(resolver.interpolate(self.input.0.clone())),
            output_idx: symbols.get_or_create(resolver.interpolate(self.output.0.clone())),
            rows_total: self.rows_total,
            seq_q: self.seq_q,
            seq_k: self.seq_k,
            causal: self.causal,
            query_offset: self.query_offset,
        })]
    }

    fn name(&self) -> &'static str {
        "Softmax"
    }
}

/// Compiled Step for Softmax.
#[derive(Debug)]
pub struct CompiledSoftmaxStep {
    pub input_idx: usize,
    pub output_idx: usize,
    pub rows_total: u32,
    pub seq_q: u32,
    pub seq_k: u32,
    pub causal: bool,
    pub query_offset: u32,
}

impl CompiledStep for CompiledSoftmaxStep {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, _bindings: &TensorBindings) -> Result<(), MetalError> {
        let input = fast_bindings
            .get(self.input_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Softmax input tensor not found at idx {}", self.input_idx)))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InvalidShape(format!("Softmax output tensor not found at idx {}", self.output_idx)))?;
        execute_softmax(
            foundry,
            input,
            output,
            self.rows_total,
            self.seq_q,
            self.seq_k,
            self.causal,
            self.query_offset,
        )
    }
}
