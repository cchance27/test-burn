use serde::{Deserialize, Serialize};

use crate::{
    MetalError, foundry::{
        Foundry, spec::{CompiledStep, FastBindings, Ref, Step, SymbolTable, TensorBindings}
    }, metals::embedding::{Embedding, EmbeddingParams, EmbeddingParamsResolved, EmbeddingQ8}, tensor::Dtype, types::TensorArg
};

/// Manual Embedding Step with runtime dtype dispatch (F16 vs Q8_0).
///
/// This keeps a single model spec (`models/qwen25.json`) compatible across
/// FP16 and Q8 GGUF weights by selecting the kernel based on the bound table dtype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStep {
    pub table: Ref,
    pub indices: Ref,
    pub output: Ref,
    pub params: EmbeddingParams,
}

#[derive(Debug, Clone)]
pub struct CompiledEmbeddingStep {
    pub table_idx: usize,
    pub derived_scale_idx: usize,
    pub indices_idx: usize,
    pub output_idx: usize,
    pub params: EmbeddingParams,
}

#[typetag::serde(name = "Embedding")]
impl Step for EmbeddingStep {
    fn name(&self) -> &'static str {
        "Embedding"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

        for (name, symbol_id) in symbols.iter() {
            if let Ok(tensor) = bindings.get(name) {
                fast_bindings.set(*symbol_id, tensor);
            }
        }

        for step in compiled {
            step.execute(foundry, &fast_bindings, bindings)?;
        }
        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let table_name = bindings.interpolate(self.table.0.clone());
        let table_idx = symbols.get_or_create(table_name.clone());
        let derived_scale_idx = symbols.get_or_create(format!("{table_name}_scales"));
        let indices_idx = symbols.get_or_create(bindings.interpolate(self.indices.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledEmbeddingStep {
            table_idx,
            derived_scale_idx,
            indices_idx,
            output_idx,
            params: self.params.clone(),
        })]
    }
}

impl CompiledStep for CompiledEmbeddingStep {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, bindings: &TensorBindings) -> Result<(), MetalError> {
        let table = fast_bindings
            .get(self.table_idx)
            .ok_or_else(|| MetalError::InputNotFound("embedding table".into()))?;
        let indices = fast_bindings
            .get(self.indices_idx)
            .ok_or_else(|| MetalError::InputNotFound("embedding indices".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InputNotFound("embedding output".into()))?;

        let params = EmbeddingParamsResolved {
            d_model: self.params.d_model.resolve(bindings),
            total_elements: self.params.total_elements.resolve(bindings),
            vocab_size: self.params.vocab_size.resolve(bindings),
        };

        if table.dtype == Dtype::U8 {
            let scale_bytes = fast_bindings
                .get(self.derived_scale_idx)
                .ok_or_else(|| MetalError::InputNotFound("embedding scale_bytes".into()))?;
            if scale_bytes.dtype != Dtype::U8 {
                return Err(MetalError::InvalidShape(format!(
                    "embedding scale_bytes must be U8 for Q8 table (got {:?})",
                    scale_bytes.dtype
                )));
            }

            let kernel = EmbeddingQ8 {
                table: TensorArg::from_tensor(table),
                scale_bytes: TensorArg::from_tensor(scale_bytes),
                indices: TensorArg::from_tensor(indices),
                output: TensorArg::from_tensor(output),
                params,
            };
            foundry.run(&kernel)
        } else {
            let kernel = Embedding {
                table: TensorArg::from_tensor(table),
                indices: TensorArg::from_tensor(indices),
                output: TensorArg::from_tensor(output),
                params,
            };
            foundry.run(&kernel)
        }
    }

    fn name(&self) -> &'static str {
        "Embedding"
    }
}
