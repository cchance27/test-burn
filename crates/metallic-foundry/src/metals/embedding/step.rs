use std::sync::Arc;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::{BufferArg, CompiledCompoundKernel, CompoundKernel, Stage}, fusion::MetalPolicy, metals::embedding::{EmbeddingParams, EmbeddingParamsResolved}, spec::{CompiledStep, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg}
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
    pub step: EmbeddingStep,
    pub table_name: String,
    pub table_resolved: ResolvedSymbols,
    pub indices_idx: usize,
    pub output_idx: usize,
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
            step.execute(foundry, &fast_bindings, bindings, &symbols)?;
        }
        Ok(())
    }

    fn compile(&self, bindings: &mut TensorBindings, symbols: &mut SymbolTable) -> Vec<Box<dyn CompiledStep>> {
        let table_name = bindings.interpolate(self.table.0.clone());
        let table_idx = symbols.get_or_create(table_name.clone());
        // Pre-resolve scales eagerly during compilation
        let scales_idx = symbols.get_or_create(format!("{table_name}_scales"));

        let indices_idx = symbols.get_or_create(bindings.interpolate(self.indices.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));

        vec![Box::new(CompiledEmbeddingStep {
            step: self.clone(),
            table_name,
            table_resolved: ResolvedSymbols {
                weights: table_idx,
                scales: Some(scales_idx),
                bias: None,
            },
            indices_idx,
            output_idx,
        })]
    }
}

impl CompiledStep for CompiledEmbeddingStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let table = fast_bindings
            .get(self.table_resolved.weights)
            .ok_or_else(|| MetalError::InputNotFound("embedding table".into()))?;
        let indices = fast_bindings
            .get(self.indices_idx)
            .ok_or_else(|| MetalError::InputNotFound("embedding indices".into()))?;
        let output = fast_bindings
            .get(self.output_idx)
            .ok_or_else(|| MetalError::InputNotFound("embedding output".into()))?;

        let params = EmbeddingParamsResolved {
            d_model: self.step.params.d_model.resolve(bindings),
            total_elements: self.step.params.total_elements.resolve(bindings),
            vocab_size: self.step.params.vocab_size.resolve(bindings),
        };

        let policy = crate::policy::resolve_policy(table.dtype.into());
        let loader = policy.loader_stage();
        let table_args = loader.bind(fast_bindings, &self.table_resolved);

        let args = EmbeddingGenericArgs {
            table: table_args[0].clone(),
            scale_bytes: if table_args.len() > 1 {
                table_args[1].clone()
            } else {
                table_args[0].clone() // Dummy for F16
            },
            indices: TensorArg::from_tensor(indices),
            output: TensorArg::from_tensor(output),
            params,
        };

        let kernel = get_embedding_kernel(policy);
        let dispatch = DispatchConfig {
            grid: GridSize::d1(params.total_elements as usize),
            group: crate::types::ThreadgroupSize::d1(256),
        };

        foundry.run(&kernel.bind_arc(args, dispatch))?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Embedding"
    }
}

#[derive(Debug, Clone, KernelArgs)]
pub struct EmbeddingGenericArgs {
    pub table: TensorArg,
    pub scale_bytes: TensorArg,
    pub indices: TensorArg,
    pub output: TensorArg,
    pub params: EmbeddingParamsResolved,
}

#[derive(Debug, Clone, KernelArgs)]
pub struct EmbeddingStage {
    pub params: EmbeddingParamsResolved,
    pub policy: Arc<dyn MetalPolicy>,
}

impl Stage for EmbeddingStage {
    fn includes(&self) -> Vec<&'static str> {
        // Policy header MUST come before embedding.metal since the template uses Policy types
        vec![self.policy.header(), "embedding/embedding.metal"]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "table",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "scale_bytes",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            BufferArg {
                name: "indices",
                metal_type: "const device uint*",
                buffer_index: 2,
            },
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 3,
            },
            BufferArg {
                name: "params",
                metal_type: "constant EmbeddingParams*",
                buffer_index: 4,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let code = format!(
            r#"
    run_embedding_core<{policy}>(table, scale_bytes, indices, output, params, gid.x);
"#,
            policy = self.policy.struct_name()
        );
        ("output".to_string(), code)
    }

    fn struct_defs(&self) -> String {
        EmbeddingParams::METAL_STRUCT_DEF.to_string()
    }
}

pub fn get_embedding_kernel(policy: Arc<dyn MetalPolicy>) -> Arc<CompiledCompoundKernel> {
    use crate::kernel_registry::{KernelCacheKey, kernel_registry};

    let variant = policy.short_name().to_string();
    let key = KernelCacheKey::new("embedding", variant);

    kernel_registry().get_or_build(key, || {
        let dummy_params = EmbeddingParamsResolved {
            d_model: 0,
            total_elements: 0,
            vocab_size: 0,
        };
        let stage = EmbeddingStage {
            params: dummy_params,
            policy: policy.clone(),
        };

        let kernel_name = format!("embedding_standalone_{}", policy.short_name());
        CompoundKernel::new(&kernel_name).main(stage).with_manual_output(true).compile()
    })
}
