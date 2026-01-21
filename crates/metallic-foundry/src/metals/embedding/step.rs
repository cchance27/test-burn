use std::sync::{Arc, OnceLock};

use metallic_macros::KernelArgs;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    Foundry, MetalError, compound::{BufferArg, CompiledCompoundKernel, CompoundKernel, Stage, stages::Quantization}, metals::embedding::{EmbeddingParams, EmbeddingParamsResolved}, spec::{CompiledStep, FastBindings, Ref, ResolvedSymbols, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg}
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
    pub pipeline: Arc<OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
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
            pipeline: Arc::new(OnceLock::new()),
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
        let quantization = loader.quantization_type();

        let table_args = loader.bind(fast_bindings, &self.table_resolved);

        let args = EmbeddingGenericArgs {
            table: table_args[0].clone(),
            scale_bytes: table_args[1].clone(),
            indices: TensorArg::from_tensor(indices),
            output: TensorArg::from_tensor(output),
            params,
        };

        let kernel = get_embedding_kernel(quantization);
        let dispatch = DispatchConfig {
            grid: GridSize::d1(params.total_elements as usize),
            group: crate::types::ThreadgroupSize::d1(256),
        };

        foundry.run(&kernel.bind(args, dispatch))?;
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
    pub quant: Quantization,
}

impl Stage for EmbeddingStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["embedding/embedding.metal"]
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
                metal_type: "constant EmbeddingParamsResolved*",
                buffer_index: 4,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        (
            "output".to_string(),
            match self.quant {
                Quantization::F16 => r#"
    run_embedding_core_f16(table, indices, output, params, gid.x);
"#
                .to_string(),
                Quantization::Q8 => r#"
    run_embedding_core_q8(table, scale_bytes, indices, output, params, gid.x);
"#
                .to_string(),
            },
        )
    }

    fn struct_defs(&self) -> String {
        r#"
struct EmbeddingParamsResolved {
    uint d_model;
    uint total_elements;
    uint vocab_size;
};
"#
        .to_string()
    }
}

fn get_embedding_kernel(quant: Quantization) -> &'static CompiledCompoundKernel {
    static KERNELS: OnceLock<std::sync::Mutex<FxHashMap<Quantization, &'static CompiledCompoundKernel>>> = OnceLock::new();
    let cache = KERNELS.get_or_init(|| std::sync::Mutex::new(FxHashMap::default()));
    let mut cache = cache.lock().unwrap();

    if let Some(kernel) = cache.get(&quant) {
        return kernel;
    }

    let dummy_params = EmbeddingParamsResolved {
        d_model: 0,
        total_elements: 0,
        vocab_size: 0,
    };
    let stage = EmbeddingStage {
        params: dummy_params,
        quant,
    };

    let kernel_name = format!("embedding_standalone_{}", quant.short_name());
    let compiled = Box::leak(Box::new(
        CompoundKernel::new(&kernel_name).main(stage).with_manual_output(true).compile(),
    ));

    cache.insert(quant, compiled);
    compiled
}
