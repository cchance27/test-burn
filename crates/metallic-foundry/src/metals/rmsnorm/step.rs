use std::sync::{Arc, OnceLock};

use metallic_macros::KernelArgs;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLComputePipelineState;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::{RmsNorm, RmsNormParamsResolved};
use crate::{
    Foundry, MetalError, compound::{BufferArg, CompiledCompoundKernel, CompoundKernel, stages::Quantization}, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// RMSNorm Step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNormStep {
    pub input: Ref,
    pub output: Ref,
    pub gamma: Ref,
    pub epsilon: Option<f32>, // Not used in kernel yet (hardcoded 1e-6)
    pub feature_dim: DynamicValue<u32>,
    pub total_elements: DynamicValue<u32>,
}

#[derive(Debug, Clone)]
pub struct CompiledRmsNormStep {
    pub step: RmsNormStep,
    pub input_resolved: crate::spec::ResolvedSymbols,
    pub output_idx: usize,
    pub gamma_idx: usize,
    pub pipeline: Arc<OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>>,
}

#[typetag::serde(name = "RmsNormV2")]
impl Step for RmsNormStep {
    fn name(&self) -> &'static str {
        "RmsNorm"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = crate::spec::SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = crate::spec::FastBindings::new(symbols.len());

        // Bind all symbols found in the table
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
        let input_name = bindings.interpolate(self.input.0.clone());
        let input_idx = symbols.get_or_create(input_name.clone());
        let _input_scales_idx = symbols.get_or_create(format!("{input_name}_scales"));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));

        vec![Box::new(CompiledRmsNormStep {
            step: self.clone(),
            input_resolved: crate::spec::ResolvedSymbols {
                weights: input_idx,
                scales: _input_scales_idx.into(),
                bias: None,
            },
            output_idx,
            gamma_idx,
            pipeline: Arc::new(OnceLock::new()),
        })]
    }
}

impl CompiledStep for CompiledRmsNormStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        _bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let input = fast_bindings
            .get(self.input_resolved.weights)
            .ok_or(MetalError::InputNotFound("input".into()))?;

        let output = fast_bindings
            .get(self.output_idx)
            .ok_or(MetalError::InputNotFound("output".into()))?;
        let gamma = fast_bindings.get(self.gamma_idx).ok_or(MetalError::InputNotFound("gamma".into()))?;

        // Resolve dynamic params
        let feature_dim = match self.step.feature_dim {
            DynamicValue::Literal(v) => v,
            _ => input.dims.last().copied().unwrap_or(0) as u32,
        };
        let total_elements = match self.step.total_elements {
            DynamicValue::Literal(v) if v > 0 => v,
            _ => input.dims.iter().product::<usize>() as u32,
        };

        let policy = crate::policy::resolve_policy(input.dtype.into());
        let loader = policy.loader_stage();
        let quantization = loader.quantization_type();

        let input_args = loader.bind(fast_bindings, &self.input_resolved);

        let params = RmsNormParamsResolved {
            feature_dim,
            total_elements,
        };

        let args = RmsNorm {
            input: input_args[0].clone(),
            scale_bytes: input_args[1].clone(),
            output: TensorArg::from_tensor(output),
            gamma: TensorArg::from_tensor(gamma),
            params,
        };

        // Dispatch: per row (warp)
        let n_rows = total_elements / feature_dim;
        let dispatch = DispatchConfig {
            grid: GridSize::d1(n_rows as usize),
            group: ThreadgroupSize::d1(256), // matches THREADS_PER_ROW constant in kernel
        };

        let kernel = get_rmsnorm_kernel(quantization);

        let pipeline = if let Some(p) = self.pipeline.get() {
            p
        } else {
            let p = foundry.load_kernel(kernel)?;
            let _ = self.pipeline.set(p);
            self.pipeline.get().unwrap()
        };

        foundry.dispatch_pipeline(pipeline, &kernel.bind(args, dispatch), dispatch)
    }

    fn name(&self) -> &'static str {
        "RmsNorm"
    }
}

// =============================================================================
// Standalone Stage Implementation
// =============================================================================

#[derive(Debug, Clone, KernelArgs)]
pub struct RmsNormStandaloneStage {
    pub params: RmsNormParamsResolved,
    pub quant: Quantization,
}

impl crate::compound::Stage for RmsNormStandaloneStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["rmsnorm/rmsnorm.metal", self.quant.include_path()]
    }

    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "input",
                metal_type: "const device uchar*",
                buffer_index: 0,
            },
            BufferArg {
                name: "scale_bytes",
                metal_type: "const device uchar*",
                buffer_index: 1,
            },
            BufferArg {
                name: "output",
                metal_type: "device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "gamma",
                metal_type: "const device half*",
                buffer_index: 3,
            },
            BufferArg {
                name: "params",
                metal_type: "constant RmsNormParams*",
                buffer_index: 4,
            },
        ]
    }

    fn emit(&self, _input_var: &str) -> (String, String) {
        let policy_name = self.quant.policy_name();
        (
            "output_ptr".to_string(),
            format!(
                r#"
    // RmsNormStandaloneStage
    threadgroup float tg_inv_rms;
    
    run_rmsnorm_core<{policy_name}>(
        input,
        output,
        gamma,
        params,
        scale_bytes,
        gid,
        lid,
        &tg_inv_rms
    );
            "#
            ),
        )
    }

    fn struct_defs(&self) -> String {
        r#"
struct RmsNormParams {
    uint feature_dim;
    uint total_elements;
};
"#
        .to_string()
    }
}

fn get_rmsnorm_kernel(quant: Quantization) -> &'static CompiledCompoundKernel {
    static KERNELS: OnceLock<std::sync::Mutex<FxHashMap<Quantization, &'static CompiledCompoundKernel>>> = OnceLock::new();
    let cache = KERNELS.get_or_init(|| std::sync::Mutex::new(FxHashMap::default()));
    let mut cache = cache.lock().unwrap();

    if let Some(kernel) = cache.get(&quant) {
        return kernel;
    }

    let dummy_params = RmsNormParamsResolved {
        feature_dim: 0,
        total_elements: 0,
    };
    let stage = RmsNormStandaloneStage {
        params: dummy_params,
        quant,
    };

    let kernel_name = format!("rmsnorm_standalone_{}", quant.short_name());
    let compiled = Box::leak(Box::new(
        CompoundKernel::new(&kernel_name).main(stage).with_manual_output(true).compile(),
    ));

    cache.insert(quant, compiled);
    compiled
}
