use std::sync::{Arc, OnceLock};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputePipelineState;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::{RmsNorm, RmsNormParamsResolved};
use crate::{
    MetalError, compound::{BufferArg, CompiledCompoundKernel, CompoundKernel}, foundry::{
        Foundry, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}
    }, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
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
    pub input_idx: usize,
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
        let mut symbols = crate::foundry::spec::SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = crate::foundry::spec::FastBindings::new(symbols.len());

        // Bind all symbols found in the table
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
        let input_idx = symbols.get_or_create(bindings.interpolate(self.input.0.clone()));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));

        vec![Box::new(CompiledRmsNormStep {
            step: self.clone(),
            input_idx,
            output_idx,
            gamma_idx,
            pipeline: Arc::new(OnceLock::new()),
        })]
    }
}

impl CompiledStep for CompiledRmsNormStep {
    fn execute(&self, foundry: &mut Foundry, fast_bindings: &FastBindings, _bindings: &TensorBindings) -> Result<(), MetalError> {
        let input = fast_bindings.get(self.input_idx).ok_or(MetalError::InputNotFound("input".into()))?;

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

        let params = RmsNormParamsResolved {
            feature_dim,
            total_elements,
        };

        let args = RmsNorm {
            input: TensorArg::from_tensor(input),
            scale_bytes: TensorArg::from_tensor(input), // F16 uses input as dummy/cast for scale
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

        let kernel = get_rmsnorm_kernel();

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
}

impl crate::compound::Stage for RmsNormStandaloneStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["rmsnorm/rmsnorm.metal"]
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
        (
            "output_ptr".to_string(),
            r#"
    // RmsNormStandaloneStage
    // Using PolicyF16 for now
    
    threadgroup float tg_inv_rms;
    
    run_rmsnorm_core<PolicyF16>(
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
            .to_string(),
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

fn get_rmsnorm_kernel() -> &'static CompiledCompoundKernel {
    static KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
    KERNEL.get_or_init(|| {
        let dummy_params = RmsNormParamsResolved {
            feature_dim: 0,
            total_elements: 0,
        };
        let stage = RmsNormStandaloneStage { params: dummy_params };

        CompoundKernel::new("rmsnorm_standalone")
            .main(stage)
            .with_manual_output(true)
            .compile()
    })
}
