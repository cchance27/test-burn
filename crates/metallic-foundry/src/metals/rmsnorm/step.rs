use std::sync::Arc;

use metallic_macros::KernelArgs;
use serde::{Deserialize, Serialize};

use super::RmsNormParamsResolved;
use crate::{
    Foundry, MetalError, ResolvedSymbols, compound::BufferArg, fusion::MetalPolicy, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::TensorArg
};

/// RMSNorm Step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsNormStep {
    pub input: Ref,
    pub output: Ref,
    pub gamma: Ref,
    /// Optional epsilon override. If unset, falls back to model global `rms_eps`, then 1e-6.
    pub epsilon: Option<f32>,
    pub feature_dim: DynamicValue<u32>,
    pub total_elements: DynamicValue<u32>,
}

#[derive(Debug, Clone)]
pub struct CompiledRmsNormStep {
    pub step: RmsNormStep,
    pub input_resolved: ResolvedSymbols,
    pub output_idx: usize,
    pub gamma_idx: usize,
}

#[typetag::serde(name = "RmsNormV2")]
impl Step for RmsNormStep {
    fn name(&self) -> &'static str {
        "RmsNorm"
    }

    fn execute(&self, foundry: &mut Foundry, bindings: &mut TensorBindings) -> Result<(), MetalError> {
        let mut symbols = SymbolTable::new();
        let compiled = self.compile(bindings, &mut symbols);
        let mut fast_bindings = FastBindings::new(symbols.len());

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
            input_resolved: ResolvedSymbols {
                weights: input_idx,
                scales: _input_scales_idx.into(),
                bias: None,
            },
            output_idx,
            gamma_idx,
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

        // Resolve epsilon:
        // 1. Model metadata (rms_eps global)
        // 2. DSL epsilon field
        // 3. Default 1e-6
        let epsilon = _bindings
            .get_var("rms_eps")
            .and_then(|v| v.parse::<f32>().ok())
            .or(self.step.epsilon)
            .unwrap_or(1e-6);

        let policy = crate::policy::resolve_policy(input.dtype.into());
        let loader = policy.loader_stage();

        let input_args = loader.bind(fast_bindings, &self.input_resolved);

        let params = RmsNormParamsResolved {
            feature_dim,
            total_elements,
            epsilon,
        };

        let args = super::RmsNorm::new(
            &input_args[0].clone(),
            if input_args.len() > 1 { Some(input_args[1].clone()) } else { None },
            &TensorArg::from_tensor(output),
            &TensorArg::from_tensor(gamma),
            params,
        );

        foundry.run(&args)
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
    pub policy: Arc<dyn MetalPolicy>,
}

impl crate::compound::Stage for RmsNormStandaloneStage {
    fn includes(&self) -> Vec<&'static str> {
        vec!["rmsnorm/rmsnorm.metal", self.policy.header()]
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
        let policy_name = self.policy.struct_name();
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
    float epsilon;
};
"#
        .to_string()
    }
}
