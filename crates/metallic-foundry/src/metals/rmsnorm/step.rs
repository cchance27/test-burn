use std::sync::Arc;

use metallic_macros::{KernelArgs, Stage as DeriveStage};
use serde::{Deserialize, Serialize};

use super::{RmsNormParams, RmsNormParamsResolved};
use crate::{
    Foundry, MetalError, ResolvedSymbols, compound::CompiledCompoundKernel, fusion::MetalPolicy, metals::common::cache::get_or_build_policy_compound_kernel, policy::f16::PolicyF16, spec::{CompiledStep, DynamicValue, FastBindings, Ref, Step, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
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
        let input_scales_idx = symbols.get_or_create(format!("{input_name}_scales"));
        let output_idx = symbols.get_or_create(bindings.interpolate(self.output.0.clone()));
        let gamma_idx = symbols.get_or_create(bindings.interpolate(self.gamma.0.clone()));

        vec![Box::new(CompiledRmsNormStep {
            step: self.clone(),
            input_resolved: ResolvedSymbols {
                weights: input_idx,
                scales: input_scales_idx.into(),
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
        bindings: &TensorBindings,
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
        let epsilon = bindings.get_f32_var("rms_eps").or(self.step.epsilon).unwrap_or(1e-6);

        let policy = crate::policy::resolve_policy(input.dtype);
        let loader = policy.loader_stage();

        let input_args = loader.bind(fast_bindings, &self.input_resolved);

        let params = RmsNormParamsResolved {
            feature_dim,
            total_elements,
            epsilon,
        };

        run_rmsnorm(
            foundry,
            &input_args[0].clone(),
            if input_args.len() > 1 { Some(&input_args[1]) } else { None },
            &TensorArg::from_tensor(output),
            &TensorArg::from_tensor(gamma),
            params,
        )
    }

    fn name(&self) -> &'static str {
        "RmsNorm"
    }
}

// =============================================================================
// Standalone Stage Implementation
// =============================================================================

#[derive(Debug, Clone, KernelArgs, DeriveStage)]
#[stage(
    includes("dtypes/runtime_types.metal", "rmsnorm/rmsnorm.metal"),
    struct_defs = "RmsNormParams",
    policy_field = "policy",
    template_bindings(policy_struct = "self.policy.struct_name()"),
    emit = r#"
    RMSNORM_RUN_CORE_STAGE({policy_struct}, input, output, gamma, params, scale_bytes, gid, lid, tptg);
"#,
    out_var = "output_ptr"
)]
pub struct RmsNormStandaloneStage {
    #[arg(buffer = 0, metal_type = "const device uchar*")]
    pub input: TensorArg,
    #[arg(buffer = 1, metal_type = "const device uchar*")]
    pub scale_bytes: TensorArg,
    #[arg(buffer = 2, output, metal_type = "device OutputStorageT*")]
    pub output: TensorArg,
    #[arg(buffer = 3, metal_type = "const device GammaStorageT*")]
    pub gamma: TensorArg,
    #[arg(buffer = 4, metal_type = "constant RmsNormParams*")]
    pub params: RmsNormParamsResolved,
    #[arg(skip, stage_skip)]
    pub policy: Arc<dyn MetalPolicy>,
}

impl Default for RmsNormStandaloneStage {
    fn default() -> Self {
        Self {
            input: TensorArg::default(),
            scale_bytes: TensorArg::default(),
            output: TensorArg::default(),
            gamma: TensorArg::default(),
            params: RmsNormParamsResolved::default(),
            policy: Arc::new(PolicyF16),
        }
    }
}

fn get_rmsnorm_kernel(policy: Arc<dyn MetalPolicy>) -> Arc<CompiledCompoundKernel> {
    get_or_build_policy_compound_kernel("rmsnorm_standalone", policy, |policy| {
        let stage = RmsNormStandaloneStage {
            policy: policy.clone(),
            ..Default::default()
        };
        crate::metals::common::composition::manual_output(&format!("rmsnorm_standalone_policy_{}", policy.short_name()))
            .main(stage)
            .compile()
    })
}

pub fn run_rmsnorm(
    foundry: &mut Foundry,
    input: &TensorArg,
    scale_bytes: Option<&TensorArg>,
    output: &TensorArg,
    gamma: &TensorArg,
    params: RmsNormParamsResolved,
) -> Result<(), MetalError> {
    let policy = crate::policy::resolve_policy(input.dtype);
    let kernel = get_rmsnorm_kernel(policy.clone());
    let args = RmsNormStandaloneStage {
        input: input.clone(),
        scale_bytes: scale_bytes.cloned().unwrap_or_else(|| input.clone()),
        output: output.clone(),
        gamma: gamma.clone(),
        params,
        policy,
    };

    let rows = if params.feature_dim == 0 {
        0
    } else {
        params.total_elements / params.feature_dim
    };
    let dispatch = DispatchConfig {
        grid: GridSize::d1(rows as usize),
        group: ThreadgroupSize::d1(256),
    };

    foundry.run(&kernel.bind_arc(args, dispatch))
}
