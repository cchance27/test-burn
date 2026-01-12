use std::sync::OnceLock;

use metallic_macros::KernelArgs;

use crate::{
    Foundry, MetalError, compound::{BufferArg, CompiledCompoundKernel, CompoundKernel, Stage}, spec::{CompiledStep, FastBindings, SymbolTable, TensorBindings}, types::{DispatchConfig, GridSize, TensorArg, ThreadgroupSize}
};

/// A simple, compiled element-wise add step: Out = A + B.
///
/// Unlike `ElemwiseAdd` (which supports broadcasting params), this step
/// runtime-infers dimensions from the input tensors and performs a direct
/// index-to-index addition. It assumes A, B, and Out have the same total element count.
#[derive(Debug)]
pub struct CompiledSimpleAddStep {
    pub a_idx: usize,
    pub b_idx: usize,
    pub out_idx: usize,
}

impl CompiledStep for CompiledSimpleAddStep {
    fn execute(
        &self,
        foundry: &mut Foundry,
        fast_bindings: &FastBindings,
        _bindings: &TensorBindings,
        _symbols: &SymbolTable,
    ) -> Result<(), MetalError> {
        let a = fast_bindings.get(self.a_idx).ok_or(MetalError::InputNotFound("Add: a".into()))?;
        let b = fast_bindings.get(self.b_idx).ok_or(MetalError::InputNotFound("Add: b".into()))?;
        let out = fast_bindings
            .get(self.out_idx)
            .ok_or(MetalError::InputNotFound("Add: out".into()))?;

        let total = a.dims.iter().product::<usize>() as u32;
        let args = AddArgs {
            a: TensorArg::from_tensor(a),
            b: TensorArg::from_tensor(b),
            out: TensorArg::from_tensor(out),
            total,
        };

        static ADD_KERNEL: OnceLock<CompiledCompoundKernel> = OnceLock::new();
        let kernel = ADD_KERNEL.get_or_init(|| {
            CompoundKernel::new("elemwise_add_f16_simple")
                .prologue(ElemwiseAddGlobalStage)
                .with_manual_output(true) // Stage handles writing
                .compile()
        });

        let grid_size = total.div_ceil(256) as usize;
        let dispatch = DispatchConfig {
            grid: GridSize::d1(grid_size),
            group: ThreadgroupSize::d1(256),
        };

        foundry.run(&kernel.bind(args, dispatch))?;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "SimpleElemwiseAdd"
    }
}

#[derive(Debug, KernelArgs)]
struct AddArgs {
    #[arg(buffer = 0)]
    a: TensorArg,
    #[arg(buffer = 1)]
    b: TensorArg,
    #[arg(buffer = 2)]
    out: TensorArg,
    #[arg(buffer = 3)]
    total: u32,
}

#[derive(Debug, Clone, KernelArgs)]
struct ElemwiseAddGlobalStage;

impl Stage for ElemwiseAddGlobalStage {
    fn includes(&self) -> Vec<&'static str> {
        vec![]
    }
    fn buffer_args(&self) -> Vec<BufferArg> {
        vec![
            BufferArg {
                name: "a",
                metal_type: "const device half*",
                buffer_index: 0,
            },
            BufferArg {
                name: "b",
                metal_type: "const device half*",
                buffer_index: 1,
            },
            BufferArg {
                name: "out",
                metal_type: "device half*",
                buffer_index: 2,
            },
            BufferArg {
                name: "total_elements",
                metal_type: "constant uint&",
                buffer_index: 3,
            },
        ]
    }

    fn struct_defs(&self) -> String {
        String::new()
    }

    fn emit(&self, _prev: &str) -> (String, String) {
        (
            "void".to_string(),
            r#"
    // Global linear thread index for a 1D dispatch.
    // Compound kernels use `gid` as threadgroup_position_in_grid and `lid` as thread_position_in_threadgroup.
    uint i = gid.x * 256 + lid.x;
    if (i < total_elements) {
        out[i] = a[i] + b[i];
    }
"#
            .to_string(),
        )
    }
}
