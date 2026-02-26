use std::sync::Arc;

use crate::{
    compound::{CompiledCompoundKernel, Stage}, fusion::MetalPolicy, metals::{
        common::{
            cache::get_or_build_compound_kernel, composition::manual_output_canonical, policy_slots::{tuple_variant_key, tuple_vector_width_elements}
        }, gemv::{GemvStrategy, stages::VectorWidth}, qkv::stages::{MultiWarpReduceStage, MultiWriteOutputStage, ParallelProjectStage, ParallelProjectUniformStage}, rmsnorm::stages::RmsNormComputeStage
    }
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FusedQkvKernelVariant {
    NoNorm,
    RmsNorm,
}

impl FusedQkvKernelVariant {
    fn select(has_norm: bool) -> Self {
        if has_norm { Self::RmsNorm } else { Self::NoNorm }
    }

    fn norm_suffix(self) -> &'static str {
        match self {
            Self::NoNorm => "",
            Self::RmsNorm => "_rmsnorm",
        }
    }
}

#[inline]
fn strategy_variant_key(strategy: GemvStrategy) -> &'static str {
    match strategy {
        GemvStrategy::Auto => "auto",
        GemvStrategy::DecodeLmHead => "decode_lmhead",
        GemvStrategy::Vectorized => "vectorized",
        GemvStrategy::Scalar => "scalar",
        GemvStrategy::Canonical => "canonical",
    }
}

#[inline]
fn vector_width_from_elements(vec_width: u32) -> VectorWidth {
    match vec_width {
        4 => VectorWidth::Vec4,
        8 => VectorWidth::Vec8,
        _ => panic!("Unsupported vector width: {}", vec_width),
    }
}

fn compile_fused_qkv_kernel(
    kernel_name: &str,
    vec_width: u32,
    qkv_variant: FusedQkvKernelVariant,
    main_stage: Box<dyn Stage>,
) -> CompiledCompoundKernel {
    let mut compound = manual_output_canonical(kernel_name, 8, vec_width);
    if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
        compound = compound.prologue(RmsNormComputeStage::new(6, 7, 21));
    }

    compound
        .main_dyn(main_stage)
        .epilogue(MultiWarpReduceStage)
        .epilogue(MultiWriteOutputStage::new())
        .compile()
}

pub(super) fn get_fused_qkv_kernel(
    strategy: GemvStrategy,
    policy_q: Arc<dyn MetalPolicy>,
    policy_k: Arc<dyn MetalPolicy>,
    policy_v: Arc<dyn MetalPolicy>,
    has_norm: bool,
) -> Arc<CompiledCompoundKernel> {
    let qkv_variant = FusedQkvKernelVariant::select(has_norm);
    let uniform_policy = policy_q.short_name() == policy_k.short_name() && policy_q.short_name() == policy_v.short_name();
    if uniform_policy {
        let variant = format!(
            "{}_{}_{}_uniform",
            strategy_variant_key(strategy),
            policy_q.short_name(),
            if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
                "rmsnorm"
            } else {
                "plain"
            }
        );
        let policy = policy_q.clone();
        return get_or_build_compound_kernel("fused_qkv", variant, move || {
            let kernel_name = format!("fused_qkv{}_{}", qkv_variant.norm_suffix(), policy.short_name());
            let vec_width = tuple_vector_width_elements(&[("q", policy.as_ref())]).expect("validated policy vector width");
            let vw = vector_width_from_elements(vec_width);

            let mut proj = ParallelProjectUniformStage::new(policy.clone()).with_vector_width(vw);
            if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
                proj = proj.with_norm("inv_rms");
            }
            compile_fused_qkv_kernel(&kernel_name, vec_width, qkv_variant, Box::new(proj))
        });
    }

    let policy_tuple = tuple_variant_key(&[("q", policy_q.as_ref()), ("k", policy_k.as_ref()), ("v", policy_v.as_ref())]);
    let variant = format!(
        "{}_{}_{}",
        strategy_variant_key(strategy),
        policy_tuple,
        if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
            "rmsnorm"
        } else {
            "plain"
        }
    );

    let q = policy_q.clone();
    let k = policy_k.clone();
    let v = policy_v.clone();
    get_or_build_compound_kernel("fused_qkv", variant, move || {
        let kernel_name = format!(
            "fused_qkv{}_{}_{}_{}",
            qkv_variant.norm_suffix(),
            q.short_name(),
            k.short_name(),
            v.short_name()
        );
        let vec_width = tuple_vector_width_elements(&[("q", q.as_ref()), ("k", k.as_ref()), ("v", v.as_ref())])
            .expect("validated policy tuple vector width");
        let vw = vector_width_from_elements(vec_width);

        let mut proj = ParallelProjectStage::new(q.clone(), k.clone(), v.clone()).with_vector_width(vw);
        if matches!(qkv_variant, FusedQkvKernelVariant::RmsNorm) {
            proj = proj.with_norm("inv_rms");
        }
        compile_fused_qkv_kernel(&kernel_name, vec_width, qkv_variant, Box::new(proj))
    })
}
