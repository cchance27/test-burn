use crate::{
    kernels::matmul_dispatcher::{dispatcher::select_policy, types::*}, tensor::dtypes::Dtype
};

fn caps(simd: bool) -> MatmulCaps {
    MatmulCaps {
        has_simdgroup_mm: simd,
        max_tg_size: 1024,
    }
}

#[test]
fn auto_smalln_prefers_custom_smalln() {
    let shape = MatShape { m: 512, k: 4096, n: 4 };
    let prefs = Prefs {
        backend: MatmulBackend::Auto,
        force_smalln: true,
    };
    let plan = select_policy(shape, Dtype::F16, &caps(true), &prefs);
    match plan {
        DispatchPlan::Gemv(MatmulVariant::SmallN(SmallNBucket::N4)) => {}
        _ => panic!("unexpected plan: {:?}", plan),
    }
}

#[test]
fn auto_simdgroup_prefers_gemm_simd_when_large() {
    let shape = MatShape { m: 128, k: 4096, n: 64 };
    let prefs = Prefs {
        backend: MatmulBackend::Auto,
        force_smalln: false,
    };
    let plan = select_policy(shape, Dtype::F16, &caps(true), &prefs);
    match plan {
        DispatchPlan::Gemv(MatmulVariant::GemmSimd(_)) => {}
        _ => panic!("unexpected plan: {:?}", plan),
    }
}

#[test]
fn auto_fallbacks_to_mlx_gemm_generic() {
    let shape = MatShape { m: 32, k: 512, n: 32 };
    let prefs = Prefs {
        backend: MatmulBackend::Auto,
        force_smalln: false,
    };
    let plan = select_policy(shape, Dtype::F16, &caps(false), &prefs);
    match plan {
        DispatchPlan::UseMLX(MatmulVariant::GemmGeneric) => {}
        _ => panic!("unexpected plan: {:?}", plan),
    }
}
