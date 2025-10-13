use crate::kernels::matmul_dispatcher::types::*;
use crate::tensor::dtypes::Dtype;

pub fn select_policy(shape: MatShape, _dtype: Dtype, caps: &MatmulCaps, prefs: &Prefs) -> DispatchPlan {
    let n_bucket = SmallNBucket::from(shape.n);
    match prefs.backend {
        MatmulBackend::Mlx => DispatchPlan::UseMLX(match n_bucket {
            SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8 if prefs.force_smalln => {
                MatmulVariant::SmallN(n_bucket)
            }
            _ if caps.has_simdgroup_mm && shape.m >= 64 && shape.n >= 16 => MatmulVariant::GemmSimd(GemmTile::T64x32xK),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Mps => DispatchPlan::UseMPS(MatmulVariant::GemmGeneric),
        MatmulBackend::Custom => DispatchPlan::UseCustom(match n_bucket {
            SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8 => MatmulVariant::SmallN(n_bucket),
            _ if caps.has_simdgroup_mm && shape.m >= 64 && shape.n >= 16 => MatmulVariant::GemmSimd(GemmTile::T64x32xK),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Auto => {
            if prefs.force_smalln && matches!(n_bucket, SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8) {
                DispatchPlan::UseCustom(MatmulVariant::SmallN(n_bucket))
            } else if caps.has_simdgroup_mm && shape.m >= 64 && shape.n >= 16 {
                DispatchPlan::UseCustom(MatmulVariant::GemmSimd(GemmTile::T64x32xK))
            } else {
                DispatchPlan::UseMLX(MatmulVariant::GemmGeneric)
            }
        }
    }
}
