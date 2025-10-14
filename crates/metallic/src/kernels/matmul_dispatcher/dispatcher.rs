use crate::kernels::matmul_dispatcher::types::*;
use crate::tensor::dtypes::Dtype;

pub fn select_policy(shape: MatShape, _dtype: Dtype, caps: &MatmulCaps, prefs: &Prefs) -> DispatchPlan {
    use super::constants::{smalln_max_n, simd_m_min, simd_n_min};
    let n_bucket = SmallNBucket::from(shape.n);
    match prefs.backend {
        MatmulBackend::Mlx => DispatchPlan::UseMLX(match n_bucket {
            b @ (SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8 | SmallNBucket::N16) if prefs.force_smalln && shape.n <= smalln_max_n() => {
                MatmulVariant::SmallN(n_bucket)
            }
            _ if caps.has_simdgroup_mm && shape.m >= simd_m_min() && shape.n >= simd_n_min() => MatmulVariant::GemmSimd(GemmTile::T64x32xK),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Mps => DispatchPlan::UseMPS(MatmulVariant::GemmGeneric),
        MatmulBackend::LegacyGemv => DispatchPlan::UseLegacyGemv(match n_bucket {
            b @ (SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8 | SmallNBucket::N16) if shape.n <= smalln_max_n() => MatmulVariant::SmallN(b),
            _ if caps.has_simdgroup_mm && shape.m >= simd_m_min() && shape.n >= simd_n_min() => MatmulVariant::GemmSimd(GemmTile::T64x32xK),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Auto => {
            if prefs.force_smalln && matches!(n_bucket, SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8) {
                DispatchPlan::UseLegacyGemv(MatmulVariant::SmallN(n_bucket))
            } else if caps.has_simdgroup_mm && shape.m >= simd_m_min() && shape.n >= simd_n_min() {
                DispatchPlan::UseLegacyGemv(MatmulVariant::GemmSimd(GemmTile::T64x32xK))
            } else {
                DispatchPlan::UseMLX(MatmulVariant::GemmGeneric)
            }
        }
    }
}
