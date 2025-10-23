use crate::{
    kernels::matmul_dispatcher::{
        constants::{simd_m_min, simd_n_min}, types::*
    }, tensor::dtypes::Dtype
};

pub fn select_policy(shape: MatShape, _dtype: Dtype, caps: &MatmulCaps, prefs: &Prefs) -> DispatchPlan {
    use super::constants::{simd_m_min, simd_n_min, smalln_max_n};
    let n_bucket = SmallNBucket::from(shape.n);

    // Define thresholds for selecting tiled GEMM - use for large matrices where tiling is beneficial
    let use_tiled_gemm = shape.m >= 512 && shape.n >= 512 && shape.k >= 512;

    match prefs.backend {
        MatmulBackend::Mlx => DispatchPlan::UseMLX(match n_bucket {
            _b @ (SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8 | SmallNBucket::N16)
                if prefs.force_smalln && shape.n <= smalln_max_n() =>
            {
                MatmulVariant::SmallN(n_bucket)
            }
            _ if use_tiled_gemm => MatmulVariant::GemmTiled(GemmTile::Generic),
            _ if caps.has_simdgroup_mm && shape.m >= simd_m_min() && shape.n >= simd_n_min() => MatmulVariant::GemmSimd(GemmTile::T64x32xK),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Mps => DispatchPlan::UseMPS(match n_bucket {
            _ if use_tiled_gemm => MatmulVariant::GemmTiled(GemmTile::Generic),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Gemv => DispatchPlan::Gemv(match n_bucket {
            b @ (SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8 | SmallNBucket::N16)
                if shape.n <= smalln_max_n() =>
            {
                MatmulVariant::SmallN(b)
            }
            _ if use_tiled_gemm => MatmulVariant::GemmTiled(GemmTile::Generic),
            _ if caps.has_simdgroup_mm && shape.m >= simd_m_min() && shape.n >= simd_n_min() => MatmulVariant::GemmSimd(GemmTile::T64x32xK),
            _ => MatmulVariant::GemmGeneric,
        }),
        MatmulBackend::Auto => select_auto_plan(shape, caps, prefs, use_tiled_gemm, n_bucket),
    }
}

fn select_auto_plan(shape: MatShape, caps: &MatmulCaps, prefs: &Prefs, use_tiled_gemm: bool, n_bucket: SmallNBucket) -> DispatchPlan {
    if prefs.force_smalln && matches!(n_bucket, SmallNBucket::N1 | SmallNBucket::N2 | SmallNBucket::N4 | SmallNBucket::N8) {
        return DispatchPlan::Gemv(MatmulVariant::SmallN(n_bucket));
    }

    // Benchmark-guided heuristics:
    // - N=1 prefers the MPS path for parity and bandwidth.
    // - N=2 is best served by dedicated GEMV kernels.
    // - N=3..4 lean toward MPS for lower latency.
    // - N=5..8 are faster on the MLX backend.
    match n_bucket {
        SmallNBucket::N1 => {
            return DispatchPlan::UseMPS(MatmulVariant::GemmGeneric);
        }
        SmallNBucket::N2 => {
            return DispatchPlan::Gemv(MatmulVariant::SmallN(SmallNBucket::N2));
        }
        SmallNBucket::N4 => {
            return DispatchPlan::UseMPS(MatmulVariant::GemmGeneric);
        }
        SmallNBucket::N8 => {
            return DispatchPlan::UseMLX(MatmulVariant::GemmGeneric);
        }
        SmallNBucket::N16 => {
            // Fall through to generic GEMM selection logic.
        }
        SmallNBucket::Other => {}
    }

    if use_tiled_gemm {
        return DispatchPlan::Gemv(MatmulVariant::GemmTiled(GemmTile::Generic));
    }
    if caps.has_simdgroup_mm && shape.m >= simd_m_min() && shape.n >= simd_n_min() {
        return DispatchPlan::Gemv(MatmulVariant::GemmSimd(GemmTile::T64x32xK));
    }

    DispatchPlan::UseMLX(MatmulVariant::GemmGeneric)
}
