use crate::tensor::dtypes::Dtype;
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulBackend {
    Auto,
    Mps,
    Mlx,
    LegacyGemv,
}

impl fmt::Display for MatmulBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatmulBackend::Auto => write!(f, "auto"),
            MatmulBackend::Mps => write!(f, "mps"),
            MatmulBackend::Mlx => write!(f, "mlx"),
            MatmulBackend::LegacyGemv => write!(f, "legacy_gemv"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmallNBucket {
    N1,
    N2,
    N4,
    N8,
    N16,
    Other,
}

impl fmt::Display for SmallNBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmallNBucket::N1 => write!(f, "n1"),
            SmallNBucket::N2 => write!(f, "n2"),
            SmallNBucket::N4 => write!(f, "n4"),
            SmallNBucket::N8 => write!(f, "n8"),
            SmallNBucket::N16 => write!(f, "n16"),
            SmallNBucket::Other => write!(f, "other"),
        }
    }
}

impl From<usize> for SmallNBucket {
    fn from(n: usize) -> Self {
        match n {
            1 => SmallNBucket::N1,
            2 => SmallNBucket::N2,
            3 | 4 => SmallNBucket::N4,
            5..=8 => SmallNBucket::N8,
            9..=16 => SmallNBucket::N16,
            _ => SmallNBucket::Other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GemmTile {
    T64x32xK,
    T64x64xK,
    Generic,
}

impl fmt::Display for GemmTile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmTile::T64x32xK => write!(f, "t64x32xk"),
            GemmTile::T64x64xK => write!(f, "t64x64xk"),
            GemmTile::Generic => write!(f, "generic"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulVariant {
    SmallN(SmallNBucket),
    GemmSimd(GemmTile),
    GemmGeneric,
}

impl fmt::Display for MatmulVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatmulVariant::SmallN(bucket) => write!(f, "smalln_{}", bucket),
            MatmulVariant::GemmSimd(tile) => write!(f, "gemm_simd_{}", tile),
            MatmulVariant::GemmGeneric => write!(f, "gemm_generic"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatmulCaps {
    pub has_simdgroup_mm: bool,
    pub max_tg_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DispatchPlan {
    UseMLX(MatmulVariant),
    UseMPS(MatmulVariant),
    UseLegacyGemv(MatmulVariant),
}

impl fmt::Display for DispatchPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchPlan::UseMLX(variant) => write!(f, "mlx_{}", variant),
            DispatchPlan::UseMPS(variant) => write!(f, "mps_{}", variant),
            DispatchPlan::UseLegacyGemv(variant) => write!(f, "legacy_gemv_{}", variant),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Prefs {
    pub backend: MatmulBackend,
    pub force_smalln: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            backend: MatmulBackend::Auto,
            force_smalln: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatmulPolicy {
    pub backend: MatmulBackend,
    pub variant: MatmulVariant,
    pub dtype: Dtype,
}
