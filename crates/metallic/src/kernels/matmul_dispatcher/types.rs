use crate::tensor::dtypes::Dtype;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatmulBackend {
    Auto,
    Mps,
    Mlx,
    Custom,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SmallNBucket {
    N1,
    N2,
    N4,
    N8,
    N16,
    Other,
}

impl From<usize> for SmallNBucket {
    fn from(n: usize) -> Self {
        match n {
            1 => SmallNBucket::N1,
            2 => SmallNBucket::N2,
            3 | 4 => SmallNBucket::N4,
            5 | 6 | 7 | 8 => SmallNBucket::N8,
            9..=16 => SmallNBucket::N16,
            _ => SmallNBucket::Other,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GemmTile {
    T64x32xK,
    T64x64xK,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MatmulVariant {
    SmallN(SmallNBucket),
    GemmSimd(GemmTile),
    GemmGeneric,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulCaps {
    pub has_simdgroup_mm: bool,
    pub max_tg_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatShape {
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DispatchPlan {
    UseMLX(MatmulVariant),
    UseMPS(MatmulVariant),
    UseCustom(MatmulVariant),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulPolicy {
    pub backend: MatmulBackend,
    pub variant: MatmulVariant,
    pub dtype: Dtype,
}
