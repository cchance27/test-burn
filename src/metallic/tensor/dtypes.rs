use std::fmt::{Debug, Display};

use half::{bf16, f16};

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U32,
    U8,
}

impl Dtype {
    /// Size in bytes for this data type
    pub fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 => 4,
            Dtype::I64 => 8,
            Dtype::U8 => 1,
        }
    }

    /// Metal format string for this data type
    pub fn metal_format(&self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F16 => "half",
            Dtype::BF16 => "bfloat",
            Dtype::I32 => "int",
            Dtype::I64 => "long",
            Dtype::U32 => "uint",
            Dtype::U8 => "uchar",
        }
    }
}

/// Trait describing how a tensor element is represented on the host.
pub trait TensorElement: Copy + Send + Sync + 'static {
    /// Host scalar type corresponding to the tensor element.
    type Scalar: Copy + Send + Sync + PartialEq + PartialOrd + Display + Debug + 'static;

    /// [`Dtype`] tag for this tensor element.
    const DTYPE: Dtype;

    /// Convert from an `f32` to the scalar representation.
    fn from_f32(value: f32) -> Self::Scalar;

    /// Convert from the scalar representation to `f32`.
    fn to_f32(value: Self::Scalar) -> f32;

    /// Convert a slice of `f32` values into the scalar representation.
    fn from_f32_slice(data: &[f32]) -> Vec<Self::Scalar> {
        data.iter().copied().map(Self::from_f32).collect()
    }

    /// Convert a slice of scalars into a `Vec<f32>`.
    fn to_f32_vec(data: &[Self::Scalar]) -> Vec<f32> {
        data.iter().copied().map(Self::to_f32).collect()
    }

    /// Convert a single scalar into an `f32` and write it to the destination slice.
    fn fill_from_f32(dest: &mut [Self::Scalar], value: f32) {
        let converted = Self::from_f32(value);
        dest.fill(converted);
    }

    /// Copy from an `f32` slice into the scalar destination slice.
    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        debug_assert_eq!(src.len(), dest.len());
        for (dst, value) in dest.iter_mut().zip(src.iter().copied()) {
            *dst = Self::from_f32(value);
        }
    }

    /// Check if the scalar value is finite (not NaN or infinite)
    fn is_finite(value: Self::Scalar) -> bool;

    /// Calculate the absolute value of the scalar
    fn abs(value: Self::Scalar) -> Self::Scalar;

    /// Find the maximum of two values
    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;

    /// Find the minimum of two values
    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar;
}

/// Marker type for `f32` tensors.
#[derive(Clone, Copy, Debug, Default)]
pub struct F32Element;

impl TensorElement for F32Element {
    type Scalar = f32;

    const DTYPE: Dtype = Dtype::F32;

    #[inline]
    fn from_f32(value: f32) -> Self::Scalar {
        value
    }

    #[inline]
    fn to_f32(value: Self::Scalar) -> f32 {
        value
    }

    #[inline]
    fn fill_from_f32(dest: &mut [Self::Scalar], value: f32) {
        dest.fill(value);
    }

    #[inline]
    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        dest.copy_from_slice(src);
    }

    #[inline]
    fn is_finite(value: Self::Scalar) -> bool {
        value.is_finite()
    }

    #[inline]
    fn abs(value: Self::Scalar) -> Self::Scalar {
        value.abs()
    }

    #[inline]
    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a.max(b)
    }

    #[inline]
    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a.min(b)
    }
}

/// Marker type for `f16` tensors.
#[derive(Clone, Copy, Debug, Default)]
pub struct F16Element;

impl TensorElement for F16Element {
    type Scalar = f16;

    const DTYPE: Dtype = Dtype::F16;

    #[inline]
    fn from_f32(value: f32) -> Self::Scalar {
        f16::from_f32(value)
    }

    #[inline]
    fn to_f32(value: Self::Scalar) -> f32 {
        value.to_f32()
    }

    #[inline]
    fn is_finite(value: Self::Scalar) -> bool {
        value.is_finite()
    }

    #[inline]
    fn abs(value: Self::Scalar) -> Self::Scalar {
        use num_traits::float::FloatCore;
        value.abs()
    }

    #[inline]
    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        if a > b { a } else { b }
    }

    #[inline]
    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        if a < b { a } else { b }
    }
}

/// Marker type for `bf16` tensors.
#[derive(Clone, Copy, Debug, Default)]
pub struct BF16Element;

impl TensorElement for BF16Element {
    type Scalar = bf16;

    const DTYPE: Dtype = Dtype::BF16;

    #[inline]
    fn from_f32(value: f32) -> Self::Scalar {
        bf16::from_f32(value)
    }

    #[inline]
    fn to_f32(value: Self::Scalar) -> f32 {
        value.to_f32()
    }

    #[inline]
    fn is_finite(value: Self::Scalar) -> bool {
        value.is_finite()
    }

    #[inline]
    fn abs(value: Self::Scalar) -> Self::Scalar {
        use num_traits::float::FloatCore;
        value.abs()
    }

    #[inline]
    fn max(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a.max(b)
    }

    #[inline]
    fn min(a: Self::Scalar, b: Self::Scalar) -> Self::Scalar {
        a.min(b)
    }
}
