use std::{
    fmt::Debug, ops::{Add, Div, Mul, Sub}
};

use crate::policy::{f16::PolicyF16, q8::PolicyQ8};

/// Defines the element type for a tensor
pub trait TensorElement: Clone + Copy + Default + 'static {
    type Scalar: Clone
        + Copy
        + Debug
        + Default
        + Add<Output = Self::Scalar>
        + Sub<Output = Self::Scalar>
        + Mul<Output = Self::Scalar>
        + Div<Output = Self::Scalar>
        + PartialOrd
        + PartialEq
        + 'static;
    const DTYPE: Dtype;
    type Policy: Default;

    fn from_f32(v: f32) -> Self::Scalar;
    fn to_f32(v: Self::Scalar) -> f32;
    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32>;
    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar>;
    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]);
    fn abs(v: Self::Scalar) -> Self::Scalar;
    fn is_finite(v: Self::Scalar) -> bool;
}

// Re-export Dtype from SDK
pub use metallic_sdk::Dtype;

pub trait DtypeExt {
    fn metal_format(&self) -> &'static str;
    fn layout_size(&self, dims: &[usize]) -> usize;
}

impl DtypeExt for Dtype {
    fn metal_format(&self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F16 => "half",
            Dtype::U32 => "uint",
            Dtype::Q4_0 | Dtype::Q6_K | Dtype::Q8_0 => "uchar",
            _ => panic!("Unsupported dtype for metal_format: {}", self),
        }
    }

    fn layout_size(&self, dims: &[usize]) -> usize {
        let elements: usize = dims.iter().product();
        match self {
            Dtype::Q4_0 => (elements + 1) / 2,
            // DEBT: Add other quantized layouts
            _ => elements * self.size_bytes(),
        }
    }
}

// DEBT: Should we consider combining TensorElements and the Policy, instead of having a F32 that implements tensorelement and a F32Policy that implements Policy? Maybe just have F32 implement policy and on the dtype to collapse and improve DX.
// DEBT: It feels like we could easily derive TensorElement on a Dtype or on the policy or something like that to clean up DX since some patterns seem very repeated.
// F32 implementation
#[derive(Clone, Copy, Default)]
pub struct F32;

impl TensorElement for F32 {
    type Scalar = f32;
    const DTYPE: Dtype = Dtype::F32;
    type Policy = PolicyF16; // TODO: Implement specific PolicyF32 if needed. Currently falls back to F16.

    fn from_f32(v: f32) -> Self::Scalar {
        v
    }

    fn to_f32(v: Self::Scalar) -> f32 {
        v
    }

    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32> {
        slice.to_vec()
    }

    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar> {
        slice.to_vec()
    }

    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        debug_assert_eq!(src.len(), dest.len());
        dest.copy_from_slice(src);
    }

    fn abs(v: Self::Scalar) -> Self::Scalar {
        v.abs()
    }

    fn is_finite(v: Self::Scalar) -> bool {
        v.is_finite()
    }
}

// U32 implementation
#[derive(Clone, Copy, Default)]
pub struct U32;

impl TensorElement for U32 {
    type Scalar = u32;
    const DTYPE: Dtype = Dtype::U32;
    type Policy = crate::policy::raw::PolicyU32;

    fn from_f32(v: f32) -> Self::Scalar {
        v as u32
    }

    fn to_f32(v: Self::Scalar) -> f32 {
        v as f32
    }

    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32> {
        slice.iter().map(|&x| x as f32).collect()
    }

    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar> {
        slice.iter().map(|&x| x as u32).collect()
    }

    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        debug_assert_eq!(src.len(), dest.len());
        for (dst, value) in dest.iter_mut().zip(src.iter().copied()) {
            *dst = value as u32;
        }
    }

    fn abs(v: Self::Scalar) -> Self::Scalar {
        v // u32 is always non-negative
    }

    fn is_finite(_v: Self::Scalar) -> bool {
        true // u32 is always finite
    }
}

// Q4_0 implementation
#[derive(Clone, Copy, Default)]
pub struct Q4_0;

impl TensorElement for Q4_0 {
    type Scalar = u8;
    const DTYPE: Dtype = Dtype::Q4_0;
    type Policy = crate::policy::q4_0::PolicyQ4_0;

    fn from_f32(v: f32) -> Self::Scalar {
        // Saturate to [0, 255]
        v.clamp(0.0, 255.0) as u8
    }

    fn to_f32(v: Self::Scalar) -> f32 {
        v as f32
    }

    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32> {
        slice.iter().map(|&x| x as f32).collect()
    }

    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar> {
        slice.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect()
    }

    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        debug_assert_eq!(src.len(), dest.len());
        for (dst, value) in dest.iter_mut().zip(src.iter().copied()) {
            *dst = value.clamp(0.0, 255.0) as u8;
        }
    }

    fn abs(v: Self::Scalar) -> Self::Scalar {
        v
    }

    fn is_finite(_v: Self::Scalar) -> bool {
        true
    }
}

// U8 implementation — enables raw/packed byte tensors (e.g., GGUF Q8_0 blocks)
#[derive(Clone, Copy, Default)]
pub struct Q8_0;

impl TensorElement for Q8_0 {
    type Scalar = u8;
    const DTYPE: Dtype = Dtype::Q8_0;
    type Policy = PolicyQ8;

    fn from_f32(v: f32) -> Self::Scalar {
        // Saturate to [0, 255]
        v.clamp(0.0, 255.0) as u8
    }

    fn to_f32(v: Self::Scalar) -> f32 {
        v as f32
    }

    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32> {
        slice.iter().map(|&x| x as f32).collect()
    }

    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar> {
        slice.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect()
    }

    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        debug_assert_eq!(src.len(), dest.len());
        for (dst, value) in dest.iter_mut().zip(src.iter().copied()) {
            *dst = value.clamp(0.0, 255.0) as u8;
        }
    }

    fn abs(v: Self::Scalar) -> Self::Scalar {
        v // u8 is non-negative
    }

    fn is_finite(_v: Self::Scalar) -> bool {
        true // integers are always finite
    }
}

// F16 implementation
#[derive(Clone, Copy, Default)]
pub struct F16;

impl TensorElement for F16 {
    type Scalar = half::f16;
    const DTYPE: Dtype = Dtype::F16;
    type Policy = PolicyF16;

    fn from_f32(v: f32) -> Self::Scalar {
        half::f16::from_f32(v)
    }

    fn to_f32(v: Self::Scalar) -> f32 {
        v.to_f32()
    }

    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32> {
        slice.iter().map(|x| x.to_f32()).collect()
    }

    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar> {
        slice.iter().map(|&x| half::f16::from_f32(x)).collect()
    }

    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]) {
        debug_assert_eq!(src.len(), dest.len());
        for (dst, value) in dest.iter_mut().zip(src.iter().copied()) {
            *dst = half::f16::from_f32(value);
        }
    }

    fn abs(v: Self::Scalar) -> Self::Scalar {
        half::f16::from_f32(v.to_f32().abs())
    }

    fn is_finite(v: Self::Scalar) -> bool {
        v.is_finite()
    }
}
