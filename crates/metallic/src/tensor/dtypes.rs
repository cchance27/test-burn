use std::{
    fmt::Debug, ops::{Add, Div, Mul, Sub}
};

use serde::{Deserialize, Serialize};

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

    fn from_f32(v: f32) -> Self::Scalar;
    fn to_f32(v: Self::Scalar) -> f32;
    fn to_f32_vec(slice: &[Self::Scalar]) -> Vec<f32>;
    fn from_f32_slice(slice: &[f32]) -> Vec<Self::Scalar>;
    fn copy_from_f32_slice(src: &[f32], dest: &mut [Self::Scalar]);
    fn abs(v: Self::Scalar) -> Self::Scalar;
    fn is_finite(v: Self::Scalar) -> bool;
}

/// Represents the data type of tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Dtype {
    F32,
    F16,
    U8,
    U32,
    // Add more as needed
}

impl Dtype {
    pub fn size_bytes(&self) -> usize {
        match self {
            Dtype::F32 => std::mem::size_of::<f32>(),
            Dtype::F16 => std::mem::size_of::<half::f16>(),
            Dtype::U8 => std::mem::size_of::<u8>(),
            Dtype::U32 => std::mem::size_of::<u32>(),
        }
    }
    pub fn metal_format(&self) -> &'static str {
        match self {
            Dtype::F32 => "float",
            Dtype::F16 => "half",
            Dtype::U8 => "uchar",
            Dtype::U32 => "uint",
        }
    }
}

impl From<Dtype> for objc2_metal_performance_shaders::MPSDataType {
    fn from(value: Dtype) -> Self {
        use objc2_metal_performance_shaders::MPSDataType;
        match value {
            Dtype::F32 => MPSDataType::Float32,
            Dtype::F16 => MPSDataType::Float16,
            Dtype::U8 => MPSDataType::UInt8,
            Dtype::U32 => MPSDataType::UInt32,
        }
    }
}

// F32 implementation
#[derive(Clone, Copy, Default)]
pub struct F32;

impl TensorElement for F32 {
    type Scalar = f32;
    const DTYPE: Dtype = Dtype::F32;

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

// U8 implementation — enables raw/packed byte tensors (e.g., GGUF Q8_0 blocks)
#[derive(Clone, Copy, Default)]
pub struct U8;

impl TensorElement for U8 {
    type Scalar = u8;
    const DTYPE: Dtype = Dtype::U8;

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
