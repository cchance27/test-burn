//! Metal type markers and traits for type-safe Rust↔Metal code generation.
//!
//! This module provides:
//! - Marker types for Metal buffer pointer semantics (`DevicePtr<T>`, `DevicePtrMut<T>`, etc.)
//! - The `MetalType` trait for Rust→Metal type mapping
//! - Compile-time inference of Metal type strings from Rust types

use std::marker::PhantomData;

use half::f16;

/// Marker for `const device T*` buffer pointer (read-only device memory).
#[derive(Debug, Clone, Copy)]
pub struct DevicePtr<T>(PhantomData<T>);

/// Marker for `device T*` buffer pointer (read-write device memory).
#[derive(Debug, Clone, Copy)]
pub struct DevicePtrMut<T>(PhantomData<T>);

/// Marker for `constant T&` reference (uniform/constant memory).
#[derive(Debug, Clone, Copy)]
pub struct ConstantRef<T>(PhantomData<T>);

/// Marker for `constant T*` pointer (constant buffer).
#[derive(Debug, Clone, Copy)]
pub struct ConstantPtr<T>(PhantomData<T>);

/// Trait for mapping Rust types to Metal type strings.
///
/// This enables compile-time generation of Metal function signatures
/// from Rust struct definitions.
pub trait MetalType {
    /// Returns the Metal type string for this type.
    /// E.g., `u32` → `"uint"`, `DevicePtr<f16>` → `"const device half*"`
    fn metal_type_str() -> &'static str;
}

// Primitive type mappings
impl MetalType for u8 {
    fn metal_type_str() -> &'static str {
        "uchar"
    }
}

impl MetalType for i8 {
    fn metal_type_str() -> &'static str {
        "char"
    }
}

impl MetalType for u16 {
    fn metal_type_str() -> &'static str {
        "ushort"
    }
}

impl MetalType for i16 {
    fn metal_type_str() -> &'static str {
        "short"
    }
}

impl MetalType for u32 {
    fn metal_type_str() -> &'static str {
        "uint"
    }
}

impl MetalType for i32 {
    fn metal_type_str() -> &'static str {
        "int"
    }
}

impl MetalType for u64 {
    fn metal_type_str() -> &'static str {
        "ulong"
    }
}

impl MetalType for i64 {
    fn metal_type_str() -> &'static str {
        "long"
    }
}

impl MetalType for f32 {
    fn metal_type_str() -> &'static str {
        "float"
    }
}

impl MetalType for f16 {
    fn metal_type_str() -> &'static str {
        "half"
    }
}

// Pointer type mappings (these use const strings since we can't concat at compile time easily)
impl MetalType for DevicePtr<u8> {
    fn metal_type_str() -> &'static str {
        "const device uchar*"
    }
}

impl MetalType for DevicePtr<f16> {
    fn metal_type_str() -> &'static str {
        "const device half*"
    }
}

impl MetalType for DevicePtr<f32> {
    fn metal_type_str() -> &'static str {
        "const device float*"
    }
}

impl MetalType for DevicePtrMut<u8> {
    fn metal_type_str() -> &'static str {
        "device uchar*"
    }
}

impl MetalType for DevicePtrMut<f16> {
    fn metal_type_str() -> &'static str {
        "device half*"
    }
}

impl MetalType for DevicePtrMut<f32> {
    fn metal_type_str() -> &'static str {
        "device float*"
    }
}

impl MetalType for ConstantRef<u32> {
    fn metal_type_str() -> &'static str {
        "constant uint&"
    }
}

impl MetalType for ConstantRef<f32> {
    fn metal_type_str() -> &'static str {
        "constant float&"
    }
}

/// Helper to get Metal type string at runtime for known types.
/// Used by macros and build.rs for dynamic type resolution.
pub fn rust_to_metal_type(rust_type: &str) -> Option<&'static str> {
    match rust_type {
        "u8" => Some("uchar"),
        "i8" => Some("char"),
        "u16" => Some("ushort"),
        "i16" => Some("short"),
        "u32" => Some("uint"),
        "i32" => Some("int"),
        "u64" => Some("ulong"),
        "i64" => Some("long"),
        "f32" => Some("float"),
        "f16" | "half::f16" => Some("half"),
        _ => None,
    }
}

#[path = "metal.test.rs"]
mod tests;
