#![cfg(test)]

use super::*;

#[test]
fn test_primitive_types() {
    assert_eq!(u32::metal_type_str(), "uint");
    assert_eq!(f32::metal_type_str(), "float");
    assert_eq!(f16::metal_type_str(), "half");
}

#[test]
fn test_pointer_types() {
    assert_eq!(DevicePtr::<f16>::metal_type_str(), "const device half*");
    assert_eq!(DevicePtrMut::<f16>::metal_type_str(), "device half*");
}
