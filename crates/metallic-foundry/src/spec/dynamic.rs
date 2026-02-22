use std::{any::TypeId, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::step::TensorBindings;
/// A value that can be either a literal or a variable reference.
///
/// In JSON, this can be:
/// - A number: `42` → `Literal(42)`
/// - A variable reference: `"{position_offset}"` → `Variable("position_offset")`
///
/// Variable references are resolved from TensorBindings globals at runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicValue<T> {
    /// A literal constant value.
    Literal(T),
    /// A variable name to look up in bindings (without braces).
    Variable(String),
}

impl<T: Default> Default for DynamicValue<T> {
    fn default() -> Self {
        DynamicValue::Literal(T::default())
    }
}

impl<T: FromStr + Copy + Default + 'static> DynamicValue<T> {
    /// Resolve this value, looking up variables from bindings if needed.
    ///
    /// Panics if variable lookup or parsing fails.
    pub fn resolve(&self, bindings: &TensorBindings) -> T {
        match self {
            DynamicValue::Literal(v) => *v,
            DynamicValue::Variable(name) => {
                // Optimization: If T is u32, checking int_globals first avoids String parsing.
                // This is a compile-time check that optimizes away for other types.
                if TypeId::of::<T>() == TypeId::of::<u32>() {
                    if let Some(v) = bindings.get_int_global(name) {
                        let val_u32 = v as u32;
                        return crate::types::cast_from_ref(&val_u32)
                            .unwrap_or_else(|| panic!("DynamicValue type cast failed for u32 '{}'", name));
                    }
                    if let Some(s) = bindings.get_var(name)
                        && let Ok(v) = s.parse::<u32>()
                    {
                        return crate::types::cast_from_ref(&v)
                            .unwrap_or_else(|| panic!("DynamicValue type cast failed for u32 '{}'", name));
                    }
                    panic_missing_dynamic_value(name, std::any::type_name::<T>());
                }

                // Optimization: If T is usize, checking int_globals first.
                if TypeId::of::<T>() == TypeId::of::<usize>() {
                    if let Some(v) = bindings.get_int_global(name) {
                        return crate::types::cast_from_ref(&v)
                            .unwrap_or_else(|| panic!("DynamicValue type cast failed for usize '{}'", name));
                    }
                    if let Some(s) = bindings.get_var(name)
                        && let Ok(v) = s.parse::<usize>()
                    {
                        return crate::types::cast_from_ref(&v)
                            .unwrap_or_else(|| panic!("DynamicValue type cast failed for usize '{}'", name));
                    }
                    panic_missing_dynamic_value(name, std::any::type_name::<T>());
                }

                // Fallback to string globals
                if let Some(s) = bindings.get_var(name)
                    && let Ok(v) = s.parse::<T>()
                {
                    return v;
                }
                panic_missing_dynamic_value(name, std::any::type_name::<T>());
            }
        }
    }

    /// Returns true if this is a literal value.
    pub fn is_literal(&self) -> bool {
        matches!(self, DynamicValue::Literal(_))
    }

    /// Returns true if this is a variable reference.
    pub fn is_variable(&self) -> bool {
        matches!(self, DynamicValue::Variable(_))
    }

    /// Get the literal value if this is a literal, None otherwise.
    pub fn as_literal(&self) -> Option<T> {
        match self {
            DynamicValue::Literal(v) => Some(*v),
            DynamicValue::Variable(_) => None,
        }
    }
}

#[cold]
#[inline(never)]
fn panic_missing_dynamic_value(name: &str, ty: &'static str) -> ! {
    panic!(
        "Missing required dynamic value '{name}' (type {ty}). Add it to the DSL (architecture.prepare.globals/derived_globals) or pass a runtime override."
    )
}

impl<T: Serialize> Serialize for DynamicValue<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            DynamicValue::Literal(v) => v.serialize(serializer),
            DynamicValue::Variable(name) => serializer.serialize_str(&format!("{{{}}}", name)),
        }
    }
}

impl<'de, T: Deserialize<'de> + FromStr> Deserialize<'de> for DynamicValue<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct DynamicValueVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T: Deserialize<'de> + FromStr> Visitor<'de> for DynamicValueVisitor<T> {
            type Value = DynamicValue<T>;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a value or a string like \"{variable_name}\"")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                // Try to convert via string parsing for generic T
                let s = value.to_string();
                T::from_str(&s)
                    .map(DynamicValue::Literal)
                    .map_err(|_| de::Error::custom(format!("invalid value: {}", value)))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let s = value.to_string();
                T::from_str(&s)
                    .map(DynamicValue::Literal)
                    .map_err(|_| de::Error::custom(format!("invalid value: {}", value)))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let s = value.to_string();
                T::from_str(&s)
                    .map(DynamicValue::Literal)
                    .map_err(|_| de::Error::custom(format!("invalid value: {}", value)))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                // Check for "{variable}" pattern
                if value.starts_with('{') && value.ends_with('}') && value.len() > 2 {
                    let var_name = &value[1..value.len() - 1];
                    Ok(DynamicValue::Variable(var_name.to_string()))
                } else {
                    // Try parsing as the target type
                    T::from_str(value)
                        .map(DynamicValue::Literal)
                        .map_err(|_| de::Error::custom(format!("invalid dynamic value: {}", value)))
                }
            }
        }

        deserializer.deserialize_any(DynamicValueVisitor(std::marker::PhantomData))
    }
}

/// Trait for params structs containing DynamicValue fields.
///
/// Implement this trait to enable runtime resolution of dynamic values.
/// The macro `#[derive(Resolvable)]` can generate this automatically.
pub trait Resolvable {
    /// The concrete output type with all dynamic values resolved.
    type Resolved;

    /// Resolve all dynamic values from bindings.
    fn resolve(&self, bindings: &TensorBindings) -> Self::Resolved;
}

#[path = "dynamic.test.rs"]
mod tests;
