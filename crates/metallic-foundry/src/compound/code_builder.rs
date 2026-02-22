//! CodeBuilder: Safe Metal code generation with strongly-typed variable tracking.
//!
//! This module provides utilities for generating Metal code in compound kernels
//! with proper variable naming, type safety, and scoping.
//!
//! # Design Philosophy
//! - **Strongly-typed**: Variables carry Metal type information
//! - **Safe references**: Undefined variable access causes compile/runtime errors
//! - **Flexible reductions**: Configurable SIMD reduce levels for future fusion support
//! - **Clean DX**: Clear API that minimizes string manipulation bugs

use std::{collections::HashMap, fmt};

// ============================================================================
// Metal Types
// ============================================================================

/// Metal scalar and vector types for type-safe code generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalType {
    // Scalar types
    Half,
    Float,
    Uint,
    Int,
    Bool,

    // Vector types
    Half2,
    Half4,
    Float2,
    Float4,
    Uint2,
    Uint4,

    /// Pointer to device memory
    DevicePtr(MetalScalar),

    /// Pointer to constant memory
    ConstantPtr(MetalScalar),

    /// Custom type for extensibility (e.g., user-defined structs)
    Custom(&'static str),
}

/// Scalar types (subset used for pointer declarations).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MetalScalar {
    Half,
    Float,
    Uint,
    Int,
    Uchar,
}

impl MetalType {
    /// Get the Metal syntax for this type.
    pub fn as_str(&self) -> &'static str {
        match self {
            MetalType::Half => "half",
            MetalType::Float => "float",
            MetalType::Uint => "uint",
            MetalType::Int => "int",
            MetalType::Bool => "bool",
            MetalType::Half2 => "half2",
            MetalType::Half4 => "half4",
            MetalType::Float2 => "float2",
            MetalType::Float4 => "float4",
            MetalType::Uint2 => "uint2",
            MetalType::Uint4 => "uint4",
            MetalType::DevicePtr(_) => "device",     // caller handles full syntax
            MetalType::ConstantPtr(_) => "constant", // caller handles full syntax
            MetalType::Custom(s) => s,
        }
    }

    /// Get full declaration syntax (for pointers).
    pub fn declaration_str(&self) -> String {
        match self {
            MetalType::DevicePtr(scalar) => format!("device {}*", scalar.as_str()),
            MetalType::ConstantPtr(scalar) => format!("constant {}*", scalar.as_str()),
            _ => self.as_str().to_string(),
        }
    }
}

impl MetalScalar {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetalScalar::Half => "half",
            MetalScalar::Float => "float",
            MetalScalar::Uint => "uint",
            MetalScalar::Int => "int",
            MetalScalar::Uchar => "uchar",
        }
    }
}

impl fmt::Display for MetalType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Metal Variables
// ============================================================================

/// A tracked Metal variable with name and type information.
///
/// Variables are created via `CodeBuilder::declare_var()` and can be
/// referenced safely in generated code.
#[derive(Clone, Debug)]
pub struct MetalVar {
    name: String,
    metal_type: MetalType,
}

impl MetalVar {
    /// Get the variable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the Metal type.
    pub fn metal_type(&self) -> MetalType {
        self.metal_type
    }

    /// Generate a declaration statement: `type name;`
    pub fn declaration(&self) -> String {
        format!("{} {};", self.metal_type.declaration_str(), self.name)
    }

    /// Generate a declaration with initializer: `type name = value;`
    pub fn declaration_with_init(&self, value: &str) -> String {
        format!("{} {} = {};", self.metal_type.declaration_str(), self.name, value)
    }
}

impl fmt::Display for MetalVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ============================================================================
// SIMD Reduction Configuration
// ============================================================================

/// Reduction operation for SIMD shuffle reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReduceOp {
    /// Addition: `val += simd_shuffle_xor(...)`
    #[default]
    Add,
    /// Maximum: `val = max(val, simd_shuffle_xor(...))`
    Max,
    /// Minimum: `val = min(val, simd_shuffle_xor(...))`
    Min,
}

impl ReduceOp {
    /// Get the Metal code for combining two values.
    pub fn combine_expr(&self, lhs: &str, rhs: &str) -> String {
        match self {
            ReduceOp::Add => format!("{} += {}", lhs, rhs),
            ReduceOp::Max => format!("{} = max({}, {})", lhs, lhs, rhs),
            ReduceOp::Min => format!("{} = min({}, {})", lhs, lhs, rhs),
        }
    }
}

/// Configuration for SIMD shuffle-based reductions.
///
/// # Levels
/// - `from_level`: Starting shuffle distance (e.g., 16 for 32-lane)
/// - `to_level`: Ending shuffle distance (typically 1)
///
/// # Example
/// ```text
/// // Full 32-lane reduction (default)
/// SimdReduceConfig::default()  // from=16, to=1, op=Add
///
/// // 16-lane reduction
/// SimdReduceConfig::new(8, 1, ReduceOp::Add)
///
/// // Single level reduction
/// SimdReduceConfig::new(4, 4, ReduceOp::Max)
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SimdReduceConfig {
    /// Starting shuffle distance (power of 2: 16, 8, 4, 2, or 1)
    pub from_level: u8,
    /// Ending shuffle distance (power of 2, must be <= from_level)
    pub to_level: u8,
    /// Reduction operation
    pub op: ReduceOp,
}

impl Default for SimdReduceConfig {
    /// Default: 32-lane reduction with Add operation.
    fn default() -> Self {
        Self {
            from_level: 16,
            to_level: 1,
            op: ReduceOp::Add,
        }
    }
}

impl SimdReduceConfig {
    /// Create a new SIMD reduce config.
    pub fn new(from_level: u8, to_level: u8, op: ReduceOp) -> Self {
        assert!(from_level.is_power_of_two() || from_level == 0, "from_level must be power of 2");
        assert!(to_level.is_power_of_two() || to_level == 0, "to_level must be power of 2");
        assert!(to_level <= from_level, "to_level must be <= from_level");
        Self { from_level, to_level, op }
    }

    /// Create a full 32-lane reduction config with given operation.
    pub fn full_32_lane(op: ReduceOp) -> Self {
        Self::new(16, 1, op)
    }

    /// Create a 16-lane reduction config (for smaller SIMD groups).
    pub fn lane_16(op: ReduceOp) -> Self {
        Self::new(8, 1, op)
    }

    /// Iterator over shuffle distances (e.g., [16, 8, 4, 2, 1]).
    pub fn levels(&self) -> impl Iterator<Item = u8> {
        let from = self.from_level;
        let to = self.to_level;
        std::iter::successors(Some(from), move |&n| {
            let next = n / 2;
            if next >= to && next > 0 { Some(next) } else { None }
        })
    }
}

// ============================================================================
// Code Builder
// ============================================================================

/// Builder for generating Metal code with automatic variable management.
///
/// # Example
/// ```text
/// let mut b = CodeBuilder::new("rmsnorm");
/// let inv_rms = b.declare_var("inv_rms", MetalType::Float);
/// b.emit_decl_init(&inv_rms, "compute_inv_rms(input)");
///
/// let output = b.declare_var("out", MetalType::Half);
/// b.emit_assign(&output, &format!("(half)(input * {})", inv_rms));
///
/// b.emit_simd_reduce(&inv_rms, SimdReduceConfig::default());
///
/// let (final_var, code) = b.finish();
/// ```
#[derive(Debug)]
pub struct CodeBuilder {
    stage_name: String,
    code: String,
    var_counter: usize,
    defined_vars: HashMap<String, MetalType>,
    current_output_var: Option<String>,
    indent: usize,
}

impl CodeBuilder {
    /// Create a new CodeBuilder for the given stage.
    pub fn new(stage_name: &str) -> Self {
        Self {
            stage_name: stage_name.to_string(),
            code: String::new(),
            var_counter: 0,
            defined_vars: HashMap::new(),
            current_output_var: None,
            indent: 2, // default 2 spaces
        }
    }

    /// Set the indentation level (in spaces).
    pub fn set_indent(&mut self, spaces: usize) {
        self.indent = spaces;
    }

    /// Get current indentation string.
    fn indent_str(&self) -> String {
        " ".repeat(self.indent)
    }

    // ========================================================================
    // Variable Management
    // ========================================================================

    /// Declare a new typed variable, returning a `MetalVar` handle.
    pub fn declare_var(&mut self, prefix: &str, metal_type: MetalType) -> MetalVar {
        self.var_counter += 1;
        let name = format!("{}_{}{}", self.stage_name, prefix, self.var_counter);
        self.defined_vars.insert(name.clone(), metal_type);
        MetalVar { name, metal_type }
    }

    /// Create a reference to an external variable (not declared by this builder).
    /// Use for referencing kernel parameters or input variables.
    pub fn external_var(&mut self, name: &str, metal_type: MetalType) -> MetalVar {
        self.defined_vars.insert(name.to_string(), metal_type);
        MetalVar {
            name: name.to_string(),
            metal_type,
        }
    }

    /// Check if a variable was defined/registered.
    pub fn has_var(&self, name: &str) -> bool {
        self.defined_vars.contains_key(name)
    }

    /// Get the type of a defined variable.
    pub fn var_type(&self, name: &str) -> Option<MetalType> {
        self.defined_vars.get(name).copied()
    }

    // ========================================================================
    // Code Emission
    // ========================================================================

    /// Emit a raw line of Metal code (with automatic indentation).
    pub fn emit(&mut self, line: &str) {
        if !self.code.is_empty() {
            self.code.push('\n');
        }
        self.code.push_str(&self.indent_str());
        self.code.push_str(line);
    }

    /// Emit a raw line with NO indentation (for labels, preprocessor, etc.).
    pub fn emit_raw(&mut self, line: &str) {
        if !self.code.is_empty() {
            self.code.push('\n');
        }
        self.code.push_str(line);
    }

    /// Emit a blank line.
    pub fn emit_blank(&mut self) {
        self.code.push('\n');
    }

    /// Emit a comment.
    pub fn emit_comment(&mut self, comment: &str) {
        self.emit(&format!("// {}", comment));
    }

    /// Emit variable declaration: `type name;`
    pub fn emit_decl(&mut self, var: &MetalVar) {
        self.emit(&var.declaration());
    }

    /// Emit variable declaration with initializer: `type name = value;`
    pub fn emit_decl_init(&mut self, var: &MetalVar, value: &str) {
        self.emit(&var.declaration_with_init(value));
    }

    /// Emit assignment: `name = value;`
    pub fn emit_assign(&mut self, var: &MetalVar, value: &str) {
        self.emit(&format!("{} = {};", var.name(), value));
    }

    /// Emit compound assignment: `name op= value;` (e.g., `+=`, `*=`)
    pub fn emit_compound_assign(&mut self, var: &MetalVar, op: &str, value: &str) {
        self.emit(&format!("{} {}= {};", var.name(), op, value));
    }

    // ========================================================================
    // SIMD Reductions
    // ========================================================================

    /// Emit SIMD reduction with full configuration.
    pub fn emit_simd_reduce_with_config(&mut self, var_name: &str, config: SimdReduceConfig) {
        for level in config.levels() {
            let shuffle = format!("simd_shuffle_xor({}, {}u)", var_name, level);
            self.emit(&format!("{};", config.op.combine_expr(var_name, &shuffle)));
        }
    }

    /// Emit SIMD reduction for a MetalVar with config.
    pub fn emit_simd_reduce_var(&mut self, var: &MetalVar, config: SimdReduceConfig) {
        self.emit_simd_reduce_with_config(var.name(), config);
    }

    /// Emit default 32-lane SIMD reduction (Add operation).
    /// Convenience method for common case.
    pub fn emit_simd_reduce(&mut self, var_name: &str) {
        self.emit_simd_reduce_with_config(var_name, SimdReduceConfig::default());
    }

    /// Emit SIMD reduction for multiple variables.
    pub fn emit_simd_reduce_multi(&mut self, var_names: &[&str], config: SimdReduceConfig) {
        for level in config.levels() {
            for var_name in var_names {
                let shuffle = format!("simd_shuffle_xor({}, {}u)", var_name, level);
                self.emit(&format!("{};", config.op.combine_expr(var_name, &shuffle)));
            }
        }
    }

    // ========================================================================
    // Output Management
    // ========================================================================

    /// Set the current output variable name.
    pub fn set_output_var(&mut self, var: &MetalVar) {
        self.current_output_var = Some(var.name().to_string());
    }

    /// Set the output variable by name (for external variables).
    pub fn set_output_var_name(&mut self, name: &str) {
        self.current_output_var = Some(name.to_string());
    }

    /// Get the current output variable name, or generate one if not set.
    pub fn output_var(&mut self) -> String {
        match &self.current_output_var {
            Some(v) => v.clone(),
            None => {
                let v = self.declare_var("out", MetalType::Half);
                self.current_output_var = Some(v.name().to_string());
                v.name().to_string()
            }
        }
    }

    /// Finish building and return (output_var, code).
    pub fn finish(mut self) -> (String, String) {
        let output_var = self.output_var();
        (output_var, self.code)
    }

    /// Get the generated code so far.
    pub fn code(&self) -> &str {
        &self.code
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_type_display() {
        assert_eq!(MetalType::Half.as_str(), "half");
        assert_eq!(MetalType::Float4.as_str(), "float4");
        assert_eq!(MetalType::Custom("MyStruct").as_str(), "MyStruct");
    }

    #[test]
    fn test_metal_type_pointer_declaration() {
        let ptr = MetalType::DevicePtr(MetalScalar::Half);
        assert_eq!(ptr.declaration_str(), "device half*");

        let const_ptr = MetalType::ConstantPtr(MetalScalar::Float);
        assert_eq!(const_ptr.declaration_str(), "constant float*");
    }

    #[test]
    fn test_metal_var_declaration() {
        let var = MetalVar {
            name: "test_var".to_string(),
            metal_type: MetalType::Float,
        };
        assert_eq!(var.declaration(), "float test_var;");
        assert_eq!(var.declaration_with_init("1.0f"), "float test_var = 1.0f;");
    }

    #[test]
    fn test_code_builder_basic() {
        let mut builder = CodeBuilder::new("rmsnorm");
        let inv_rms = builder.declare_var("inv_rms", MetalType::Float);
        assert_eq!(inv_rms.name(), "rmsnorm_inv_rms1");
        assert_eq!(inv_rms.metal_type(), MetalType::Float);

        builder.set_output_var(&inv_rms);
        builder.emit_comment("test comment");
        builder.emit_decl_init(&inv_rms, "1.0f");

        let (out, code) = builder.finish();
        assert!(code.contains("// test comment"));
        assert!(code.contains("float rmsnorm_inv_rms1 = 1.0f;"));
        assert_eq!(out, inv_rms.name());
    }

    #[test]
    fn test_simd_reduce_config_levels() {
        let config = SimdReduceConfig::default();
        let levels: Vec<u8> = config.levels().collect();
        assert_eq!(levels, vec![16, 8, 4, 2, 1]);

        let config_16 = SimdReduceConfig::lane_16(ReduceOp::Add);
        let levels_16: Vec<u8> = config_16.levels().collect();
        assert_eq!(levels_16, vec![8, 4, 2, 1]);

        let config_single = SimdReduceConfig::new(4, 4, ReduceOp::Max);
        let levels_single: Vec<u8> = config_single.levels().collect();
        assert_eq!(levels_single, vec![4]);
    }

    #[test]
    fn test_simd_reduce_default() {
        let mut builder = CodeBuilder::new("test");
        builder.emit_simd_reduce("acc");
        let code = builder.code();
        assert!(code.contains("acc += simd_shuffle_xor(acc, 16u)"));
        assert!(code.contains("acc += simd_shuffle_xor(acc, 8u)"));
        assert!(code.contains("acc += simd_shuffle_xor(acc, 1u)"));
    }

    #[test]
    fn test_simd_reduce_max() {
        let mut builder = CodeBuilder::new("test");
        let config = SimdReduceConfig::full_32_lane(ReduceOp::Max);
        builder.emit_simd_reduce_with_config("val", config);
        let code = builder.code();
        assert!(code.contains("val = max(val, simd_shuffle_xor(val, 16u))"));
        assert!(code.contains("val = max(val, simd_shuffle_xor(val, 1u))"));
    }

    #[test]
    fn test_simd_reduce_multi() {
        let mut builder = CodeBuilder::new("test");
        let config = SimdReduceConfig::new(4, 1, ReduceOp::Add);
        builder.emit_simd_reduce_multi(&["a", "b"], config);
        let code = builder.code();
        // Should interleave reductions for a and b at each level
        assert!(code.contains("a += simd_shuffle_xor(a, 4u)"));
        assert!(code.contains("b += simd_shuffle_xor(b, 4u)"));
        assert!(code.contains("a += simd_shuffle_xor(a, 1u)"));
        assert!(code.contains("b += simd_shuffle_xor(b, 1u)"));
    }

    #[test]
    fn test_external_var() {
        let mut builder = CodeBuilder::new("stage");
        let params = builder.external_var("params", MetalType::ConstantPtr(MetalScalar::Float));
        assert_eq!(params.name(), "params");
        assert!(builder.has_var("params"));
    }

    #[test]
    fn test_reduce_op_combine() {
        assert_eq!(ReduceOp::Add.combine_expr("x", "y"), "x += y");
        assert_eq!(ReduceOp::Max.combine_expr("x", "y"), "x = max(x, y)");
        assert_eq!(ReduceOp::Min.combine_expr("x", "y"), "x = min(x, y)");
    }
}
