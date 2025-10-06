//! Shared process environment helpers for Metallic instrumentation.

pub mod environment;

pub use environment::guard::EnvVarGuard;
pub use environment::instrument::InstrumentEnvVar;
pub use environment::{EnvVar, Environment};

#[cfg(test)]
pub use environment::{guard, instrument};
