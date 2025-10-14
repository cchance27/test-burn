pub mod dispatch_op;
pub mod dispatcher;
mod execute;
pub mod types;
pub mod prefs; // expose environment-driven dispatcher preferences

pub use dispatch_op::SoftmaxDispatchOp;
