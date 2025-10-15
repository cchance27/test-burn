pub mod dispatch_op;
pub mod dispatcher;
mod execute;
pub mod prefs;
pub mod types; // expose environment-driven dispatcher preferences

pub use dispatch_op::SoftmaxDispatchOp;
