pub mod constants;
pub mod dispatch_op;
pub mod dispatcher;
pub mod execute;
pub mod prefs;
pub mod types;

pub use dispatch_op::MatmulDispatchOp;

#[cfg(test)]
mod dispatcher_test;

#[cfg(test)]
mod prefs_test;
