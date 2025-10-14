pub mod types;
pub mod dispatcher;
pub mod prefs;
pub mod execute;
pub mod dispatch_op;
pub mod constants;

pub use dispatch_op::MatmulDispatchOp;

#[cfg(test)]
mod dispatcher_test;

#[cfg(test)]
mod prefs_test;


