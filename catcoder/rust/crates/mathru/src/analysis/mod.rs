//! Interpolations, integration, differential equations, etc.

#[macro_use]
mod function;
mod hessian;
mod jacobian;

mod newton_raphson;

pub mod differential_equation;
pub mod integral;
pub mod interpolation;

pub use function::Function;
pub use hessian::Hessian;
pub use jacobian::Jacobian;
pub use newton_raphson::NewtonRaphson;
