//! Elementary functions (the ones you learn in high school)

pub mod exponential;
pub mod hyperbolic;
pub mod power;
pub mod trigonometry;

pub use self::{
    exponential::Exponential, hyperbolic::Hyperbolic, power::Power, trigonometry::Trigonometry,
};
