//! Solves an ODE using Ssprk3's method.
use crate::algebra::abstr::Real;
use crate::analysis::differential_equation::ordinary::solver::explicit::runge_kutta::fixed::{
    ExplicitRK, ExplicitRKMethod,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;
use std::default::Default;

/// Solves an ODE using Third-order Strong Stability Preserving Runge-Kutta(SSPRK3).
///
/// <https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods>
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Ssprk3<T> {
    butcher: ExplicitRK<T>,
}

impl<T> Default for Ssprk3<T>
where
    T: Real,
{
    /// Creates a Ssprk3 instance
    fn default() -> Ssprk3<T> {
        let a: Vec<T> = vec![T::one(), T::from_f64(1.0 / 4.0), T::from_f64(1.0 / 4.0)];
        let b: Vec<T> = vec![
            T::from_f64(1.0 / 6.0),
            T::from_f64(1.0 / 6.0),
            T::from_f64(2.0 / 3.0),
        ];
        let c: Vec<T> = vec![T::one(), T::from_f64(0.5)];

        Ssprk3 {
            butcher: ExplicitRK::new(a, b, 3, c),
        }
    }
}

impl<T> ExplicitRKMethod<T> for Ssprk3<T> {
    fn tableau(&self) -> &ExplicitRK<T> {
        &self.butcher
    }
}
