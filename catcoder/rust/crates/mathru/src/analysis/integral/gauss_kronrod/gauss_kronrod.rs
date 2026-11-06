use crate::algebra::abstr::Real;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::RootWeight;

/// Gauss Kronrod quadrature
///
/// ```math
/// \int_{a}^{b}f(x)\,dx \approx \frac{b - a}{2}\sum_{i=1}^{n}f(\frac{b - a}{2}x_i + \frac{a + b}{2}) w_i
/// ```
///
/// <https://en.wikipedia.org/wiki/Gauss–Kronrod_quadrature_formula>
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct GaussKronrod<T> {
    root_weight: RootWeight<T>,
}

impl<T> GaussKronrod<T>
where
    T: Real,
{
    /// # Arguments
    ///
    ///
    /// # Panics
    ///
    /// # Examples
    /// ```
    /// use mathru::analysis::integral::gauss_kronrod::GaussKronrod;
    /// use mathru::assert_relative_eq;
    ///
    /// let gk: GaussKronrod<f64> = GaussKronrod::new();
    /// let f: fn(f64) -> f64 = | x | {x};
    ///
    /// let integral: f64 = gk.integrate(f, 2.0, 4.0);
    ///
    /// assert_relative_eq!(integral, 6.0, epsilon=1.0e-8);
    /// ```
    pub fn new() -> GaussKronrod<T> {
        GaussKronrod {
            root_weight: RootWeight::new(),
        }
    }

    /// Integrate function f from lower bound a to upper bound b
    ///
    /// # Arguments
    /// * a: lower bound of the definite integral
    /// * b: upper bound of the definite integral
    pub fn integrate<F>(&self, f: F, a: T, b: T) -> T
    where
        F: Fn(T) -> T,
    {
        let sum = (&self.root_weight).iter().fold(T::zero(), |s, (x_i, a_i)| {
            let x_i_scaled = (b - a) / T::from_f64(2.0) * x_i + (a + b) / T::from_f64(2.0);
            s + f(x_i_scaled) * a_i
        });

        (b - a) / T::from_f64(2.0) * sum
    }
}
