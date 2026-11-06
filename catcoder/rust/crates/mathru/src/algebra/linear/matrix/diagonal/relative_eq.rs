use super::Diagonal;
use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::RelativeEq;
use crate::algebra::abstr::Scalar;

impl<T> RelativeEq for Diagonal<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    fn default_max_relative() -> T {
        T::default_max_relative()
    }

    /// A test for equality that uses a relative comparison if the values are far apart.
    fn relative_eq(
        &self,
        other: &Diagonal<T>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for i in 0..self.dim().0 {
            let a = self[[i, i]];
            let b = other[[i, i]];
            if a.relative_ne(&b, epsilon, max_relative) {
                return false;
            }
        }
        true
    }
}
