use super::Diagonal;
use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;

impl<T> AbsDiffEq for Diagonal<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T>,
{
    type Epsilon = T;

    fn default_epsilon() -> T {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Diagonal<T>, epsilon: T) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for i in 0..self.dim().0 {
            let a = self[[i, i]];
            let b = other[[i, i]];
            if a.abs_diff_ne(&b, epsilon) {
                return false;
            }
        }

        true
    }
}
