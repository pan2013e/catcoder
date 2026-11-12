use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::General;

impl<T> AbsDiffEq for General<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T>,
{
    type Epsilon = T;

    fn default_epsilon() -> T {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &General<T>, epsilon: T) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for (a, b) in self.iter().zip(other.iter()) {
            if a.abs_diff_ne(b, epsilon) {
                return false;
            }
        }

        true
    }
}
