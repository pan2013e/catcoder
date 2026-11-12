use super::LowerTriangular;
use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;

impl<T> AbsDiffEq for LowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T>,
{
    type Epsilon = T;

    fn default_epsilon() -> T {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &LowerTriangular<T>, epsilon: T) -> bool {
        // Implement optimized version
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}
