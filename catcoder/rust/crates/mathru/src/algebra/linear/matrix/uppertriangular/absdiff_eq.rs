use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::UpperTriangular;

impl<T> AbsDiffEq for UpperTriangular<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T>,
{
    type Epsilon = T;

    fn default_epsilon() -> T {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &UpperTriangular<T>, epsilon: T) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}
