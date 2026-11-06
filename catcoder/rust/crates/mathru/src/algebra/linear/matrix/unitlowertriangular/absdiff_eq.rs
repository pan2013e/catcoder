use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::UnitLowerTriangular;

impl<T> AbsDiffEq for UnitLowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T>,
{
    type Epsilon = T;

    fn default_epsilon() -> T {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &UnitLowerTriangular<T>, epsilon: T) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}
