use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::RelativeEq;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::UnitLowerTriangular;

impl<T> RelativeEq for UnitLowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    fn default_max_relative() -> T {
        T::default_max_relative()
    }

    /// A test for equality that uses a relative comparison if the values are far apart.
    fn relative_eq(
        &self,
        other: &UnitLowerTriangular<T>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}
