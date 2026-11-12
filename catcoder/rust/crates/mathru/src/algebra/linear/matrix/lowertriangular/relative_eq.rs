use crate::algebra::abstr::{AbsDiffEq, Field, RelativeEq, Scalar};

use super::LowerTriangular;

impl<T> RelativeEq for LowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    fn default_max_relative() -> T {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &LowerTriangular<T>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        // TODO implement opitimized vsersion
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}
