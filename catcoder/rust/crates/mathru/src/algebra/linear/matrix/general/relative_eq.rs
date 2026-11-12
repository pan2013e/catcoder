use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::RelativeEq;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::General;

impl<T> RelativeEq for General<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    fn default_max_relative() -> T {
        T::default_max_relative()
    }

    /// A test for equality that uses a relative comparison if the values are far apart.
    fn relative_eq(
        &self,
        other: &General<T>,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        if self.data.len() != other.data.len() {
            return false;
        }

        for (a, b) in self.iter().zip(other.iter()) {
            if a.relative_ne(b, epsilon, max_relative) {
                return false;
            }
        }
        true
    }
}
