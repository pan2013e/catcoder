use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::Determinant;
use crate::algebra::linear::matrix::UnitUpperTriangular;

impl<T> Determinant<T> for UnitUpperTriangular<T>
where
    T: Field + Scalar,
{
    /// Calculates the determinant
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Determinant, UnitUpperTriangular, General};
    /// use mathru::matrix;
    ///
    /// let a: UnitUpperTriangular<f64> = matrix![  1.0, 0.0;
    ///                                             5.0, 1.0].into();
    /// let det: f64 = a.det();
    /// assert_eq!(det, 1.0)
    /// ```
    fn det(&self) -> T {
        T::one()
    }
}
