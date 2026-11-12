use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::Determinant;
use crate::algebra::linear::matrix::UpperTriangular;

impl<T> Determinant<T> for UpperTriangular<T>
where
    T: Field + Scalar,
{
    /// Calculates the determinant
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Determinant, UpperTriangular, General};
    /// use mathru::matrix;
    ///
    /// let a: UpperTriangular<f64> = matrix![  2.0, 0.0;
    ///                                         5.0, -3.0].into();
    /// let det: f64 = a.det();
    /// assert_eq!(det, -6.0)
    /// ```
    fn det(&self) -> T {
        let (m, n) = self.matrix.dim();
        debug_assert_eq!(m, n);

        let mut det = T::one();
        for i in 0..m {
            det *= self[[i, i]];
        }

        det
    }
}
