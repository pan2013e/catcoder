use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::Determinant;
use crate::algebra::linear::matrix::LowerTriangular;

impl<T> Determinant<T> for LowerTriangular<T>
where
    T: Field + Scalar,
{
    /// Calculates the determinant
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Determinant, LowerTriangular, General};
    /// use mathru::matrix;
    ///
    /// let a: LowerTriangular<f64> = matrix![  2.0, 0.0;
    ///                                         3.0, -7.0].into();
    /// let det: f64 = a.det();
    /// assert_eq!(det, -14.0)
    /// ```
    fn det(&self) -> T {
        let (m, n) = self.matrix.dim();
        debug_assert_eq!(m, n);

        let mut det = T::one();

        for k in 0..m {
            det *= self[[k, k]];
        }

        det
    }
}
