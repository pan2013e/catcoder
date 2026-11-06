use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::Determinant;
use crate::algebra::linear::matrix::Diagonal;

impl<T> Determinant<T> for Diagonal<T>
where
    T: Field + Scalar,
{
    /// Calculates the determinant of the diagonal matrix `self`
    ///
    /// # Arguments
    /// * `self`
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, Determinant};
    //`
    /// let a: Diagonal<f64> = Diagonal::new(&[1.0, -2.0, 3.0, -7.0]);
    /// assert_eq!(42.0, a.det());
    /// ```
    fn det(&self) -> T {
        let mut det = T::one();

        for i in 0..self.matrix.m {
            det *= self[[i, i]];
        }
        det
    }
}
