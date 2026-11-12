use crate::algebra::linear::matrix::UpperHessenberg;

impl<T> PartialEq for UpperHessenberg<T>
where
    T: PartialEq,
{
    /// Checks if two matrices are equal
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UpperHessenberg};
    /// use mathru::matrix;
    ///
    /// let a: UpperHessenberg<f64> = matrix![1.0, 5.0; 0.0, 1.0].into();
    /// let b: UpperHessenberg<f64> = matrix![1.0, 5.0; 0.0, 1.0].into();
    ///
    /// assert_eq!(true, a == b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        self.matrix.eq(&other.matrix)
    }
}
