use crate::algebra::linear::matrix::UpperTriangular;

impl<T> PartialEq for UpperTriangular<T>
where
    T: PartialEq,
{
    /// Checks if two matrices are equal
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UpperTriangular};
    /// use mathru::matrix;
    ///
    /// let a: UpperTriangular<f64> = matrix![1.0, 0.0; 3.0, -7.0].into();
    /// let b: UpperTriangular<f64> = matrix![1.0, 0.0; 3.0, -7.0].into();
    ///
    /// assert_eq!(true, a == b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        self.matrix.eq(&other.matrix)
    }
}
