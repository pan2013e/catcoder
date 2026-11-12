use crate::algebra::linear::matrix::Diagonal;

impl<T> PartialEq for Diagonal<T>
where
    T: PartialEq,
{
    /// Checks if two diagonal matrices are equal
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, Diagonal};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![1.0, 0.0; 0.0, 5.0].into();
    /// let b: Diagonal<f64> = matrix![1.0, 0.0; 0.0, 5.0].into();
    ///
    /// assert_eq!(a ,b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        for i in 0..self.matrix.m {
            if self[[i, i]] != other[[i, i]] {
                return false;
            }
        }
        true
    }
}
