use super::General;

impl<T> PartialEq for General<T>
where
    T: PartialEq,
{
    /// Checks if two matrices are equal
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    ///
    /// assert_eq!(true, a == b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        if self.dim() != other.dim() {
            return false;
        }

        if self.data == other.data {
            return true;
        }

        false
    }
}
