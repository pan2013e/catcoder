pub trait Inverse<T> {
    type Output;
    /// Inverse of a matrix
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Inverse, General};
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b_inv: General<f64> = a.inv().unwrap();
    /// ```
    fn inv(&self) -> Result<Self::Output, ()>;
}
