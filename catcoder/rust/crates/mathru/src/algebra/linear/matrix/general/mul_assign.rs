use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};
use std::ops::MulAssign;

// Multiply matrix with a matrix
impl<T> MulAssign<General<T>> for General<T>
where
    T: Field + Scalar,
{
    /// Multiply a matrix with a matrix
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![2.0, 3.0, -5.0, 2.0]);
    /// a *= b;
    /// ```
    fn mul_assign(&mut self, rhs: General<T>) {
        let _ = self * &rhs;
    }
}

// Multiply matrix with a scalar
impl<T> MulAssign<T> for General<T>
where
    T: Field + Scalar,
{
    /// Multiply matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// a *= -4.0;
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a: &mut T| *a *= rhs);
    }
}
