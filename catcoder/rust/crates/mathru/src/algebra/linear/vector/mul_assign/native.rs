use crate::algebra::{
    abstr::{Field, Scalar},
    linear::vector::Vector,
};
use std::ops::MulAssign;

// Multiply vector with scalar
impl<T> MulAssign<T> for Vector<T>
where
    T: Field + Scalar,
{
    /// Multiply vector with scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::vector::Vector;
    ///
    /// let mut a: Vector<f64> = Vector::new_column(vec![1.0, 0.0, 3.0, -7.0]);
    /// a *= -4.0;
    /// ```
    fn mul_assign(&mut self, rhs: T) {
        self.data *= rhs
    }
}
