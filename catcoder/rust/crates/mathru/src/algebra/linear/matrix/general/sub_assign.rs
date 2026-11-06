use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};
use std::ops::SubAssign;

impl<T> SubAssign<General<T>> for General<T>
where
    T: Field + Scalar,
{
    /// Subtract the matrix rhs from matrix self and assign the difference to self
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![2.0, 3.0, -5.0, 2.0]);
    /// a -= b;
    /// ```
    fn sub_assign(&mut self, rhs: General<T>) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(a, b)| *a -= *b)
    }
}

impl<T> SubAssign<T> for General<T>
where
    T: Field + Scalar,
{
    /// Subtract the scalar rhs from matrix self and assign the difference to self
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// a += -4.0;
    /// ```
    fn sub_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a: &mut T| *a -= rhs);
    }
}
