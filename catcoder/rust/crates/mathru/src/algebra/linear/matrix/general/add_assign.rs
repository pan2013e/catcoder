use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};
use std::ops::AddAssign;

impl<T> AddAssign<General<T>> for General<T>
where
    T: Field + Scalar,
{
    /// Add the matrix rhs to matrix self and assign the sum to self
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![2.0, 3.0, -5.0, 2.0]);
    /// a += b;
    /// ```
    fn add_assign(&mut self, rhs: General<T>) {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(a, b)| *a += *b)
    }
}

impl<T> AddAssign<T> for General<T>
where
    T: Field + Scalar,
{
    /// Add a scalar rhs to matrix self and assign result to self
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// a += -4.0;
    /// ```
    fn add_assign(&mut self, rhs: T) {
        self.data.iter_mut().for_each(|a: &mut T| *a += rhs);
    }
}
