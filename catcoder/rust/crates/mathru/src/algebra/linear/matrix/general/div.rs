use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};
use std::ops::Div;

//Divides all  matrix elements with a scalar
impl<T> Div<T> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Divides all matrix element with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let res_ref: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let f: f64 = 7.0;
    /// let a: General<f64> = General::new(2, 2, vec![7.0, 0.0, 21.0, -49.0]);
    /// ```
    fn div(self, m: T) -> General<T> {
        self * (T::one() / m)
    }
}

impl<'a, 'b, T> Div<&'b T> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Divide all matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let res_ref: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let a: General<f64> = General::new(2, 2, vec![4.0, 0.0, 12.0, -28.0]);
    /// ```
    fn div(self, m: &'b T) -> General<T> {
        self.clone() * (T::one() / *m)
    }
}
