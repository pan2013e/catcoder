use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};
use std::ops::Sub;

impl<T> Sub<Self> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Subtracts two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    ///
    /// let c: General<f64> = a - b;
    /// ```
    fn sub(mut self, rhs: Self) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(x, y)| *x -= *y);
        self
    }
}

impl<'a, 'b, T> Sub<&'b General<T>> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Subtracts two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    ///
    /// let c: General<f64> = &b - &a;
    /// ```
    fn sub(self, rhs: &'b General<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        let (m, n) = self.dim();
        General {
            m,
            n,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(x, y)| *x - *y)
                .collect::<Vec<T>>(),
        }
    }
}

impl<'a, 'b, T> Sub<&'b General<T>> for &'a mut General<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut General<T>;

    /// Subtracts two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    ///
    /// let _ = &mut a - &b;
    /// ```
    fn sub(self, rhs: &'b General<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(x, y)| *x -= *y);
        self
    }
}

/// Subtracts scalar from all matrix elements
impl<'a, 'b, T> Sub<&'b T> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Subtracts a scalar value from all matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = &a - &-4.0;
    /// ```
    fn sub(self, rhs: &'b T) -> Self::Output {
        self.apply(&|x: &T| -> T { *x - *rhs })
    }
}

impl<'a, T> Sub<&'a T> for &'a mut General<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut General<T>;

    /// Subtracts a scalar value from all matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b = &mut a - &-4.0;
    /// ```
    fn sub(self, rhs: &'a T) -> Self::Output {
        self.data.iter_mut().for_each(&|x: &mut T| *x -= *rhs);
        self
    }
}

impl<T> Sub<T> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Subtracts a scalar from all matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = a - -4.0;
    /// ```
    fn sub(mut self, rhs: T) -> Self::Output {
        let _ = (&mut self).sub(&rhs);
        self
    }
}
