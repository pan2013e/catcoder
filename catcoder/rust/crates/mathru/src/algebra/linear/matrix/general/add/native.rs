use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};
use std::ops::Add;

impl<T> Add<General<T>> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Adds two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let sum: General<f64> = General::new(2, 2, vec![2.0, 0.0, 6.0, -14.0]);
    ///
    /// let c: General<f64> = a + b;
    /// assert_eq!(sum, c);
    /// ```
    fn add(mut self, rhs: Self) -> Self::Output {
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(x, y)| *x += *y);
        self
    }
}

impl<'a, 'b, T> Add<&'b General<T>> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Adds two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let sum: General<f64> = General::new(2, 2, vec![2.0, 0.0, 6.0, -14.0]);
    ///
    /// let c: General<f64> = &b + &a;
    /// assert_eq!(sum, c)
    /// ```
    fn add(self, rhs: &'b General<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        let (m, n) = self.dim();
        General {
            m,
            n,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(x, y)| *x + *y)
                .collect::<Vec<T>>(),
        }
    }
}

impl<'a, 'b, T> Add<&'b General<T>> for &'a mut General<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut General<T>;

    /// Adds two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let sum: General<f64> = General::new(2, 2, vec![2.0, 0.0, 6.0, -14.0]);
    ///
    /// let c = &mut a + &b;
    /// assert_eq!(&sum, c)
    /// ```
    fn add(self, rhs: &'b General<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        self.data
            .iter_mut()
            .zip(rhs.data.iter())
            .for_each(|(x, y)| *x += *y);
        self
    }
}

/// Add scalar to matrix
impl<'a, 'b, T> Add<&'b T> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Add a scalar value from all matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = &a + &-4.0;
    /// ```
    fn add(self, rhs: &'b T) -> Self::Output {
        self.apply(&|x: &T| -> T { *x + *rhs })
    }
}

impl<'a, T> Add<&'a T> for &'a mut General<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut General<T>;

    /// Adds a scalar scalar value from all matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b = &mut a + &-4.0;
    /// ```
    fn add(self, rhs: &'a T) -> Self::Output {
        self.data.iter_mut().for_each(&|x: &mut T| *x += *rhs);
        self
    }
}

impl<T> Add<T> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Adds a scalar from all matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = a + -4.0;
    /// ```
    fn add(mut self, rhs: T) -> Self::Output {
        let _ = (&mut self).add(&rhs);
        self
    }
}
