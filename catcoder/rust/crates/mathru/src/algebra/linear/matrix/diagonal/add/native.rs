use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::Diagonal,
};
use std::ops::Add;

impl<T> Add<Diagonal<T>> for Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Adds two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![ 1.0, 0.0;
    ///                                 0.0, -7.0].into();
    /// let b: Diagonal<f64> = matrix![  1.0, 0.0;
    ///                                 0.0, 2.0].into();
    /// let sum: Diagonal<f64> = matrix![2.0, 0.0;
    ///                                 0.0, -5.0].into();
    ///
    /// let c: Diagonal<f64> = a + b;
    /// assert_eq!(sum, c);
    /// ```
    fn add(mut self, rhs: Self) -> Self::Output {
        let _ = (&mut self).add(&rhs);
        self
    }
}

impl<'a, 'b, T> Add<&'b Diagonal<T>> for &'a Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Adds two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![ 1.0, 0.0;
    ///                                 0.0, -7.0].into();
    /// let b: Diagonal<f64> = matrix![  1.0, 0.0;
    ///                                 0.0, 2.0].into();
    /// let sum: Diagonal<f64> = matrix![2.0, 0.0;
    ///                                  0.0, -5.0].into();
    ///
    /// let c: Diagonal<f64> = &b + &a;
    /// assert_eq!(sum, c)
    /// ```
    fn add(self, rhs: &'b Diagonal<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        let (m, n) = self.dim();
        let k = m.min(n);

        let mut this = self.clone();
        for i in 0..k {
            this[[i, i]] += rhs[[i, i]];
        }
        this
    }
}

impl<'a, 'b, T> Add<&'b Diagonal<T>> for &'a mut Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut Diagonal<T>;

    /// Adds two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let mut a: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                    0.0, -7.0].into();
    /// let b: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                0.0, 2.0].into();
    /// let sum: Diagonal<f64> = matrix![2.0, 0.0;
    ///                                  0.0, -5.0].into();
    ///
    /// let c = &mut a + &b;
    /// assert_eq!(&sum, c)
    /// ```
    fn add(self, rhs: &'b Diagonal<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());
        let (m, n) = self.dim();
        let k = m.min(n);
        for i in 0..k {
            self[[i, i]] += rhs[[i, i]];
        }
        self
    }
}

/// Add scalar to matrix
impl<'a, 'b, T> Add<&'b T> for &'a Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Add a scalar value to all diagonal elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                0.0, -7.0].into();
    /// let b: Diagonal<f64> = &a + &-4.0;
    /// let sum: Diagonal<f64> = matrix![-3.0, 0.0;
    ///                                      0.0, -11.0].into();
    /// assert_eq!(b, sum);
    /// ```
    fn add(self, rhs: &'b T) -> Self::Output {
        self.apply(&|x: &T| -> T { *x + *rhs })
    }
}

impl<'a, T> Add<&'a T> for &'a mut Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut Diagonal<T>;

    /// Adds a scalar to all diagonal matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let mut a: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                    0.0, -4.0].into();
    /// let b = &mut a + &-4.0;
    /// ```
    fn add(self, rhs: &'a T) -> Self::Output {
        let (m, n) = self.dim();
        let k = m.min(n);
        for i in 0..k {
            self[[i, i]] += *rhs
        }
        self
    }
}

impl<T> Add<T> for Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Adds a scalar to all diagonal matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![ 1.0, 0.0;
    ///                                 0.0, -7.0].into();
    /// let b: Diagonal<f64> = a + -4.0;
    /// ```
    fn add(mut self, rhs: T) -> Self::Output {
        let _ = (&mut self).add(&rhs);
        self
    }
}
