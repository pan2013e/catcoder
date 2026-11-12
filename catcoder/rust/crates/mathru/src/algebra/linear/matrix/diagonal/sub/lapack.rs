use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::Diagonal,
};
use std::ops::Sub;

impl<T> Sub<Diagonal<T>> for Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Subtracts two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![ 1.0, 0.0;
    ///                                 0.0, -7.0].into();
    /// let b: Diagonal<f64> = matrix![ 1.0, 0.0;
    ///                                 0.0, 2.0].into();
    /// let sum: Diagonal<f64> = matrix![0.0, 0.0;
    ///                                  0.0, -9.0].into();
    ///
    /// let c: Diagonal<f64> = a - b;
    /// assert_eq!(sum, c);
    /// ```
    fn sub(self, rhs: Self) -> Self::Output {
        self.matrix.sub(rhs.matrix).into()
    }
}

///
///Subs two matrices
impl<'a, 'b, T> Sub<&'b Diagonal<T>> for &'a Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Subtracts two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![ 1.0, 0.0;
    ///                                 0.0, -7.0].into();
    /// let b: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                0.0, 2.0].into();
    /// let sum: Diagonal<f64> = matrix![0.0, 0.0;
    ///                                  0.0, -9.0].into();
    ///
    /// let c: Diagonal<f64> = &a - &b;
    /// assert_eq!(sum, c)
    /// ```
    fn sub(self, rhs: &'b Diagonal<T>) -> Self::Output {
        Sub::sub(&self.matrix, &rhs.matrix).into()
    }
}

impl<'a, 'b, T> Sub<&'b Diagonal<T>> for &'a mut Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut Diagonal<T>;

    /// Subtracts two matrices
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
    /// let sum: Diagonal<f64> = matrix![0.0, 0.0;
    ///                                  0.0, -9.0].into();
    ///
    /// let c = &a - &b;
    /// assert_eq!(sum, c)
    /// ```
    fn sub(self, rhs: &'b Diagonal<T>) -> Self::Output {
        let _ = Sub::sub(&mut self.matrix, &rhs.matrix);
        self
    }
}

impl<T> Sub<T> for Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Subtracts a scalar to all diagonal matrix elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                0.0, -7.0].into();
    /// let b: Diagonal<f64> = a - -4.0;
    /// ```
    fn sub(self, rhs: T) -> Self::Output {
        Sub::sub(self.matrix, rhs).into()
    }
}

impl<'a, 'b, T> Sub<&'b T> for &'a Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Subtracts a scalar value to all diagonal elements
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let a: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                0.0, -7.0].into();
    /// let b: Diagonal<f64> = &a - &-4.0;
    /// let sum: Diagonal<f64> = matrix![5.0, 0.0;
    ///                                  0.0, -3.0].into();
    /// assert_eq!(b, sum);
    /// ```
    fn sub(self, rhs: &T) -> Self::Output {
        self.apply(&|x: &T| -> T { *x - *rhs })
    }
}

impl<'a, 'b, T> Sub<&'b T> for &'a mut Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut Diagonal<T>;

    /// Sub a scalar to the matrix
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Diagonal, General};
    /// use mathru::matrix;
    ///
    /// let mut a: Diagonal<f64> = matrix![1.0, 0.0;
    ///                                    0.0, -7.0].into();
    /// let b = &mut a - &-4.0;
    /// ```
    fn sub(self, rhs: &T) -> Self::Output {
        let (m, n) = self.dim();
        let k = m.min(n);

        for i in 0..k {
            self[[i, i]] -= *rhs;
        }
        self
    }
}
