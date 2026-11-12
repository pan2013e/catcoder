use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::Diagonal,
};
use std::ops::Mul;

impl<T> Mul<Diagonal<T>> for Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let a: Diagonal<f64> = Diagonal::new(&[1.0, -2.0, 3.0, -7.0]);
    /// let b: Diagonal<f64> = Diagonal::new(&[1.0, 2.0, 3.0, -2.0]);
    /// let res_ref: Diagonal<f64> = Diagonal::new(&[1.0, -4.0, 9.0, 14.0]);
    /// assert_eq!(res_ref, a * b);
    /// ```
    fn mul(mut self, rhs: Self) -> Self::Output {
        let _ = &mut self * &rhs;
        self
    }
}

impl<'a, 'b, T> Mul<&'b Diagonal<T>> for &'a mut Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut Diagonal<T>;
    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let mut a: Diagonal<f64> = Diagonal::new(&[1.0, -2.0, 3.0, -7.0]);
    /// let b: Diagonal<f64> = Diagonal::new(&[1.0, 2.0, 3.0, -2.0]);
    /// let res_ref: Diagonal<f64> = Diagonal::new(&[1.0, -4.0, 9.0, 14.0]);
    /// assert_eq!(res_ref, *(&mut a * &b));
    /// ```
    fn mul(self, rhs: &'b Diagonal<T>) -> Self::Output {
        debug_assert_eq!(self.dim(), rhs.dim());

        let m = self.matrix.m;

        for i in 0..m {
            self[[i, i]] *= rhs[[i, i]];
        }

        self
    }
}

impl<'a, 'b, T> Mul<&'b Diagonal<T>> for &'a Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let mut a: Diagonal<f64> = Diagonal::new(&[1.0, -2.0, 3.0, -7.0]);
    /// let b: Diagonal<f64> = Diagonal::new(&[1.0, 2.0, 3.0, -2.0]);
    /// let res_ref: Diagonal<f64> = Diagonal::new(&[1.0, -4.0, 9.0, 14.0]);
    /// assert_eq!(res_ref, &a * &b);
    /// ```
    fn mul(self, rhs: &'b Diagonal<T>) -> Self::Output {
        let mut this = self.clone();

        let _ = (&mut this).mul(rhs);
        this
    }
}

/// Multiplies matrix by scalar
impl<T> Mul<T> for Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Multiplies a matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let a: Diagonal<f64> = Diagonal::new(&[1.0, 3.0, -7.0]);
    /// let res_ref: Diagonal<f64> = Diagonal::new(&[4.0, 12.0, -28.0]);
    ///
    /// assert_eq!(res_ref, a * 4.0);
    /// ```
    fn mul(mut self, m: T) -> Diagonal<T> {
        let _ = &mut self * &m;
        self
    }
}

// Multiplies matrix by scalar
impl<'a, 'b, T> Mul<&'b T> for &'a Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = Diagonal<T>;

    /// Multiplies a matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let a: Diagonal<f64> = Diagonal::new(&[1.0, 3.0, -7.0]);
    /// let res_ref: Diagonal<f64> = Diagonal::new(&[4.0, 12.0, -28.0]);
    ///
    /// assert_eq!(res_ref, &a * &4.0);
    /// ```
    fn mul(self, rhs: &'b T) -> Diagonal<T> {
        let mut this = self.clone();

        let _ = (&mut this).mul(rhs);
        this
    }
}

//
impl<'a, 'b, T> Mul<&'b T> for &'a mut Diagonal<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut Diagonal<T>;

    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let a: Diagonal<f64> = Diagonal::new(&[1.0, 3.0, -7.0]);
    /// let res_ref: Diagonal<f64> = Diagonal::new(&[4.0, 12.0, -28.0]);
    ///
    /// assert_eq!(res_ref, &a * &4.0);
    /// ```
    fn mul(self, rhs: &'b T) -> Self::Output {
        let m = self.matrix.m;

        for i in 0..m {
            self[[i, i]] *= *rhs;
        }

        self
    }
}
