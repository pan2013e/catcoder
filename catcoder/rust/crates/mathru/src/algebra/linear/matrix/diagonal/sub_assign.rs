use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::Diagonal,
};
use std::ops::SubAssign;

impl<T> SubAssign<Diagonal<T>> for Diagonal<T>
where
    T: Field + Scalar,
{
    /// Subtracts the matrix rhs from the matrix self and assign the difference to self
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let mut a: Diagonal<f64> = Diagonal::new(&vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: Diagonal<f64> = Diagonal::new(&vec![2.0, 3.0, -5.0, 2.0]);
    /// a -= b;
    /// ```
    fn sub_assign(&mut self, rhs: Diagonal<T>) {
        debug_assert!(self.dim() == rhs.dim());
        for i in 0..self.matrix.m {
            self.matrix[[i, i]] -= rhs.matrix[[i, i]];
        }
    }
}

impl<T> SubAssign<T> for Diagonal<T>
where
    T: Field + Scalar,
{
    /// Subtracts a scalar rhs from the diagonal of diagonal matrix self and assigns result to self
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::Diagonal;
    ///
    /// let mut a: Diagonal<f64> = Diagonal::new(&vec![1.0, 0.0, 3.0, -7.0]);
    /// a -= -4.0;
    /// ```
    fn sub_assign(&mut self, rhs: T) {
        self.matrix.data.iter_mut().for_each(|a: &mut T| *a -= rhs);
    }
}
