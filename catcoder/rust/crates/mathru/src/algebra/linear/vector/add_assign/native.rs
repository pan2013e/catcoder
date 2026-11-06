use crate::algebra::{
    abstr::{Field, Scalar},
    linear::vector::Vector,
};
use std::ops::AddAssign;

impl<T> AddAssign<Vector<T>> for Vector<T>
where
    T: Field + Scalar,
{
    /// Add a vector rhs to the vector self and assign the sum to self
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::vector::Vector;
    ///
    /// let mut a: Vector<f64> = Vector::new_column(vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: Vector<f64> = Vector::new_column(vec![2.0, 3.0, -5.0, 2.0]);
    /// a += b;
    /// ```
    fn add_assign(&mut self, rhs: Vector<T>) {
        self.data += rhs.data
    }
}

impl<T> AddAssign<T> for Vector<T>
where
    T: Field + Scalar,
{
    /// Add a vector rhs to the vector self and assign the sum to self
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::vector::Vector;
    ///
    /// let mut a: Vector<f64> = Vector::new_column(vec![1.0, 0.0, 3.0, -7.0]);
    /// a += -4.0;
    /// ```
    fn add_assign(&mut self, rhs: T) {
        self.data += rhs
    }
}
