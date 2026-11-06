use std::ops::{Index, IndexMut};

use crate::algebra::linear::vector::Vector;

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    /// Returns the component
    ///
    /// # Panics
    ///
    /// if index is out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::vector::Vector;
    ///
    /// let mut a: Vector<f64> = Vector::new_row(vec![1.0, 0.0, 3.0, -2.0]);
    ///
    /// assert_eq!(-2.0, a[3])
    /// ```
    fn index(&self, index: usize) -> &Self::Output {
        let num_rows = self.data.m;

        return if num_rows == 1 {
            debug_assert!(
                index < self.data.n,
                "index out of bounds: is a 1x{} vector but the index is 0x{}",
                self.data.n,
                index
            );
            &self.data[[0, index]]
        } else {
            debug_assert!(
                index < num_rows,
                "index out of bounds: is a {}x1 vector but the index is {}x0",
                num_rows,
                index
            );
            &self.data[[index, 0]]
        };
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    /// Returns the component
    ///
    /// # Panics
    ///
    /// if index is out of bounds
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::vector::Vector;
    ///
    /// let mut a: Vector<f64> = Vector::new_row(vec![1.0, 0.0, 3.0, -2.0]);
    ///
    /// assert_eq!(-2.0, a[3])
    /// ```
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let num_rows = self.data.m;

        return if num_rows == 1 {
            debug_assert!(
                index < self.data.n,
                "index out of bounds: is a 1x{} vector but the index is 0x{}",
                self.data.n,
                index
            );
            &mut self.data[[0, index]]
        } else {
            debug_assert!(
                index < num_rows,
                "index out of bounds: is a {}x1 vector but the index is {}x0",
                num_rows,
                index
            );
            &mut self.data[[index, 0]]
        };
    }
}
