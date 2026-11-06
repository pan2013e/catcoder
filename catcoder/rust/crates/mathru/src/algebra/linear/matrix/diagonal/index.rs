use crate::algebra::linear::matrix::Diagonal;
use std::ops::Index;
use std::ops::IndexMut;

impl<T> Index<[usize; 2]> for Diagonal<T> {
    type Output = T;

    /// Gets the element in the matrix
    ///
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::{General, Diagonal};
    /// use mathru::matrix;
    ///
    /// let m: Diagonal<f64> = matrix![ -7.0, 3.0;
    ///                                 0.0, -5.0].into();
    ///
    /// assert_eq!(-5.0, m[[1, 1]])
    /// ```
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.matrix[index]
    }
}

impl<T> IndexMut<[usize; 2]> for Diagonal<T> {
    /// Sets the element in the matrix
    ///
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::{General, Diagonal};
    /// use mathru::matrix;
    ///
    /// let mut m: Diagonal<f64> = matrix![ -7.0, 0.0;
    ///                                     0.0, -5.0].into();
    ///
    /// m[[1, 1]] = -2.0;
    ///
    /// assert_eq!(-2.0, m[[1, 1]])
    /// ```
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        debug_assert!(index[1] >= index[0]);
        &mut self.matrix[index]
    }
}
