use crate::algebra::linear::matrix::UpperTriangular;
use std::ops::Index;
use std::ops::IndexMut;

impl<T> Index<[usize; 2]> for UpperTriangular<T> {
    type Output = T;

    /// Gets the element in the matrix
    ///
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UpperTriangular};
    /// use mathru::matrix;
    ///
    /// let m: UpperTriangular<f64> = matrix![  -7.0, 3.0;
    ///                                         0.0, -5.0].into();
    ///
    /// assert_eq!(3.0, m[[0, 1]])
    /// ```
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.matrix[index]
    }
}

impl<T> IndexMut<[usize; 2]> for UpperTriangular<T> {
    /// Sets the element in the matrix
    ///
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UpperTriangular};
    /// use mathru::matrix;
    ///
    /// let mut m: UpperTriangular<f64> = matrix![  -7.0, 3.0;
    ///                                             0.0, -5.0].into();
    ///
    /// m[[0, 1]] = -2.0;
    ///
    /// assert_eq!(-2.0, m[[0, 1]])
    /// ```
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        debug_assert!(index[1] >= index[0]);
        &mut self.matrix[index]
    }
}
