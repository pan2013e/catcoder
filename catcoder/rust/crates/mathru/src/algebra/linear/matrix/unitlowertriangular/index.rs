use crate::algebra::linear::matrix::UnitLowerTriangular;
use std::ops::Index;
use std::ops::IndexMut;

impl<T> Index<[usize; 2]> for UnitLowerTriangular<T> {
    type Output = T;

    /// Gets the element in the matrix
    ///
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UnitLowerTriangular};
    /// use mathru::matrix;
    ///
    /// let m: UnitLowerTriangular<f64> = matrix![  1.0, 0.0;
    ///                                         3.0, 1.0].into();
    ///
    /// assert_eq!(3.0, m[[1, 0]])
    /// ```
    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.matrix[index]
    }
}

impl<T> IndexMut<[usize; 2]> for UnitLowerTriangular<T> {
    /// Sets the element in the matrix
    ///
    ///
    /// # Example
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UnitLowerTriangular};
    /// use mathru::matrix;
    ///
    /// let mut m: UnitLowerTriangular<f64> = matrix![  1.0, 0.0;
    ///                                                 3.0, 1.0].into();
    ///
    /// m[[1, 0]] = -2.0;
    ///
    /// assert_eq!(-2.0, m[[1, 0]])
    /// ```
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        assert!(index[1] < index[0]);
        &mut self.matrix[index]
    }
}
