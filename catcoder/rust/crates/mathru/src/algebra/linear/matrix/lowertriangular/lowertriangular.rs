use crate::algebra::linear::matrix::General;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LowerTriangular<T> {
    pub(crate) matrix: General<T>,
}

impl<T> LowerTriangular<T> {
    pub fn new(matrix: General<T>) -> LowerTriangular<T> {
        LowerTriangular { matrix }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.matrix.dim()
    }
}

impl<T> PartialEq for LowerTriangular<T>
where
    T: PartialEq,
{
    /// Checks if two matrices are equal
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, LowerTriangular};
    /// use mathru::matrix;
    ///
    /// let a: LowerTriangular<f64> = LowerTriangular::new(matrix![ 1.0, 0.0;
    ///                                                             3.0, -7.0]);
    /// let b: LowerTriangular<f64> = LowerTriangular::new(matrix![ 1.0, 0.0;
    ///                                                             3.0, -7.0]);
    ///
    /// assert_eq!(true, a == b);
    /// ```
    fn eq(&self, other: &Self) -> bool {
        // TODO implement optimized comparison
        self.matrix.eq(&other.matrix)
    }
}
