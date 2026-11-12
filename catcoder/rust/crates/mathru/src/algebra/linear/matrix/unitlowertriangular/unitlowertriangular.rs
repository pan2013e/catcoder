use crate::algebra::linear::matrix::General;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Lower triangular matrix with unit diagonal
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct UnitLowerTriangular<T> {
    pub(crate) matrix: General<T>,
}

impl<T> UnitLowerTriangular<T> {
    pub fn new(matrix: General<T>) -> UnitLowerTriangular<T> {
        UnitLowerTriangular { matrix }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.matrix.dim()
    }
}
