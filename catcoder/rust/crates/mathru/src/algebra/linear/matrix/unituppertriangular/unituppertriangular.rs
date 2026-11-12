use crate::algebra::linear::matrix::General;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Upper triangular matrix with unit diagonal
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct UnitUpperTriangular<T> {
    pub(crate) matrix: General<T>,
}

impl<T> UnitUpperTriangular<T> {
    pub fn new(matrix: General<T>) -> UnitUpperTriangular<T> {
        UnitUpperTriangular { matrix }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.matrix.dim()
    }
}
