use crate::algebra::linear::matrix::General;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Upper triangular matrix
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct UpperTriangular<T> {
    pub(crate) matrix: General<T>,
}

impl<T> UpperTriangular<T> {
    pub fn new(matrix: General<T>) -> UpperTriangular<T> {
        UpperTriangular { matrix }
    }

    pub fn dim(&self) -> (usize, usize) {
        self.matrix.dim()
    }
}
