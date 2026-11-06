use crate::algebra::linear::matrix::{Diagonal, General};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

pub trait EigenDecomposition<T> {
    fn dec_eigen(&self) -> Result<EigenDec<T>, String>;
}

/// Result of a Eigen decomposition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct EigenDec<T> {
    values: Diagonal<T>,
    vectors: General<T>,
}

impl<T> EigenDec<T> {
    pub(super) fn new(values: Diagonal<T>, vectors: General<T>) -> EigenDec<T> {
        EigenDec { values, vectors }
    }

    pub fn values(self) -> Diagonal<T> {
        self.values
    }

    pub fn vectors(self) -> General<T> {
        self.vectors
    }

    pub fn pair(self) -> (Diagonal<T>, General<T>) {
        (self.values, self.vectors)
    }
}
