use crate::algebra::linear::matrix::General;
use core::fmt::{self, Display};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Upper Hessenberg matrix
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct UpperHessenberg<T> {
    pub(crate) matrix: General<T>,
}

impl<T> UpperHessenberg<T> {
    pub fn new(matrix: General<T>) -> UpperHessenberg<T> {
        UpperHessenberg { matrix }
    }
}

impl<T> Display for UpperHessenberg<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.matrix.fmt(f)
    }
}
