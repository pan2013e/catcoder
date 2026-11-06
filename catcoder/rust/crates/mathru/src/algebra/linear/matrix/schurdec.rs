use crate::algebra::linear::matrix::{General, UpperTriangular};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

/// Result of Schur decomposition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct SchurDec<T> {
    q: General<T>,
    u: UpperTriangular<T>,
}

///
/// ```math
/// A = Q * U Q ^-1
/// ```
impl<T> SchurDec<T> {
    pub(super) fn new(q: General<T>, u: UpperTriangular<T>) -> SchurDec<T> {
        SchurDec { q, u }
    }

    /// Return the unitary matrix q
    ///
    /// # Arguments
    ///
    /// * `self`
    pub fn q(self) -> General<T> {
        self.q
    }

    /// Return the upper triangular matrix u
    pub fn u(self) -> UpperTriangular<T> {
        self.u
    }

    pub fn qu(self) -> (General<T>, UpperTriangular<T>) {
        (self.q, self.u)
    }
}
