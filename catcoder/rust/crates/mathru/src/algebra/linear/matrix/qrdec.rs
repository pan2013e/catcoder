use crate::algebra::linear::matrix::{General, UpperTriangular};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

/// Reulst of QR decomposition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct QRDec<T> {
    q: General<T>,
    r: UpperTriangular<T>,
}

impl<T> QRDec<T> {
    pub(super) fn new(q: General<T>, r: UpperTriangular<T>) -> QRDec<T> {
        QRDec { q, r }
    }

    /// Return the q matrix of the QR decomposition
    ///
    /// # Arguments
    ///
    /// * `self`
    pub fn q(self) -> General<T> {
        self.q
    }

    /// Return the r matrix of the qr decomposition
    pub fn r(self) -> UpperTriangular<T> {
        self.r
    }

    pub fn qr(self) -> (General<T>, UpperTriangular<T>) {
        (self.q, self.r)
    }
}
