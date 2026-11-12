use crate::algebra::linear::matrix::{General, UpperHessenberg};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

pub trait HessenbergDecomposition<T> {
    fn dec_hessenberg(&self) -> HessenbergDec<T>;
}

/// Result of a UpperHessenberg decomposition
/// ```math
/// A = Q H Q^H
/// ```
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct HessenbergDec<T> {
    q: General<T>,
    h: UpperHessenberg<T>,
}

impl<T> HessenbergDec<T> {
    pub(super) fn new(q: General<T>, h: UpperHessenberg<T>) -> HessenbergDec<T> {
        HessenbergDec { q, h }
    }

    pub fn q(self) -> General<T> {
        self.q
    }

    pub fn h(self) -> UpperHessenberg<T> {
        self.h
    }

    pub fn qh(self) -> (General<T>, UpperHessenberg<T>) {
        (self.q, self.h)
    }
}
