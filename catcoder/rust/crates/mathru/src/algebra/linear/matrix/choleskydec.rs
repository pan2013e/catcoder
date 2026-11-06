//! Cholesky decomposition
//!
//! The Cholesky decomposition of a Hermitian positive definite matrix $A$ is
//! a product of a lower triangular matrix $L$ and its conjugate transpose.
//! Expressed in symbols that is $A = LL^*$.
//!
//! Here is an example on how to compute the Cholesky decomposition of a real
//! valued matrix:
//! ```
//! use mathru::algebra::linear::matrix::{General, LowerTriangular};
//! use mathru::matrix;
//!
//! let a: General<f64> = matrix![   2.0, -1.0, 0.0;
//!                                -1.0, 2.0, -1.0;
//!                                 0.0, -1.0,  2.0];
//! let l: LowerTriangular<f64> = a.dec_cholesky().unwrap().l();
//! ```

use super::LowerTriangular;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

/// Result of a Cholesky decomposition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct CholeskyDec<T> {
    l: LowerTriangular<T>,
}

impl<T> CholeskyDec<T> {
    pub fn new(l: LowerTriangular<T>) -> CholeskyDec<T> {
        CholeskyDec { l }
    }
}

impl<T> CholeskyDec<T> {
    /// Return the l matrix
    pub fn l(self) -> LowerTriangular<T> {
        self.l
    }
}
