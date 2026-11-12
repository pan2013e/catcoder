use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::RelativeEq;
use crate::algebra::linear::matrix::substitute::SubstituteBackward;
use crate::algebra::linear::matrix::substitute::SubstituteForward;
use crate::algebra::{
    abstr::{Field, Scalar},
    linear::{
        matrix::{General, Inverse, Solve, UnitLowerTriangular, UpperTriangular},
        vector::Vector,
    },
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

/// Result of a LU decomposition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct LUDec<T> {
    l: UnitLowerTriangular<T>,
    u: UpperTriangular<T>,
    p: General<T>,
}

///
/// P * A = L * U
impl<T> LUDec<T> {
    pub(super) fn new(l: UnitLowerTriangular<T>, u: UpperTriangular<T>, p: General<T>) -> LUDec<T> {
        LUDec { l, u, p }
    }

    /// Return l matrix of LU decomposition
    pub fn l(self) -> UnitLowerTriangular<T> {
        self.l
    }

    /// Return u matrix of LU decomposition
    pub fn u(self) -> UpperTriangular<T> {
        self.u
    }

    /// Return p matrix of LU decomposition
    pub fn p(self) -> General<T> {
        self.p
    }

    /// Return l, u, and p matrix of the LU decomposition
    pub fn lup(self) -> (UnitLowerTriangular<T>, UpperTriangular<T>, General<T>) {
        (self.l, self.u, self.p)
    }
}

impl<T> Inverse<T> for LUDec<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    type Output = General<T>;
    /// Inverse Matrix
    ///
    /// PAX = LUX = I
    /// X = (PA)^-1
    /// X = A^-1P^-1
    /// XP = A^-1
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{Inverse, General};
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b_inv: General<f64> = a.inv().unwrap();
    /// ```
    fn inv(&self) -> Result<General<T>, ()> {
        let b = General::one(self.p.nrows());
        let x: General<T> = self.solve(&b)?;
        Ok(x)
    }
}

impl<T> Solve<Vector<T>> for LUDec<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    /// Solves Ax = y
    /// ```math
    /// where A \in R^{m * n}, x \in R^n, y \in R^m
    /// ```
    ///
    /// ```math
    /// P * A = L *U \\
    /// A = P^-1 * L * U = P^T * L * U \\
    /// P^T * L *U x = y \\
    /// L * U x = (P^T)^-1 * y = P * y = b_{hat} \\
    /// L * U * x = b_{hat} \\
    /// L * c = b_hat{x} \\
    /// U * x = c\\
    /// ```
    fn solve(&self, rhs: &Vector<T>) -> Result<Vector<T>, ()> {
        let b_hat: Vector<T> = &self.p * rhs;
        let c: Vector<T> = self.l.substitute_forward(b_hat)?;
        self.u.substitute_backward(c)
    }
}

impl<T> Solve<General<T>> for LUDec<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    fn solve(&self, rhs: &General<T>) -> Result<General<T>, ()> {
        let b_hat: General<T> = &self.p * rhs;

        let c: General<T> = self.l.substitute_forward(b_hat)?;
        self.u.substitute_backward(c)
    }
}
