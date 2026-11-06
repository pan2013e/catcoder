use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::abstr::Field;
use crate::algebra::abstr::Scalar;
use crate::algebra::linear::matrix::substitute::SubstituteForward;
use crate::algebra::linear::matrix::{General, LowerTriangular};
use crate::algebra::linear::vector::Vector;

impl<T> SubstituteForward<Vector<T>> for LowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    fn substitute_forward(&self, a: Vector<T>) -> Result<Vector<T>, ()> {
        let mut b: Vector<T> = a;
        let rows = self.matrix.nrows();

        for k in 0..rows {
            for l in 0..k {
                b[k] = b[k] - self[[k, l]] * b[l];
            }
            if self[[k, k]] == T::zero() {
                if b[k] != T::zero() {
                    return Err(());
                } else {
                    b[k] = T::one();
                }
            } else {
                b[k] /= self[[k, k]];
            }
        }

        Ok(b)
    }
}

impl<T> SubstituteForward<General<T>> for LowerTriangular<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    fn substitute_forward(&self, a: General<T>) -> Result<General<T>, ()> {
        let mut b: General<T> = a;
        let rows = self.matrix.nrows();

        for k in 0..rows {
            for l in 0..k {
                b.set_row(&(b.get_row(k) - (b.get_row(l) * self[[k, l]])), k);
            }
            if self[[k, k]] == T::zero() {
                return Err(());
            }
            b.set_row(&(b.get_row(k) / self[[k, k]]), k);
        }

        Ok(b)
    }
}
