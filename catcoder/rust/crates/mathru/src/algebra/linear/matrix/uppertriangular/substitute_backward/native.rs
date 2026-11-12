use crate::algebra::abstr::{AbsDiffEq, RelativeEq};
use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::substitute::SubstituteBackward;
use crate::algebra::linear::{
    matrix::{General, UpperTriangular},
    vector::Vector,
};
use crate::relative_eq;

impl<T> SubstituteBackward<Vector<T>> for UpperTriangular<T>
where
    T: Field + Scalar + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    fn substitute_backward(&self, c: Vector<T>) -> Result<Vector<T>, ()> {
        let mut b: Vector<T> = c;
        let rows = self.matrix.nrows();

        for k in (0..rows).rev() {
            for l in (k + 1..rows).rev() {
                b[k] = b[k] - self[[k, l]] * b[l];
            }

            let div = b[k] / self[[k, k]];
            if div.to_f64().is_finite() {
                b[k] = div;
            } else {
                if !relative_eq!(b[k], T::from_f64(0.000000000001) * T::default_epsilon()) {
                    return Err(());
                } else {
                    b[k] = T::one();
                }
            }
        }

        Ok(b)
    }
}

impl<T> SubstituteBackward<General<T>> for UpperTriangular<T>
where
    T: Field + Scalar,
{
    fn substitute_backward(&self, a: General<T>) -> Result<General<T>, ()> {
        let mut b: General<T> = a;
        let rows = self.matrix.nrows();

        for k in (0..rows).rev() {
            for l in (k + 1..rows).rev() {
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
