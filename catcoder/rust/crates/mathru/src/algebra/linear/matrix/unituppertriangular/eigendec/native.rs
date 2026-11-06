use crate::algebra::abstr::{AbsDiffEq, Field, Scalar};
use crate::algebra::linear::matrix::{Diagonal, UnitUpperTriangular};
use crate::algebra::linear::matrix::{EigenDec, EigenDecomposition, General};
use crate::elementary::Power;

impl<T> EigenDecomposition<T> for UnitUpperTriangular<T>
where
    T: Field + Scalar + Power + AbsDiffEq<Epsilon = T>,
{
    /// Computes the eigen decomposition
    ///
    /// # Arguments
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{EigenDec, EigenDecomposition, General, UnitUpperTriangular};
    /// use mathru::{matrix, assert_relative_eq};
    ///
    /// let a: UnitUpperTriangular<f64> = matrix![  1.0, 3.0, 6.0;
    ///                                             0.0, 1.0, -6.0;
    ///                                             0.0, 0.0, 1.0].into();
    ///
    /// let (values, vectors) = a.dec_eigen().unwrap().pair();
    ///
    /// assert_relative_eq!(
    ///    &Into::<General<f64>>::into(a) * &vectors,
    ///    &vectors * &Into::<General<f64>>::into(values),
    ///    epsilon = 1.0e-5
    /// );
    /// ```
    fn dec_eigen(&self) -> Result<EigenDec<T>, String> {
        let (m, _): (usize, usize) = self.dim();
        let eigen_values = vec![T::one(); m];
        let values: Diagonal<T> = Diagonal::new(&eigen_values);

        let vectors: General<T> = self.calc_eigenvector(&eigen_values);
        Ok(EigenDec::new(values, vectors))
    }
}

impl<T> UnitUpperTriangular<T>
where
    T: Field + Scalar + Power + AbsDiffEq<Epsilon = T>,
{
    pub fn calc_eigenvector(self: &Self, eigen_values: &Vec<T>) -> General<T> {
        let mut x: General<T> = General::zero(eigen_values.len(), eigen_values.len());

        for (lambda_idx, lambda) in eigen_values.iter().rev().enumerate() {
            for row in (0..eigen_values.len()).rev() {
                let sum = (row..self.matrix.n).fold(T::zero(), |s, i| {
                    s + self.matrix[[row, i]] * x[[i, lambda_idx]]
                });

                let divisor = *lambda - self.matrix[[row, row]];
                if divisor != T::zero() {
                    x[[row, lambda_idx]] = sum / divisor;
                } else {
                    if row == 0 {
                        x[[row, lambda_idx]] = T::one();
                    } else {
                        x[[row, lambda_idx]] = T::zero();
                    }
                }
            }
        }

        UnitUpperTriangular::norm(x)
    }

    fn norm(mut m: General<T>) -> General<T> {
        let (rows, cols) = m.dim();
        for c in 0..cols {
            let norm = m.get_column(c).p_norm(&T::from_f32(2.0));

            for r in 0..rows {
                m[[r, c]] /= norm;
            }
        }

        m
    }
}
