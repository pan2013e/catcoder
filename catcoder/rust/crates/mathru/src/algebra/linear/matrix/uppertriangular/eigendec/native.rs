use crate::algebra::abstr::{AbsDiffEq, Field, Scalar};
use crate::algebra::linear::matrix::Diagonal;
use crate::algebra::linear::matrix::{EigenDec, EigenDecomposition, General, UpperTriangular};
use crate::elementary::Power;

impl<T> EigenDecomposition<T> for UpperTriangular<T>
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
    /// use mathru::algebra::linear::matrix::{EigenDec, EigenDecomposition, General, UpperTriangular};
    /// use mathru::matrix;
    ///
    /// let a: UpperTriangular<f64> = matrix![  -3.0, 3.0, 6.0;
    ///                                         0.0, -5.0, -6.0;
    ///                                         0.0, 0.0, 4.0].into();
    ///
    /// let eigen: EigenDec<f64> = a.dec_eigen().unwrap();
    /// ```
    fn dec_eigen(&self) -> Result<EigenDec<T>, String> {
        let (m, _): (usize, usize) = self.dim();

        let mut eigen_values = Vec::with_capacity(m);
        for i in 0..m {
            eigen_values.push(self[[i, i]]);
        }
        let values: Diagonal<T> = Diagonal::new(&eigen_values);
        let vectors: General<T> = self.calc_eigenvector(&eigen_values);

        Ok(EigenDec::new(values, vectors))
    }
}

impl<T> UpperTriangular<T>
where
    T: Field + Scalar + Power + AbsDiffEq<Epsilon = T>,
{
    pub fn calc_eigenvector(self: &Self, eigen_values: &Vec<T>) -> General<T> {
        let mut x: General<T> = General::zero(eigen_values.len(), eigen_values.len());

        for (idx, lambda) in eigen_values.iter().enumerate().rev() {
            x[[idx, idx]] = T::one();

            for n in (0..idx).rev() {
                let mut sum = T::zero();

                for i in n + 1..eigen_values.len() {
                    sum += self.matrix[[n, i]] * x[[i, idx]];
                }

                let divisor = *lambda - self.matrix[[n, n]];

                x[[n, idx]] = sum / divisor;
            }
        }

        UpperTriangular::norm(x)
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
