use crate::algebra::abstr::Zero;
use crate::algebra::abstr::{Complex, Real, Scalar};
use crate::{
    algebra::linear::matrix::{CholeskyDec, General, LowerTriangular},
    elementary::Power,
};

impl<T> General<T>
where
    T: Real,
{
    /// Decomposes a symmetric, positive definite matrix $A$ into a product of
    /// a lower triangular matrix $L$ and its transpose such that $A = LL^T$.
    ///
    /// # Panics
    ///
    /// If the matrix $A$ is not quadratic or not positive definite.
    ///
    /// For efficiency reasons, the function may not check, if the matrix is
    /// symmetric, but just assume so.
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, LowerTriangular};
    /// use mathru::{matrix, assert_abs_diff_eq};
    ///
    /// let a: General<f64> = matrix![   2.0, -1.0, 0.0;
    ///                                -1.0, 2.0, -1.0;
    ///                                 0.0, -1.0,  2.0];
    ///
    /// let l: LowerTriangular<f64> = a.dec_cholesky().unwrap().l();
    ///
    /// let l_ref: LowerTriangular<f64> = matrix![  1.4142, 0.0, 0.0;
    ///                                             -0.7071, 1.2247, 0.0;
    ///                                             0.0, -0.8165, 1.1547].into();
    ///
    /// assert_abs_diff_eq!(l_ref, l, epsilon=0.001);
    /// ```
    pub fn dec_cholesky(&self) -> Result<CholeskyDec<T>, String> {
        let (m, n) = self.dim();
        debug_assert_eq!(m, n);
        debug_assert_ne!(m, 0);

        let mut l: General<T> = General::zero(m, n);

        for i in 0..n {
            for j in 0..i + 1 {
                let mut sum = T::zero();
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    assert!(
                        self[[i, i]] - sum >= T::zero(),
                        "The matrix is not positive definite."
                    );
                    l[[i, j]] = (self[[i, i]] - sum).sqrt();
                } else {
                    assert_ne!(l[[j, j]], T::zero(), "The matrix is not positive definite");
                    l[[i, j]] = (self[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        Ok(CholeskyDec::new(LowerTriangular::new(l)))
    }
}

impl<T> General<Complex<T>>
where
    T: Real,
    Complex<T>: Scalar,
{
    /// Decomposes a Hermitian, positive definite matrix $A$ into a product of
    /// a lower triangular matrix $L$ and its conjugate transpose, such that
    /// $A = LL^*$.
    ///
    /// # Panics
    ///
    /// If the matrix $A$ is not quadratic or not positive definite.
    ///
    /// For efficiency reasons, the function may not check if the matrix is
    /// Hermitian, but just assume so.
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    /// use mathru::algebra::linear::matrix::LowerTriangular;
    /// use mathru::matrix;
    ///
    /// let a: General<f64> = matrix![   2.0, -1.0, 0.0;
    ///                                -1.0, 2.0, -1.0;
    ///                                 0.0, -1.0,  2.0];
    ///
    /// let l: LowerTriangular<f64> = a.dec_cholesky().unwrap().l();
    /// ```
    pub fn dec_cholesky(&self) -> Result<CholeskyDec<Complex<T>>, String> {
        let (m, n) = self.dim();
        debug_assert_eq!(m, n);

        let (m, n) = self.dim();
        let mut l: General<Complex<T>> = General::zero(m, n);

        for i in 0..n {
            for j in 0..i + 1 {
                let mut sum = Complex::<T>::zero();
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]].conj();
                }

                if i == j {
                    assert!(
                        (self[[i, i]] - sum).re >= T::zero(),
                        "The matrix is not positive definite."
                    );
                    assert!(
                        (self[[i, i]] - sum).im == T::zero(),
                        "The matrix is not Hermitian."
                    );
                    l[[i, j]] = (self[[i, i]] - sum).sqrt();
                } else {
                    assert_ne!(
                        l[[j, j]],
                        Complex::new(T::zero(), T::zero()),
                        "The matrix is not positive definite"
                    );
                    l[[i, j]] = (self[[i, j]] - sum) / l[[j, j]];
                }
            }
        }
        Ok(CholeskyDec::new(LowerTriangular::new(l)))
    }
}
