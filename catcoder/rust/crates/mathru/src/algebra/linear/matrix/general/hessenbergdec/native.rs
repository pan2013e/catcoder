use crate::{
    algebra::{
        abstr::{Field, Scalar},
        linear::{
            matrix::{General, HessenbergDec, HessenbergDecomposition, Transpose, UpperHessenberg},
            vector::Vector,
        },
    },
    elementary::Power,
};

impl<T> HessenbergDecomposition<T> for General<T>
where
    T: Field + Scalar + Power,
{
    /// Decomposes self in to the M
    ///
    /// q * h * q^T = self
    ///
    /// # Arguments
    ///
    /// # Return
    ///
    /// (q, h)
    ///
    /// # Panics
    ///
    /// if M is not a square matrix
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UpperHessenberg, HessenbergDecomposition};
    ///
    /// let a: General<f64> = General::new(3, 3, vec![1.0, 5.0, 3.0, 1.0, 0.0, -7.0, 3.0, 8.0, 9.0]);
    /// let (q, h): (General<f64>, UpperHessenberg<f64>) = a.dec_hessenberg().qh();
    /// ```
    fn dec_hessenberg(&self) -> HessenbergDec<T> {
        let (m, n): (usize, usize) = self.dim();
        debug_assert_eq!(
            m, n,
            "Unable to compute the hessenberg decomposition of a non-square matrix"
        );
        debug_assert_ne!(
            m, 0,
            "Unable to compute the hessenberg decomposition of an empty matrix."
        );

        let (m, _n): (usize, usize) = self.dim();

        let mut q: General<T> = General::one(m);
        let mut h: General<T> = self.clone();

        for k in 1..m - 1 {
            let v: Vector<T> = h.get_column(k - 1);

            let househ: General<T> = General::householder(&v, k);
            h = &househ.clone().transpose() * &h;
            q = &househ * &q;
            h = &h.clone() * &househ;
        }

        HessenbergDec::new(q, UpperHessenberg::new(h))
    }
}
