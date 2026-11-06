use crate::algebra::abstr::{AbsDiffEq, Field, Scalar};
use crate::algebra::linear::matrix::{Diagonal, EigenDec, EigenDecomposition, General};
use crate::elementary::Power;

impl<T> EigenDecomposition<T> for Diagonal<T>
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
    /// use mathru::algebra::linear::matrix::{EigenDec, EigenDecomposition, General, Diagonal};
    /// use mathru::{matrix, assert_relative_eq};
    ///
    /// let a: Diagonal<f64> = Diagonal::new(&[1.0, 2.0, 3.0]);
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

        let values: Diagonal<T> = self.clone();
        let vectors: General<T> = Diagonal::new(&vec![T::one(); m]).into();

        Ok(EigenDec::new(values, vectors))
    }
}
