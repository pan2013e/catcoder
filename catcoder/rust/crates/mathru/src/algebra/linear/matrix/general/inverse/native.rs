use crate::algebra::{
    abstr::{AbsDiffEq, Field, Scalar},
    linear::matrix::{General, Inverse, LUDec},
};

impl<T> Inverse<T> for General<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    type Output = Self;
    /// Calculates the inverse of a general matrix
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::{matrix::*, matrix::General};
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b_inv: General<f64> = a.inv().unwrap();
    /// ```
    fn inv(&self) -> Result<General<T>, ()> {
        self.inv_r()
    }
}

impl<T> General<T>
where
    T: Field + Scalar + AbsDiffEq,
{
    pub fn inv_r(&self) -> Result<General<T>, ()> {
        let lu_dec: LUDec<T> = self.dec_lu()?;
        lu_dec.inv()
    }
}
