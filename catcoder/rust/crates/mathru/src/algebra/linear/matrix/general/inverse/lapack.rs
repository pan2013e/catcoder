use crate::algebra::{
    abstr::{Field, Scalar, Zero},
    linear::matrix::{General, Inverse, Transpose},
};

impl<T> Inverse<T> for General<T>
where
    T: Field + Scalar,
{
    type Output = Self;
    /// Inverse Matrix
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
    T: Field + Scalar,
{
    pub fn inv_r(&self) -> Result<General<T>, ()> {
        let (m, n): (usize, usize) = self.dim();
        let m_i32: i32 = m as i32;
        let n_i32: i32 = n as i32;

        let dim_min: i32 = m_i32.min(n_i32);
        let mut ipiv: Vec<i32> = vec![Zero::zero(); dim_min as usize];

        let mut info: i32 = 0;

        let mut self_data: Vec<T> = self.clone().transpose().data;

        T::xgetrf(
            m_i32,
            n_i32,
            self_data.as_mut_slice(),
            m_i32,
            ipiv.as_mut_slice(),
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        let lwork: i32 =
            T::xgetri_work_size(n_i32, &mut self_data[..], n_i32, &mut ipiv, &mut info);

        if info != 0 {
            return Err(());
        }

        let mut work: Vec<T> = vec![T::zero(); lwork as usize];

        T::xgetri(
            n_i32,
            &mut self_data[..],
            n_i32,
            &mut ipiv,
            &mut work,
            lwork,
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        let self_inv: General<T> = General::new(n, m, self_data);

        Ok(self_inv.transpose())
    }
}
