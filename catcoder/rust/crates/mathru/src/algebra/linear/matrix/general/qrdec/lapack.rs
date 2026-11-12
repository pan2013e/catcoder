use crate::{
    algebra::{
        abstr::{Field, Scalar, Zero},
        linear::matrix::{General, QRDec, UpperTriangular},
    },
    elementary::Power,
};

impl<T> General<T>
where
    T: Field + Scalar + Power,
{
    /// QR Decomposition with Givens rotations
    ///
    /// A = QR \
    /// Q is an orthogonal matrix \
    /// R is an upper triangular matrix \
    ///
    /// # Panics
    ///
    /// if A is not a square matrix
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UpperTriangular};
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, -2.0, 3.0, -7.0]);
    ///
    /// let (q, r): (General<f64>, UpperTriangular<f64>) = a.dec_qr().unwrap().qr();
    /// ```
    pub fn dec_qr(&self) -> Result<QRDec<T>, ()> {
        let (m, n) = self.dim();
        debug_assert!(m >= n);

        let (m, n): (usize, usize) = self.dim();

        //lapack(fortran) uses column major order
        let mut self_data = self.clone().data;

        let m_i32: i32 = m as i32;
        let n_i32: i32 = n as i32;
        let m_n_min: usize = m.min(n);

        let mut tau: Vec<T> = vec![Zero::zero(); m_n_min];

        let mut info: i32 = 0;

        let lwork: i32 = T::xgeqrf_work_size(
            m_i32,
            n_i32,
            &mut self_data[..],
            m_i32,
            &mut tau[..],
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        let mut work: Vec<T> = vec![T::zero(); lwork as usize];

        T::xgeqrf(
            m_i32,
            n_i32,
            &mut self_data[..],
            m_i32,
            tau.as_mut(),
            &mut work,
            lwork,
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        let a: General<T> = General::new(m, n, self_data.clone());
        let r: UpperTriangular<T> = a.r();

        let lwork = T::xorgqr_work_size(
            m_i32,
            m_n_min as i32,
            tau.len() as i32,
            &mut self_data[..],
            m_i32,
            &mut tau[..],
            &mut info,
        );
        if info != 0 {
            return Err(());
        }

        let mut work = vec![T::zero(); lwork as usize];

        T::xorgqr(
            m_i32,
            m_n_min as i32,
            tau.len() as i32,
            &mut self_data[..],
            m_i32,
            &mut tau[..],
            &mut work,
            lwork,
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        let q: General<T> = General::new(m, n, self_data);

        Ok(QRDec::new(q, r))
    }

    fn r(mut self) -> UpperTriangular<T> {
        for i in 1..self.m {
            for k in 0..(i.min(self.n)) {
                self[[i, k]] = T::zero();
            }
        }

        UpperTriangular::new(self)
    }
}
