use crate::algebra::abstr::Zero;
use crate::algebra::linear::matrix::{UnitLowerTriangular, UpperTriangular};
use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::{General, LUDec},
};

impl<T> General<T>
where
    T: Field + Scalar,
{
    /// Decomposes the matrix into a upper and a lower matrix
    ///
    /// PA = LU
    ///
    /// # Arguments
    ///
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, UnitLowerTriangular, UpperTriangular};
    /// let a: General<f64> = General::new(2, 2, vec![1.0, -2.0, 3.0, -7.0]);
    ///
    /// let (l, u, p): (UnitLowerTriangular<f64>, UpperTriangular<f64>, General<f64>) = a.dec_lu().unwrap().lup();
    /// ```
    pub fn dec_lu(&self) -> Result<LUDec<T>, ()> {
        let (m, n): (usize, usize) = self.dim();
        debug_assert_eq!(m, n);

        let (m, n): (usize, usize) = self.dim();
        let m_i32: i32 = m as i32;
        let n_i32: i32 = n as i32;

        let dim_min: i32 = m_i32.min(n_i32);
        let mut ipiv: Vec<i32> = vec![Zero::zero(); dim_min as usize];

        let mut info: i32 = 0;

        let mut self_data = self.clone().data;

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

        let mat: General<T> = General::new(m, n, self_data);
        let l: UnitLowerTriangular<T> = General::l(mat.clone());
        let u: UpperTriangular<T> = General::u(mat.clone());
        let p: General<T> = General::p(ipiv);

        Ok(LUDec::new(l, u, p))
    }

    fn l(mut mat: General<T>) -> UnitLowerTriangular<T> {
        let (m, n): (usize, usize) = mat.dim();

        //fill upper triangle with zero
        for i in 0..m {
            for k in i..n {
                mat[[i, k]] = T::zero();
            }
        }

        //set diagonal to 1
        for i in 0..m {
            mat[[i, i]] = T::one();
        }

        UnitLowerTriangular { matrix: mat }
    }

    fn u(mut mat: General<T>) -> UpperTriangular<T> {
        let (m, _n): (usize, usize) = mat.dim();

        //fill lower triangle with zero
        for i in 0..m {
            for k in 0..i {
                mat[[i, k]] = T::zero();
            }
        }
        UpperTriangular { matrix: mat }
    }

    /// transforms a sequence of permutations to a permutation matrix
    fn p(per: Vec<i32>) -> Self {
        let length = per.len();

        let mut perm: Vec<usize> = vec![0; length];

        for i in 0..length {
            perm[i] = i;
        }

        for i in 0..length - 1 {
            let temp = perm[(per[i] - 1) as usize];
            perm[(per[i] - 1) as usize] = perm[i];
            perm[i] = temp;
        }

        let mut p: General<T> = General::zero(length, length);

        for i in 0..length {
            let k = perm[i];
            p[[i, k]] = T::one();
        }

        p
    }
}
