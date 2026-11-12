use crate::algebra::linear::matrix::UnitLowerTriangular;
use crate::algebra::linear::matrix::UpperTriangular;
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
    /// ```math
    /// PA = LU
    /// ```
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

        let mut l: General<T> = General::one(self.m);
        let mut u: General<T> = General::one(self.n);
        let mut p: General<T> = General::one(self.m);

        let mut a: General<T> = self.clone();

        for i in 0..a.m {
            //pivoting
            let mut max: T = T::zero();
            let mut i_max: usize = i;

            for l in i..a.m {
                let p_cand: T = a[[l, i]].abs();
                if p_cand > max {
                    max = p_cand;
                    i_max = l;
                }
            }

            if i != i_max {
                a.swap_rows(i, i_max);
                p.swap_rows(i, i_max);
            }

            for j in (i + 1)..a.n {
                let f: T = if a[[i, i]] != T::zero() {
                    a[[j, i]] / a[[i, i]]
                } else {
                    a[[j, i]]
                };

                for k in (i + 1)..a.n {
                    a[[j, k]] = a[[j, k]] - f * a[[i, k]];
                }
                a[[j, i]] = f;
            }
        }

        for i in 1..a.n {
            for j in 0..i {
                l.data[j * a.m + i] = a.data[j * a.m + i];
            }
        }

        for i in 0..a.n {
            for k in i..a.n {
                u.data[k * a.m + i] = a.data[k * a.m + i];
            }
        }

        Ok(LUDec::new(
            UnitLowerTriangular::new(l),
            UpperTriangular::new(u),
            p,
        ))
    }
}
