use crate::algebra::abstr::AbsDiffEq;
use crate::algebra::linear::matrix::UpperHessenberg;
use crate::algebra::linear::vector::Vector;
use crate::matrix;
use crate::{
    algebra::{
        abstr::{Field, Scalar},
        linear::matrix::{General, SchurDec, Transpose, UpperTriangular},
    },
    elementary::Power,
};

impl<T> UpperHessenberg<T>
where
    T: Field + Scalar + Power + AbsDiffEq<Epsilon = T>,
{
    /// Schur Decomposition
    /// ```math
    /// H = QUQ^{-1} \
    /// ```
    ///
    /// # Panics
    ///
    /// if H is not a square matrix
    ///
    /// # Example
    ///
    pub fn dec_schur(&self) -> Result<SchurDec<T>, ()> {
        let (q, u): (General<T>, UpperTriangular<T>) = if self.matrix.m > 2 {
            let h = self.clone();
            h.francis()
        } else {
            if self.matrix.m == 2 {
                let a_11 = self[[0, 0]];
                let a_12 = self[[0, 1]];
                let a_21 = self[[1, 0]];
                let a_22 = self[[1, 1]];
                let b = -(a_11 + a_22);
                let c = a_11 * a_22 - a_21 * a_12;
                let t_1 = (b * b - T::from_f32(4.0) * c).sqrt();
                let x_1 = (-b - t_1) / T::from_f32(2.0);
                let x_2 = (-b + t_1) / T::from_f32(2.0);
                (
                    General::zero(self.matrix.m, self.matrix.m),
                    UpperTriangular::from(matrix![x_1, T::zero(); T::zero(), x_2]),
                )
            } else {
                (matrix![T::one()], self.matrix.clone().into())
            }
        };

        Result::Ok(SchurDec::new(q, u))
    }

    //https://people.inf.ethz.ch/arbenz/ewp/Lnotes/chapter4.pdf
    fn francis(mut self) -> (General<T>, UpperTriangular<T>) {
        let epsilon: T = T::default_epsilon();

        let (m, n): (usize, usize) = self.matrix.dim();

        let mut u: General<T> = General::one(m);

        let mut p: usize = n;

        while p > 2 {
            let q = p - 1;

            // Bulge generating
            let h_qq = self[[q - 1, q - 1]];
            let h_pp = self[[p - 1, p - 1]];
            let s: T = h_qq + h_pp;

            let h_qp = self[[q - 1, p - 1]];
            let h_pq = self[[p - 1, q - 1]];
            let t: T = h_qq * h_pp - h_qp * h_pq;

            // compute first 3 elements of first column of M
            let h_11 = self[[0, 0]];
            let h_12 = self[[0, 1]];
            let h_21 = self[[1, 0]];
            let mut x: T = h_11 * h_11 + h_12 * h_21 - s * h_11 + t;

            let h_22 = self[[1, 1]];
            let mut y: T = h_21 * (h_11 + h_22 - s);

            let h_32 = self[[2, 1]];
            let mut z: T = h_21 * h_32;

            for k in 0..=(p - 3) {
                let b: Vector<T> = Vector::new_column(vec![x, y, z]);
                let hr: General<T> = General::householder(&b, 0);

                //Determine the Householder reflector P with P^T [x; y; z]^T = αe1 ;
                {
                    let r: usize = k.max(1);

                    let temp = &hr.clone().transpose() * &self.get_slice(k, k + 2, r - 1, n - 1);
                    self = self.set_slice(&temp, k, r - 1);
                }

                {
                    let r: usize = p.min(k + 4);
                    let temp: General<T> = &self.get_slice(0, r - 1, k, k + 2) * &hr;
                    self = self.set_slice(&temp, 0, k);

                    let temp1: General<T> = &u.get_slice(0, n - 1, k, k + 2) * &hr;

                    u = u.set_slice(&temp1, 0, k);
                }

                let h_k2_k1 = self[[k + 1, k]];
                x = h_k2_k1;
                let h_k3_k1 = self[[k + 2, k]];
                y = h_k3_k1;
                if k < (p - 3) {
                    let h_k4_k1 = self[[k + 3, k]];
                    z = h_k4_k1;
                }
            }

            // Determine the Givens rotation P with P [x; y]T = αe1 ;
            let (c, s): (T, T) = General::givens_cosine_sine_pair(x, y);
            let g: General<T> = General::givens(2, 0, 1, c, s);

            {
                let temp: General<T> = &g * &self.get_slice(q - 1, p - 1, p - 3, n - 1);
                self = self.set_slice(&temp, q - 1, p - 3);
            }

            {
                let g_trans: General<T> = g.transpose();
                let temp: General<T> = &self.get_slice(0, p - 1, p - 2, p - 1) * &g_trans;
                self = self.set_slice(&temp, 0, p - 2);

                let u_slice = &self.get_slice(0, p - 1, p - 2, p - 1) * &g_trans;
                u = u.set_slice(&u_slice, 0, p - 2);
            }

            // check for convergence
            let m: T = self[[q - 1, q - 1]].abs();
            let n: T = self[[p - 1, p - 1]].abs();
            if self[[p - 1, q - 1]].abs() < epsilon * (m + n) {
                self[[p - 1, q - 1]] = T::zero();
                p -= 1;
            } else {
                let k: T = self[[q - 2, q - 2]].abs();
                let l: T = self[[q - 1, q - 1]].abs();
                if self[[p - 2, q - 2]].abs() < epsilon * (k + l) {
                    self[[p - 2, q - 2]] = T::zero();
                    p -= 2;
                }
            }
        }

        (u, self.matrix.into())
    }
}
