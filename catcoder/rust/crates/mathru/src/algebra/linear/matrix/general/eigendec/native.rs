use crate::algebra::abstr::{AbsDiffEq, Field, RelativeEq, Scalar};
use crate::algebra::linear::matrix::Diagonal;
use crate::algebra::linear::{
    matrix::{
        EigenDec, EigenDecomposition, General, HessenbergDecomposition, Solve, UpperHessenberg,
    },
    vector::Vector,
};
use crate::elementary::Power;

/// ```math
/// A v = \lambda v \\\\
/// \lambda_1, \lambda_2, \lambda_3 \\\\
/// ```
/// $ D = $Diagonal matrix with $\lambda_1, \lambda_2 \dots$
/// ```math
/// A * (v_1; v_2; v_3) = (v_1; v_2; v_3)* D
/// ```
impl<T> EigenDecomposition<T> for General<T>
where
    T: Field + Scalar + Power + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    /// Computes the eigenvalues of a real matrix
    ///
    /// # Arguments
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::{matrix::{EigenDec, EigenDecomposition, General}, vector::Vector};
    /// use mathru::matrix;
    ///
    /// let a: General<f64> = matrix![  1.0, -3.0, 3.0;
    ///                                 3.0, -5.0, 3.0;
    ///                                 6.0, -6.0, 4.0];
    ///
    /// let eigen: EigenDec<f64> = a.dec_eigen().unwrap();
    /// ```
    fn dec_eigen(&self) -> Result<EigenDec<T>, String> {
        let (m, n): (usize, usize) = self.dim();
        debug_assert_eq!(
            m, n,
            "Unable to compute the eigen value of a non-square matrix"
        );
        debug_assert_ne!(
            m, 0,
            "Unable to compute the eigen value of an empty matrix."
        );

        let values = self.eigenvalue_r()?;
        let vectors: General<T> = self.eigenvector_r(&values);

        Ok(EigenDec::new(Diagonal::new(&values), vectors))
    }
}

impl<T> General<T>
where
    T: Field + Scalar + Power + AbsDiffEq<Epsilon = T> + RelativeEq,
{
    pub fn eigenvalue_r(&self) -> Result<Vec<T>, String> {
        let (m, n): (usize, usize) = self.dim();

        let h: UpperHessenberg<T> = self.dec_hessenberg().h();

        let dec_schur_res = h.dec_schur();

        if dec_schur_res.is_err() {
            return Result::Err("".to_string());
        }
        let u = dec_schur_res.unwrap().u();

        let mut eig = Vec::with_capacity(m);

        let mut i = 0;
        while i <= n - 1 {
            if i == (n - 1) || u[[i + 1, i]] == T::zero() {
                eig.push(u[[i, i]]);
                i += 1;
            } else {
                let a_ii = u[[i, i]];
                let a_ii1 = u[[i, i + 1]];
                let a_i1i = u[[i + 1, i]];
                let a_i1i1 = u[[i + 1, i + 1]];

                let (l1, l2) = General::eigen_2by2(a_ii, a_ii1, a_i1i, a_i1i1);
                eig.push(l1);
                eig.push(l2);
                i += 2;
            }
        }

        Result::Ok(eig)
    }

    pub fn eigenvector_r(&self, values: &Vec<T>) -> General<T> {
        let eye: General<T> = General::one(self.m);
        let zero_vector: Vector<T> = Vector::zero(self.m);
        let mut vectors: General<T> = General::zero(self.m, self.m);

        for (c, val) in values.iter().enumerate() {
            let diff: General<T> = self - &(&eye * val);
            let vec: Vector<T> = diff.solve(&zero_vector).unwrap();
            let v = General::norm(vec);
            vectors.set_column(&v, c);
        }

        vectors
    }

    fn norm(mut v: Vector<T>) -> Vector<T> {
        let norm = v.p_norm(&T::from_f32(2.0));
        if norm != T::zero() {
            v *= T::one() / norm;
        }
        v
    }

    fn eigen_2by2(a11: T, a12: T, a21: T, a22: T) -> (T, T) {
        let m = (a11 + a22) / T::from_f32(2.0);
        let p = a11 * a22 - a12 * a21;

        let k = (m * m - p).sqrt();
        let l1 = m + k;
        let l2 = m - k;

        (l1, l2)
    }
}
