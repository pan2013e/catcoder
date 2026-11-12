use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::matrix::Determinant;
use crate::algebra::linear::matrix::General;

impl<T> Determinant<T> for General<T>
where
    T: Field + Scalar,
{
    /// Calculates the determinant
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::{General, Determinant};
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, -2.0, 3.0, -7.0]);
    /// let det: f64 = a.det();
    /// assert_eq!(det, -1.0)
    /// ```
    fn det(&self) -> T {
        debug_assert_eq!(self.m, self.n);

        if self.m == 1 {
            return self[[0, 0]];
        }

        if self.m == 2 {
            let a_11: T = self[[0, 0]];
            let a_12: T = self[[0, 1]];
            let a_21: T = self[[1, 0]];
            let a_22: T = self[[1, 1]];
            return a_11 * a_22 - a_12 * a_21;
        }

        let (_l, u, p) = match self.dec_lu() {
            Err(_e) => return T::zero(),
            Ok(dec) => dec.lup(),
        };

        let mut det: T = T::one();

        for i in 0..self.m {
            det *= u[[i, i]];
        }

        // Determine the sign of the determinant due to the permutation matrix `p`.
        // If the number of even-sized cycles of the permutation `p` is odd,
        // then the sign of the determinant needs to be inverted.
        // See https://math.stackexchange.com/a/65938.
        let mut negated = false;
        let mut visited = vec![false; p.m];
        for i in 0..p.m {
            if visited[i] {
                continue;
            }
            let mut j = i;
            while p[[i, j]] == T::zero() {
                negated = !negated;
                for k in 0..p.m {
                    if p[[k, j]] != T::zero() {
                        j = k;
                        break;
                    }
                }
                visited[j] = true;
            }
        }
        if negated {
            det = -det;
        }

        det
    }
}
