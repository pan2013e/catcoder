use crate::algebra::{
    abstr::{Field, Scalar},
    linear::matrix::General,
};

use super::UpperHessenberg;

impl<T> UpperHessenberg<T>
where
    T: Field + Scalar,
{
    /// Returns a slice of the matrix
    ///
    /// # Arugments
    ///
    /// 0 <= row_s < m \
    /// 0 <= row_e < m \
    /// 0 <= column_s < n \
    /// 0 <= column_e <= n \
    ///
    /// row_s: start row \
    /// row_e: end row \
    /// column_s: start column \
    /// column_e: end column \
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate mathru;
    /// # fn main()
    /// # {
    /// # }
    /// ```
    pub fn get_slice(
        &self,
        row_s: usize,
        row_e: usize,
        column_s: usize,
        column_e: usize,
    ) -> General<T> {
        debug_assert!(row_s < self.matrix.m);
        debug_assert!(row_e < self.matrix.m);
        debug_assert!(column_s < self.matrix.n);
        debug_assert!(column_e < self.matrix.n);

        let mut slice: General<T> = General::zero(row_e - row_s + 1, column_e - column_s + 1);

        for r in row_s..=(row_e) {
            for c in column_s..=(column_e) {
                slice[[r - row_s, c - column_s]] = self[[r, c]];
            }
        }
        slice
    }

    /// Replaces parts of the matrix with the given values
    ///
    /// # Arugments
    ///
    /// 0 <= row < m \
    /// 0 <= column < n
    ///
    /// # Example
    ///
    /// ```
    /// # #[macro_use]
    /// # extern crate mathru;
    /// # fn main()
    /// # {
    /// # }
    /// ```
    pub fn set_slice(
        mut self,
        slice: &General<T>,
        row: usize,
        column: usize,
    ) -> UpperHessenberg<T> {
        let (s_m, s_n): (usize, usize) = slice.dim();
        let (m, n): (usize, usize) = self.matrix.dim();
        debug_assert!(row + s_m <= m);
        debug_assert!(column + s_n <= n);

        for r in row..(row + s_m) {
            for c in column..(column + s_n) {
                self[[r, c]] = slice[[r - row, c - column]];
            }
        }
        self
    }
}
