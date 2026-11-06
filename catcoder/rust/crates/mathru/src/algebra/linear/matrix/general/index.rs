use crate::algebra::linear::matrix::General;
use std::ops::{Index, IndexMut};

impl<T> Index<[usize; 2]> for General<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        debug_assert!(
            index[0] < self.m,
            "index out of bounds: is a {}x{} matrix but the index is {}x{}",
            self.nrows(),
            self.ncols(),
            index[0],
            index[1]
        );
        debug_assert!(
            index[1] < self.n,
            "index out of bounds: is a {}x{} matrix but the index is {}x{}",
            self.nrows(),
            self.ncols(),
            index[0],
            index[1]
        );

        &self.data[index[1] * self.m + index[0]]
    }
}

impl<T> IndexMut<[usize; 2]> for General<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        debug_assert!(
            index[0] < self.m,
            "index out of bounds: is a {}x{} matrix but the index is {}x{}",
            self.m,
            self.n,
            index[0],
            index[1]
        );
        debug_assert!(
            index[1] < self.n,
            "index out of bounds: is a {}x{} matrix but the index is {}x{}",
            self.m,
            self.n,
            index[0],
            index[1]
        );

        &mut self.data[index[1] * self.m + index[0]]
    }
}
