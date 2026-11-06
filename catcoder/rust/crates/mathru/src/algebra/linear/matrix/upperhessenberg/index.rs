use std::ops::{Index, IndexMut};

use super::UpperHessenberg;

impl<T> Index<[usize; 2]> for UpperHessenberg<T> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        debug_assert!(index[0] < self.matrix.m);
        debug_assert!(index[1] < self.matrix.n);

        &self.matrix[index]
    }
}

impl<T> IndexMut<[usize; 2]> for UpperHessenberg<T> {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        debug_assert!(index[0] < self.matrix.m);
        debug_assert!(index[1] < self.matrix.n);

        &mut self.matrix[index]
    }
}
