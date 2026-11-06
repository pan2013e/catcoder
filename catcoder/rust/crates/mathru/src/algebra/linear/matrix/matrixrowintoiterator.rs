use crate::algebra::abstr::{Field, Scalar};
use crate::algebra::linear::{matrix::General, vector::vector::Vector};

pub struct MatrixRowIntoIterator<'a, T> {
    m: &'a General<T>,
    row: usize,
}

impl<'a, T> MatrixRowIntoIterator<'a, T> {
    pub fn new(m: &'a General<T>) -> MatrixRowIntoIterator<'a, T> {
        MatrixRowIntoIterator { m, row: 0 }
    }
}

impl<'a, T> Iterator for MatrixRowIntoIterator<'a, T>
where
    T: Field + Scalar,
{
    type Item = Vector<T>;

    // just return the str reference
    fn next(&mut self) -> Option<Self::Item> {
        if self.row < self.m.nrows() {
            let row: Vector<T> = self.m.get_row(self.row);
            self.row += 1;

            Some(row)
        } else {
            None
        }
    }
}
