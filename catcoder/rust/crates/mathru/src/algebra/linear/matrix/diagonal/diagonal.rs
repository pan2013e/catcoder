use crate::algebra::{
    abstr::{Field, Scalar, Zero},
    linear::matrix::General,
};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Diagonal matrix
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Diagonal<T> {
    pub(crate) matrix: General<T>,
}

impl<T> Diagonal<T>
where
    T: Field + Scalar + Zero,
{
    /// Construct a matrix with vec as its diagonal.
    pub fn new(vec: &[T]) -> Diagonal<T> {
        let mut g = General::zero(vec.len(), vec.len());
        for (idx, v) in vec.iter().enumerate() {
            g[[idx, idx]] = *v;
        }
        Diagonal { matrix: g }
    }
}
impl<T> Diagonal<T> {
    pub fn dim(&self) -> (usize, usize) {
        self.matrix.dim()
    }
}

impl<T> Diagonal<T>
where
    T: Clone,
{
    /// Applies the function f on every diagonal element in the matrix
    pub fn apply_mut(mut self: Diagonal<T>, f: &dyn Fn(&T) -> T) -> Diagonal<T> {
        let (m, n) = self.dim();
        let k = m.min(n);
        for i in 0..k {
            self[[i, i]] = f(&self[[i, i]]);
        }

        self
    }

    pub fn apply(self: &Diagonal<T>, f: &dyn Fn(&T) -> T) -> Diagonal<T> {
        (self.clone()).apply_mut(f)
    }

    pub fn mut_apply(self: &mut Diagonal<T>, f: &dyn Fn(&mut T) -> T) {
        let (m, n) = self.dim();
        let k = m.min(n);
        for i in 0..k {
            self[[i, i]] = f(&mut self[[i, i]]);
        }
    }
}
