use crate::algebra::abstr::Real;

use super::RootWeight;

pub struct RootWeightIterator<'a, T> {
    n: u8,
    i: u8,
    k: u8,
    root_weight: &'a RootWeight<T>,
}

impl<'a, T> Iterator for RootWeightIterator<'a, T>
where
    T: Real,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let f_idx = if self.n % 2 == 0 {
            RootWeightIterator::idx_n_even
        } else {
            RootWeightIterator::idx_n_odd
        };

        if self.i <= self.n {
            let idx = f_idx(self);
            let root = self.root_weight.roots[idx];
            let weight = self.root_weight.weights[idx];
            let pair = if self.i <= self.k {
                Some((-root, weight))
            } else {
                Some((root, weight))
            };
            self.i += 1;
            pair
        } else {
            None
        }
    }
}

impl<'a, T> RootWeightIterator<'a, T>
where
    T: Real,
{
    pub fn new(root_weight: &'a RootWeight<T>) -> RootWeightIterator<T> {
        let n = if root_weight.roots[0] == T::zero() {
            (root_weight.roots.len() * 2 - 1) as u8
        } else {
            (root_weight.roots.len() * 2) as u8
        };

        let k = if n % 2 == 0 { n / 2 } else { n / 2 + 1 };

        RootWeightIterator {
            n,
            i: 1,
            k,
            root_weight,
        }
    }

    fn idx_n_even(self: &Self) -> usize {
        if self.i <= self.k {
            (self.i % self.k) as usize
        } else {
            (self.i - (self.k + 1)) as usize
        }
    }

    fn idx_n_odd(self: &Self) -> usize {
        (self.i as i8 - self.k as i8).abs() as usize
    }
}
