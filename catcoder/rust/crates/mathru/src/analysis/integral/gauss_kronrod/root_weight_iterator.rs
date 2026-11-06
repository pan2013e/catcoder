use crate::algebra::abstr::Real;

use super::root_weight::RootWeight;

pub struct RootWeightIterator<'a, T> {
    n: u8,
    i: u8,
    root_weight: &'a RootWeight<T>,
}

impl<'a, T> Iterator for RootWeightIterator<'a, T>
where
    T: Real,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.i <= (2 * self.n) {
            let idx = self.idx();

            let root = self.root_weight.roots[idx];
            let weight = self.root_weight.kronrod_weights[idx];

            let pair = if self.i <= self.n {
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
        RootWeightIterator {
            n: (root_weight.roots.len() - 1) as u8,
            i: 0,
            root_weight,
        }
    }

    fn idx(self: &Self) -> usize {
        (-((self.i as i8 - self.n as i8).abs()) + self.n as i8) as usize
    }
}
