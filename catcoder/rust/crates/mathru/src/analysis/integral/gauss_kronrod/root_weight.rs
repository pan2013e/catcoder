use crate::algebra::abstr::Real;
use std::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::root_weight_iterator::RootWeightIterator;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct RootWeight<T> {
    pub roots: Vec<T>,
    pub gauss_weights: Vec<T>,
    pub kronrod_weights: Vec<T>,
}

impl<T> RootWeight<T>
where
    T: Real,
{
    pub fn new() -> RootWeight<T> {
        let roots = vec![
            T::from_f64(0.991455371120813),
            T::from_f64(0.949107912342759),
            T::from_f64(0.864864423359769),
            T::from_f64(0.741531185599394),
            T::from_f64(0.586087235467691),
            T::from_f64(0.405845151377397),
            T::from_f64(0.207784955007898),
            T::zero(),
        ];

        let gauss_weights = vec![
            T::from_f64(0.129484966168870),
            T::from_f64(0.279705391489277),
            T::from_f64(0.381830050505119),
            T::from_f64(0.417959183673469),
        ];

        let kronrod_weights = vec![
            T::from_f64(0.022935322010529),
            T::from_f64(0.063092092629979),
            T::from_f64(0.104790010322250),
            T::from_f64(0.140653259715525),
            T::from_f64(0.169004726639267),
            T::from_f64(0.190350578064785),
            T::from_f64(0.204432940075298),
            T::from_f64(0.209482141084728),
        ];

        RootWeight {
            roots,
            gauss_weights,
            kronrod_weights,
        }
    }

    pub fn iter(&self) -> RootWeightIterator<T> {
        RootWeightIterator::new(&self)
    }
}
