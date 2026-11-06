use crate::algebra::abstr::Real;
use crate::algebra::linear::vector::Vector;
use crate::analysis::differential_equation::ordinary::ExplicitODE;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::clone::Clone;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct ExplicitRKEmbedded<T> {
    a: Vec<T>,
    b: Vec<T>,
    b_order: u8,
    b_s: Vec<T>,
    b_s_order: u8,
    c: Vec<T>,
}

impl<T> ExplicitRKEmbedded<T>
where
    T: Real,
{
    pub fn new(
        a: Vec<T>,
        b: Vec<T>,
        b_order: u8,
        b_s: Vec<T>,
        b_s_order: u8,
        c: Vec<T>,
    ) -> ExplicitRKEmbedded<T> {
        ExplicitRKEmbedded {
            a,
            b,
            b_order,
            b_s,
            b_s_order,
            c,
        }
    }

    pub fn do_step<O>(&self, ode: &O, t_n: &T, x_n: &Vector<T>, h: &T) -> (Vector<T>, Vector<T>)
    where
        O: ExplicitODE<T>,
    {
        let mut k: Vec<Vector<T>> = Vec::with_capacity(self.b.len());
        let (rows, _columns): (usize, usize) = x_n.dim();

        k.push(ode.ode(t_n, x_n));

        for j in 1..self.b.len() {
            let i_b = (j - 1) * j / 2;
            let i_e = i_b + j;

            let sum = ExplicitRKEmbedded::add_special(Vector::zero(rows), &k, &self.a[i_b..i_e]);

            let k_i = ode.ode(&(*t_n + self.c[j - 1] * *h), &(x_n + &(&sum * h)));

            k.push(k_i);
        }

        let sum = ExplicitRKEmbedded::add_special(Vector::zero(rows), &k, &self.b);

        let x_n_1 = x_n + &(&sum * h);

        let sum_s = ExplicitRKEmbedded::add_special(Vector::zero(rows), &k, &self.b_s);

        let x_s_n_1 = x_n + &(&sum_s * h);

        (x_n_1, x_s_n_1)
    }

    pub fn order(&self) -> (u8, u8) {
        (self.b_order, self.b_s_order)
    }
}

impl<T> ExplicitRKEmbedded<T>
where
    T: Real,
{
    fn add_special(mut s: Vector<T>, k: &Vec<Vector<T>>, b: &[T]) -> Vector<T> {
        let (m, _n) = s.dim();

        k.iter().zip(b.iter()).for_each(|(k_i, b_i)| {
            for l in 0..m {
                s[l] += k_i[l] * *b_i;
            }
        });

        s
    }
}
