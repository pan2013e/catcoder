use crate::algebra::abstr::Polynomial;
use crate::algebra::abstr::Real;
use crate::algebra::linear::matrix::Solve;
use crate::algebra::linear::{matrix::General, vector::Vector};
use std::cmp::Ordering;
use std::fmt::{Display, Formatter, Result as FmtResult};

/// Piecewise polynomial of degree 3
#[derive(Clone, Debug)]
pub struct CubicSpline<T>
where
    T: Real,
{
    polynomials: Vec<((T, T), Polynomial<T>)>,
}

pub enum CubicSplineConstraint {
    Natural,
    Periodic,
}

impl<T> CubicSpline<T>
where
    T: Real,
{
    pub fn eval(&self, x: T) -> T {
        let pw = self.polynomials.binary_search_by(|pw| {
            if x < pw.0 .0 {
                Ordering::Greater
            } else if x >= pw.0 .1 {
                Ordering::Less
            } else {
                Ordering::Equal
            }
        });

        if pw.is_err() {
            if x < self.polynomials[0].0 .0 {
                panic!("Out of range")
            }

            let last = self.polynomials.last().unwrap();
            if x > last.0 .1 {
                panic!("Out of range");
            }
            if x == last.0 .1 {
                return last.1.eval(x);
            }
            panic!("Not found")
        }

        self.polynomials[pw.unwrap()].1.eval(x)
    }

    /// Interpolate cubic spline to given data, with given contraint
    ///
    ///
    /// `constraint`:
    /// * Natural
    /// * Cubic
    ///
    /// # Panics
    ///
    /// Panics if x.len() < 2 or 'x.len() != y.len()' or if $x_{n} >= x_{n+1}$
    ///
    /// # Example
    /// ```
    /// use mathru::analysis::interpolation::spline::{CubicSpline, CubicSplineConstraint};
    /// use mathru::assert_relative_eq;
    ///
    /// let x = vec![-2.0, 0.0, 1.0, 2.0, 4.0];
    /// let y = vec![6.0, 2.0, 4.0, 5.0, -2.0];
    ///
    /// let cubic_spline = CubicSpline::interpolate(&x, &y, CubicSplineConstraint::Periodic);
    ///
    /// assert_relative_eq!(6.0, cubic_spline.eval(-2.0));
    /// ```
    pub fn interpolate(x: &[T], y: &[T], constraint: CubicSplineConstraint) -> CubicSpline<T> {
        if x.len() < 2 {
            panic!("In minimum two points are required");
        }

        if x.len() != y.len() {
            panic!("x.len() == {} != y.len() ==  {}", x.len(), y.len());
        }

        match constraint {
            CubicSplineConstraint::Natural => Self::solve_natural(x, y),
            CubicSplineConstraint::Periodic => Self::solve_periodic(x, y),
        }
    }

    // https://www3.math.tu-berlin.de/Vorlesungen/SoSe09/EinfNumMath/numerik1.pdf
    fn solve_natural(x: &[T], y: &[T]) -> CubicSpline<T> {
        let n = x.len() - 1;

        let mut m = if n > 2 {
            let h_0 = x[1] - x[0];
            let h_1 = x[2] - x[1];
            let h_1n = x[n] - x[n - 1];
            let h_2n = x[n - 1] - x[n - 2];

            let y_n = y[n];
            let y_1n = y[n - 1];
            let y_2n = y[n - 2];

            let y_0 = y[0];
            let y_1 = y[1];
            let y_2 = y[2];

            let mut a: General<T> = General::zero(n - 1, n - 1);
            let mut b: Vector<T> = Vector::zero(n - 1);

            a[[0, 0]] = T::from_f64(2.0) * (h_0 + h_1);
            a[[0, 1]] = h_1;
            b[0] = T::from_f64(6.0) * ((y_2 - y_1) / h_1 - (y_1 - y_0) / h_0);

            for i in 2..=(n - 2) {
                let y_1i = y[i - 1];
                let y_i = y[i];
                let y_i1 = y[i + 1];
                let h_1i = x[i] - x[i - 1];
                let h_i = x[i + 1] - x[i];

                a[[i - 1, i - 2]] = h_1i;
                a[[i - 1, i - 1]] = T::from_f64(2.0) * (h_1i + h_i);
                a[[i - 1, i]] = h_i;

                b[i - 1] = T::from_f64(6.0) * ((y_i1 - y_i) / h_i - (y_i - y_1i) / h_1i);
            }

            a[[n - 2, n - 2]] = T::from_f64(2.0) * (h_2n + h_1n);
            a[[n - 2, n - 3]] = h_2n;
            b[n - 2] = T::from_f64(6.0) * ((y_n - y_1n) / h_1n - (y_1n - y_2n) / h_2n);

            let d = a.dec_lu().unwrap().solve(&b).unwrap();
            d.convert_to_vec()
        } else {
            if n == 1 {
                vec![]
            } else {
                let y_0 = y[0];
                let y_1 = y[1];
                let y_2 = y[2];

                let h_0 = x[1] - x[0];
                let h_1 = x[2] - x[1];

                let g_1 = T::from_f64(6.0) * ((y_2 - y_1) / h_1 - (y_1 - y_0) / h_0);
                vec![g_1 / (T::from_f64(2.0) * (h_0 + h_1))]
            }
        };

        m.insert(0, T::zero());
        m.push(T::zero());

        Self::construct_cubic_spline(&x, &y, &m)
    }

    // https://www3.math.tu-berlin.de/Vorlesungen/SoSe09/EinfNumMath/numerik1.pdf
    fn solve_periodic(x: &[T], y: &[T]) -> CubicSpline<T> {
        let n = x.len() - 1;

        let mut data = if n >= 3 {
            let h_0 = x[1] - x[0];
            let h_1n = x[n] - x[n - 1];
            let h_2n = x[n - 1] - x[n - 2];

            let y_n = y[n];
            let y_1n = y[n - 1];
            let y_2n = y[n - 2];

            let y_0 = y[0];
            let y_1 = y[1];

            let mut a: General<T> = General::zero(n, n);
            let mut b: Vector<T> = Vector::zero(n);

            a[[0, 0]] = T::from_f64(2.0) * (h_1n + h_0);
            a[[0, 1]] = h_0;
            b[0] = T::from_f64(6.0) * ((y_1 - y_0) / h_0 - (y_n - y_1n) / h_1n);

            for i in 1..=(n - 2) {
                let y_1i = y[i - 1];
                let y_i = y[i];
                let y_i1 = y[i + 1];
                let h_1i = x[i] - x[i - 1];
                let h_i = x[i + 1] - x[i];

                a[[i, i - 1]] = h_1i;
                a[[i, i]] = T::from_f64(2.0) * (h_1i + h_i);
                a[[i, i + 1]] = h_i;

                b[i] = T::from_f64(6.0) * ((y_i1 - y_i) / h_i - (y_i - y_1i) / h_1i);
            }

            a[[n - 1, n - 2]] = h_2n;
            a[[n - 1, n - 1]] = T::from_f64(2.0) * (h_2n + h_1n);
            b[n - 1] = T::from_f64(6.0) * ((y_n - y_1n) / h_1n - (y_1n - y_2n) / h_2n);

            a[[0, n - 1]] = h_1n;
            a[[n - 1, 0]] = h_1n;

            let d = a.dec_lu().unwrap().solve(&b).unwrap();

            d.convert_to_vec()
        } else {
            if n == 1 {
                let h_0 = x[1] - x[0];
                let h_1n = x[n] - x[n - 1];

                let y_n = y[n];
                let y_1n = y[n - 1];

                let y_0 = y[0];
                let y_1 = y[1];

                let a_00 = T::from_f64(2.0) * (h_1n + h_0);
                let a_01 = h_1n;
                let b_0 = T::from_f64(6.0) * ((y_1 - y_0) / h_0 - (y_n - y_1n) / h_1n);

                vec![b_0 / (a_00 + a_01)]
            } else {
                let h_0 = x[1] - x[0];
                let h_1n = x[n] - x[n - 1];
                let h_1 = x[2] - x[1];

                let y_n = y[n];
                let y_1n = y[n - 1];

                let y_0 = y[0];
                let y_1 = y[1];
                let y_2 = y[2];

                let mut a: General<T> = General::zero(n, n);
                let mut b: Vector<T> = Vector::zero(n);

                a[[0, 0]] = T::from_f64(2.0) * (h_1n + h_0);
                a[[0, 1]] = h_0;
                b[0] = T::from_f64(6.0) * ((y_1 - y_0) / h_0 - (y_n - y_1n) / h_1n);

                a[[1, 0]] = h_0;
                a[[1, 1]] = T::from_f64(2.0) * (h_1 + h_0);
                b[1] = T::from_f64(6.0) * ((y_2 - y_1) / h_1 - (y_1 - y_0) / h_0);

                a[[0, n - 1]] += h_1n;
                a[[n - 1, 0]] += h_1n;

                let d = a.dec_lu().unwrap().solve(&b).unwrap();

                d.convert_to_vec()
            }
        };

        let m_0 = data[0];
        data.push(m_0);

        Self::construct_cubic_spline(&x, &y, &data)
    }

    fn construct_cubic_spline(x: &[T], y: &[T], s: &[T]) -> CubicSpline<T> {
        let n = x.len() - 1;

        let mut polynomials = Vec::new();

        for i in 0..n {
            let h_i = x[i + 1] - x[i];
            let a_i = y[i];
            let b_i =
                (y[i + 1] - y[i]) / h_i - h_i / T::from(6.0) * (s[i + 1] + T::from_f64(2.0) * s[i]);
            let c_i = s[i] / T::from_f64(2.0);
            let d_i = (s[i + 1] - s[i]) / (T::from_f64(6.0) * h_i);

            let x_i_1 = x[i];
            let x_i_2 = x_i_1 * x_i_1;
            let x_i_3 = x_i_2 * x_i_1;

            let f_0 = a_i - b_i * x_i_1 + c_i * x_i_2 - d_i * x_i_3;
            let f_1 = b_i - T::from_f64(2.0) * c_i * x_i_1 + T::from_f64(3.0) * d_i * x_i_2;
            let f_2 = c_i - T::from_f64(3.0) * d_i * x_i_1;
            let f_3 = d_i;

            let polynomial = Polynomial::from_coef(vec![f_3, f_2, f_1, f_0]);

            let range = (x[i], x[i + 1]);
            polynomials.push((range, polynomial));
        }

        CubicSpline { polynomials }
    }

    pub fn solve_thomas(
        a: &Vector<T>,
        b: &Vector<T>,
        mut c: Vector<T>,
        mut d: Vector<T>,
    ) -> Result<Vector<T>, String> {
        let (n, _) = b.dim();

        c[0] /= b[0];
        for i in 1..=(n - 2) {
            let c_m = c[i - 1];
            c[i] /= b[i] - c_m * a[i];
        }

        d[0] /= b[0];
        for i in 1..=(n - 1) {
            d[i] = (d[i] - d[i - 1] * a[i - 1]) / (b[i] - c[i - 1] * a[i - 1])
        }

        for i in (0..=(n - 2)).rev() {
            d[i] = d[i] - c[i] * d[i + 1]
        }

        Ok(d)
    }
}

impl<T> CubicSpline<T>
where
    T: Real,
{
    pub fn differentiate(&self) -> CubicSpline<T> {
        let diffs = self
            .polynomials
            .iter()
            .map(|p| (p.0, p.1.differentiate()))
            .collect();
        CubicSpline { polynomials: diffs }
    }
}

impl<T> Display for CubicSpline<T>
where
    T: Display + Real,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for (i, p_i) in self.polynomials.iter().enumerate() {
            let range = &p_i.0;
            let polynomial = &p_i.1;

            if i == 0 {
                writeln!(f, "{}\t, if x in [{}, {}]", polynomial, range.0, range.1)?;
            } else {
                writeln!(f, "{}\t, if x in ({}, {}]", polynomial, range.0, range.1)?;
            }
        }

        Ok(())
    }
}
