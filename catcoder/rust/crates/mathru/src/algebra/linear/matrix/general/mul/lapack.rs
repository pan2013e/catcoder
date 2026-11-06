use crate::algebra::{
    abstr::{Field, Scalar},
    linear::{matrix::General, vector::Vector},
};
use std::ops::Mul;

/// Multiplies matrix by vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = Vector<T>;

    fn mul(self, v: &'b Vector<T>) -> Vector<T> {
        let (self_m, self_n): (usize, usize) = self.dim();
        let (v_m, v_n): (usize, usize) = v.dim();

        if self_n != v_m {
            panic!("Matrix and Vector dimension do not match");
        }

        let m = self_m as i32;
        let k = self_n as i32;
        let n = v_n as i32;

        let mut prod_data = Vec::with_capacity(self_m);
        unsafe { prod_data.set_len(self_m) }

        T::xgemm(
            'N' as u8,
            'N' as u8,
            m,
            n,
            k,
            T::one(),
            &self.data[..],
            m,
            &v.data.data[..],
            k,
            T::zero(),
            &mut prod_data[..],
            m,
        );

        Vector::new_column(prod_data)
    }
}

// Multiplies matrix by vector.
impl<T> Mul<Vector<T>> for General<T>
where
    T: Field + Scalar,
{
    type Output = Vector<T>;

    fn mul(self, v: Vector<T>) -> Vector<T> {
        (&self) * (&v)
    }
}

impl<T> Mul<General<T>> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let res_ref: General<f64> = General::new(2, 2, vec![1.0, 0.0, -18.0, 49.0]);
    /// assert_eq!(res_ref, a * b);
    /// ```
    fn mul(self, rhs: Self) -> Self::Output {
        (&self).mul(&rhs)
    }
}

impl<'a, 'b, T> Mul<&'b General<T>> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let res_ref: General<f64> = General::new(2, 2, vec![1.0, 0.0, -18.0, 49.0]);
    /// assert_eq!(res_ref, &a * &b);
    /// ```
    fn mul(self, rhs: &'b General<T>) -> Self::Output {
        let (self_rows, self_cols) = self.dim();
        let (rhs_rows, rhs_cols) = rhs.dim();

        debug_assert_eq!(self_cols, rhs_rows);

        let m = self_rows as i32;
        let n = rhs_cols as i32;
        let k = self_cols as i32;
        let mut c: General<T> = General::zero(m as usize, n as usize);

        T::xgemm(
            'N' as u8,
            'N' as u8,
            m,
            n,
            k,
            T::one(),
            &self.data[..],
            m,
            &rhs.data[..],
            k,
            T::zero(),
            &mut c.data[..],
            m,
        );

        return c;
    }
}

impl<'a, 'b, T> Mul<&'b General<T>> for &'a mut General<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut General<T>;

    /// Multiplies two matrices
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let b: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let res_ref: General<f64> = General::new(2, 2, vec![1.0, 0.0, -18.0, 49.0]);
    /// assert_eq!(res_ref, *(&mut a * &b));
    /// ```
    fn mul(self, rhs: &'b General<T>) -> Self::Output {
        let (self_rows, self_cols) = self.dim();
        let (rhs_rows, rhs_cols) = rhs.dim();

        debug_assert_eq!(self_cols, rhs_rows);

        let m = self_rows as i32;
        let n = rhs_cols as i32;
        let k = self_cols as i32;

        T::xgemm(
            'N' as u8,
            'N' as u8,
            m,
            n,
            k,
            T::one(),
            &self.data.clone()[..],
            m,
            &rhs.data[..],
            k,
            T::zero(),
            &mut self.data[..],
            m,
        );

        self.data.truncate(self_rows * rhs_cols);
        self.m = self_rows;
        self.n = rhs_cols;
        self
    }
}

impl<'a, 'b, T> General<T>
where
    T: Field + Scalar,
{
    fn mul_scalar(mut self, s: &'b T) -> General<T> {
        let (rows, cols): (usize, usize) = self.dim();
        //
        let m: i32 = rows as i32;
        let n: i32 = cols as i32;

        T::xscal(m * n, *s, &mut self.data[..], 1);
        return self;
    }
}

//Multiplies matrix by scalar
impl<T> Mul<T> for General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Multiplies a matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let f: f64 = 7.0;
    /// let res_ref: General<f64> = General::new(2, 2, vec![7.0, 0.0, 21.0, -49.0]);
    ///
    /// assert_eq!(res_ref, a * f);
    /// ```
    fn mul(self, s: T) -> General<T> {
        self.mul_scalar(&s)
    }
}

// Multiplies matrix by scalar
impl<'a, 'b, T> Mul<&'b T> for &'a General<T>
where
    T: Field + Scalar,
{
    type Output = General<T>;

    /// Multiplies a matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let res_ref: General<f64> = General::new(2, 2, vec![4.0, 0.0, 12.0, -28.0]);
    ///
    /// assert_eq!(res_ref, &a * &4.0);
    /// ```
    fn mul(self, m: &'b T) -> General<T> {
        return self.clone().mul_scalar(m);
    }
}

// Multiplies matrix by scalar
impl<'a, 'b, T> Mul<&'b T> for &'a mut General<T>
where
    T: Field + Scalar,
{
    type Output = &'a mut General<T>;

    /// Multiplies a matrix with a scalar
    ///
    /// # Example
    ///
    /// ```
    /// use mathru::algebra::linear::matrix::General;
    ///
    /// let mut a: General<f64> = General::new(2, 2, vec![1.0, 0.0, 3.0, -7.0]);
    /// let res_ref: General<f64> = General::new(2, 2, vec![4.0, 0.0, 12.0, -28.0]);
    ///
    /// assert_eq!(res_ref, *(&mut a * &4.0));
    /// ```
    fn mul(self, m: &'b T) -> Self::Output {
        let _ = self.data.iter_mut().for_each(&|a: &mut T| *a *= *m);
        self
    }
}
