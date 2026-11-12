use crate::algebra::{
    abstr::{Field, Scalar, Zero},
    linear::{matrix::General, vector::Vector},
};

use crate::algebra::linear::matrix::Solve;

impl<T> Solve<Vector<T>> for General<T>
where
    T: Field + Scalar,
{
    /// Solves $Ax = y$
    /// where $A \in \mathbb{R}^{m * n}, x \in \mathbb{R}^n, y \in \mathbb{R}^m$
    fn solve(&self, rhs: &Vector<T>) -> Result<Vector<T>, ()> {
        return self.solve_vector_r(rhs);
    }
}

impl<T> Solve<General<T>> for General<T>
where
    T: Field + Scalar,
{
    fn solve(&self, rhs: &General<T>) -> Result<General<T>, ()> {
        return self.solve_matrix_r(rhs);
    }
}

impl<T> General<T>
where
    T: Field + Scalar,
{
    fn solve_vector_r(&self, y: &Vector<T>) -> Result<Vector<T>, ()> {
        let (m, n): (usize, usize) = self.dim();
        let m_i32: i32 = m as i32;
        let n_i32: i32 = n as i32;

        let (y_m, _y_n): (usize, usize) = y.dim();
        let y_m_i32: i32 = y_m as i32;

        let dim_min: i32 = m_i32.min(n_i32);
        let mut ipiv: Vec<i32> = vec![Zero::zero(); dim_min as usize];

        let mut info: i32 = 0;

        let mut self_data: Vec<T> = self.clone().data;
        let mut y_data: Vec<T> = y.clone().convert_to_vec();

        T::xgetrf(
            m_i32,
            n_i32,
            self_data.as_mut_slice(),
            m_i32,
            ipiv.as_mut_slice(),
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        T::xgetrs(
            m_i32,
            1,
            self_data.as_mut_slice(),
            n_i32,
            ipiv.as_mut_slice(),
            y_data.as_mut_slice(),
            y_m_i32,
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        Ok(Vector::new_column(y_data))
    }
}

impl<T> General<T>
where
    T: Field + Scalar,
{
    pub fn solve_matrix_r(&self, y: &General<T>) -> Result<General<T>, ()> {
        let (m, n): (usize, usize) = self.dim();
        let m_i32: i32 = m as i32;
        let n_i32: i32 = n as i32;

        let (y_m, y_n): (usize, usize) = y.dim();
        let y_n_i32: i32 = y_n as i32;

        let dim_min: i32 = m_i32.min(n_i32);
        let mut ipiv: Vec<i32> = vec![Zero::zero(); dim_min as usize];

        let mut info: i32 = 0;

        let mut self_data: Vec<T> = self.clone().data;
        let mut y_data: Vec<T> = y.clone().convert_to_vec();

        T::xgetrf(
            m_i32,
            n_i32,
            self_data.as_mut_slice(),
            m_i32,
            ipiv.as_mut_slice(),
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        T::xgetrs(
            n_i32,
            y_n_i32,
            self_data.as_mut_slice(),
            m_i32,
            ipiv.as_mut_slice(),
            y_data.as_mut_slice(),
            y_n_i32,
            &mut info,
        );

        if info != 0 {
            return Err(());
        }

        Ok(General::new(y_m, y_n, y_data))
    }
}
