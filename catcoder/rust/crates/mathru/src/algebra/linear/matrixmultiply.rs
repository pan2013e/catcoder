use crate::algebra::abstr::Complex;
use crate::algebra::abstr::MatrixMultiply;
use matrixmultiply::{cgemm, dgemm, sgemm, zgemm, CGemmOption};

macro_rules! matrix_multiply_impl (
    ($T: ty, $xgemm: path, $cgemm: path)
    => (
        impl MatrixMultiply for $T
       	{
       		fn xgemm(
                m: usize,
                k: usize,
                n: usize,
                alpha: Self,
                a: *const Self,
                rsa: isize,
                csa: isize,
                b: *const Self,
                rsb: isize,
                csb: isize,
                beta: Self,
                c: *mut Self,
                rsc: isize,
                csc: isize
            )
			{
				unsafe
				{
					$xgemm(
						m,
                        k,
						n,
						alpha,
						a,
                        rsa,
                        csa,
						b,
						rsb,
                        csb,
						beta,
						c,
						rsc,
                        csc
						)
				}
			}
        }

        impl MatrixMultiply for Complex<$T>
       	{
       		fn xgemm(
                m: usize,
                k: usize,
                n: usize,
                alpha: Self,
                a: *const Self,
                rsa: isize,
                csa: isize,
                b: *const Self,
                rsb: isize,
                csb: isize,
                beta: Self,
                c: *mut Self,
                rsc: isize,
                csc: isize
            )
			{
				unsafe
				{
					$cgemm(
                        CGemmOption::Standard,
                        CGemmOption::Standard,
						m,
                        k,
						n,
						*(&alpha as *const _ as *const _),
						a as *const _ as *const _ ,
                        rsa,
                        csa,
						b as *const _ as *const _ ,
						rsb,
                        csb,
						*(&beta as *const _ as *const _),
						c as *mut _ as *mut _ ,
						rsc,
                        csc
						)
				}
			}
        }
    )
);

matrix_multiply_impl!(f32, sgemm, cgemm);
matrix_multiply_impl!(f64, dgemm, zgemm);
