use mathru::algebra::linear::matrix::{General, HessenbergDecomposition, UpperHessenberg};
use mathru::matrix;

fn main() {
    let a: General<f64> = matrix![  1.0, 5.0, 3.0;
                                    1.0, 0.0, -7.0;
                                    3.0, 8.0, 9.0];

    let (_q, _h): (General<f64>, UpperHessenberg<f64>) = a.dec_hessenberg().qh();
}
