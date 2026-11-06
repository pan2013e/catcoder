use mathru::algebra::linear::matrix::{General, UpperTriangular};
use mathru::matrix;

fn main() {
    let a: General<f64> = matrix![  6.0, 5.0, 0.0;
                                    5.0, 1.0, 4.0;
                                    0.0, 4.0, 3.0];

    let (_q, _r): (General<f64>, UpperTriangular<f64>) = a.dec_qr().unwrap().qr();
}
