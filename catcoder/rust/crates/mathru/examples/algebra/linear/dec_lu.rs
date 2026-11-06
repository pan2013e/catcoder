use mathru::algebra::linear::matrix::{General, UnitLowerTriangular, UpperTriangular};
use mathru::matrix;

fn main() {
    let a: General<f64> = matrix![  1.0, -2.0, 3.0;
                                    2.0, -5.0, 12.0;
                                    0.0, 2.0, -10.0];

    let (_l, _u, _p): (UnitLowerTriangular<f64>, UpperTriangular<f64>, General<f64>) =
        a.dec_lu().unwrap().lup();
}
