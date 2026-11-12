use mathru::algebra::linear::matrix::General;
use mathru::algebra::linear::matrix::LowerTriangular;
use mathru::matrix;

fn main() {
    let a: General<f64> = matrix![  2.0, -1.0, 0.0;
                                	-1.0, 2.0, -1.0;
                                	0.0, -1.0,  2.0];

    let _l: LowerTriangular<f64> = a.dec_cholesky().unwrap().l();
}
