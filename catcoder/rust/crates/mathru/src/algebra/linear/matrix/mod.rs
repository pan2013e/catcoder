#[macro_use]
//mod matrixcolumniterator;
//mod matrixcolumniteratormut;
mod matrixintoiterator;
mod matrixiterator;
mod matrixiteratormut;
//mod matrixrowiterator;
//mod matrixrowiteratormut;
mod eigendec;
mod matrixcolumnintoiterator;
mod matrixrowintoiterator;
pub use self::eigendec::{EigenDec, EigenDecomposition};

mod hessenbergdec;
pub use hessenbergdec::{HessenbergDec, HessenbergDecomposition};

mod ludec;
pub use ludec::LUDec;

mod qrdec;
pub use qrdec::QRDec;

mod schurdec;
pub use schurdec::SchurDec;

mod inverse;

mod choleskydec;
pub use choleskydec::CholeskyDec;

mod det;
pub use det::Determinant;

mod solve;
mod substitute;
mod transpose;

mod diagonal;
mod general;
mod lowertriangular;
mod unitlowertriangular;
mod unituppertriangular;
mod upperhessenberg;
mod uppertriangular;

pub use self::{
    diagonal::Diagonal,
    general::General,
    inverse::Inverse,
    lowertriangular::LowerTriangular,
    //matrixcolumniterator::MatrixColumnIterator,
    //matrixcolumniteratormut::MatrixColumnIteratorMut,
    matrixcolumnintoiterator::MatrixColumnIntoIterator,
    matrixintoiterator::MatrixIntoIterator,
    matrixiterator::MatrixIterator,
    matrixiteratormut::MatrixIteratorMut,
    //matrixrowiterator::MatrixRowIterator,
    //matrixrowiteratormut::MatrixRowIteratorMut,
    matrixrowintoiterator::MatrixRowIntoIterator,
    solve::Solve,
    substitute::{SubstituteBackward, SubstituteForward},
    transpose::Transpose,
    unitlowertriangular::UnitLowerTriangular,
    unituppertriangular::UnitUpperTriangular,
    upperhessenberg::UpperHessenberg,
    uppertriangular::UpperTriangular,
};
