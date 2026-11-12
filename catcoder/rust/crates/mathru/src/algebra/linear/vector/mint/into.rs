use crate::algebra::abstr::Real;
use crate::algebra::linear::vector::Vector;
use mint::{Vector2, Vector3, Vector4};

macro_rules! impl_into_mint(
    ($($rows: literal => $MV: ident, $data: ident, $ret_column: expr, $ret_row: expr);* $(;)*) => {$(

        impl<T> Into<$MV<T>> for Vector<T>
            where T: Real
        {
            fn into(self) -> $MV<T> {
                let (m, n) = self.dim();
                let $data = self.data;
                if (m != 1 && n != $rows) && (m != $rows && n != 1) {
                    panic!("Vector can not be converted into a $MV because it is not a 1 by {} or {} by 1 vector", $rows, $rows)
                }

                if m == 1 {
                    $ret_row
                } else {
                    $ret_column
                }
            }
        }
    )*}
);

impl_into_mint!(
    2 => Vector2, data, Vector2{ x: data[[0, 0]], y: data[[1, 0]]}, Vector2{ x: data[[0, 0]], y: data[[0, 1]]};
    3 => Vector3, data, Vector3{ x: data[[0, 0]], y: data[[1, 0]], z: data[[2, 0]]}, Vector3{ x: data[[0, 0]], y: data[[0, 1]], z: data[[0, 2]]};
    4 => Vector4, data, Vector4{ x: data[[0, 0]], y: data[[1, 0]], z: data[[2, 0]], w: data[[3, 0]]}, Vector4{ x: data[[0, 0]], y: data[[0, 1]], z: data[[0, 2]], w: data[[0, 3]]};
);
