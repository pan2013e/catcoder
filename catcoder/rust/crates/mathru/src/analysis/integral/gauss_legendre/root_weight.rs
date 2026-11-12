use crate::algebra::abstr::Real;
use std::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::root_weight_iterator::RootWeightIterator;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct RootWeight<T> {
    pub roots: Vec<T>,
    pub weights: Vec<T>,
}

const R1: [f64; 1] = [0.0];
const R2: [f64; 1] = [0.577350269189626];
const R3: [f64; 2] = [0.0, 0.774596669241483];
const R4: [f64; 2] = [0.339981043584856, 0.861136311594053];
const R5: [f64; 3] = [0.0, 0.538469310105683, 0.906179845938664];
const R6: [f64; 3] = [0.238619186083197, 0.661209386466265, 0.932469514203152];
const R7: [f64; 4] = [0.0, 0.405845151377397, 0.741531185599394, 0.949107912342759];
const R8: [f64; 4] = [
    0.183434642495650,
    0.525532409916329,
    0.796666477413627,
    0.960289856497536,
];
const R9: [f64; 5] = [
    0.0,
    0.324253423403809,
    0.613371432700590,
    0.836031107326636,
    0.968160239507626,
];

const ROOTS: [&[f64]; 9] = [&R1, &R2, &R3, &R4, &R5, &R6, &R7, &R8, &R9];

const W1: [f64; 1] = [2.0];
const W2: [f64; 1] = [1.0];
const W3: [f64; 2] = [0.888888888888889, 0.555555555555556];
const W4: [f64; 2] = [0.652145154862546, 0.347854845137454];
const W5: [f64; 3] = [0.568888888888889, 0.478628670499366, 0.236926885056189];
const W6: [f64; 3] = [0.467913934572691, 0.360761573048139, 0.171324492379170];
const W7: [f64; 4] = [
    0.417959183673469,
    0.381830050505119,
    0.279705391489277,
    0.129484966168870,
];
const W8: [f64; 4] = [
    0.362683783378362,
    0.313706645877887,
    0.222381034453374,
    0.101228536290376,
];
const W9: [f64; 5] = [
    0.330239355001260,
    0.312347077040003,
    0.260610696402935,
    0.180648160694857,
    0.081274388361574,
];

const WEIGHTS: [&[f64]; 9] = [&W1, &W2, &W3, &W4, &W5, &W6, &W7, &W8, &W9];

impl<T> RootWeight<T>
where
    T: Real,
{
    ///
    /// Panics
    /// if n == 0
    ///
    pub fn new(n: u8) -> RootWeight<T> {
        debug_assert!(n != 0, "n == 0 is not allowed");

        let roots = Self::roots(n);

        let weights = Self::weights(n);

        RootWeight { roots, weights }
    }

    pub fn iter(&self) -> RootWeightIterator<T> {
        RootWeightIterator::new(&self)
    }

    fn roots(n: u8) -> Vec<T> {
        let mut num_roots = (n as f32 / 2.0).ceil() as usize;

        let roots = if n < 10 {
            ROOTS[n as usize - 1]
                .iter()
                .map(|r| T::from_f64(*r))
                .collect::<Vec<T>>()
        } else {
            let mut rts = Vec::with_capacity(num_roots);
            if n % 2 == 1 {
                rts.push(T::zero());
                num_roots -= 1;
            }

            for k in (1..=num_roots as u8).rev() {
                let root = Self::theta(n, k);
                rts.push(root.cos());
            }
            rts
        };

        roots
    }

    fn weights(n: u8) -> Vec<T> {
        let num_roots = (n as f32 / 2.0).ceil() as usize;

        let weights = if n < 10 {
            WEIGHTS[n as usize - 1]
                .iter()
                .map(|r| T::from_f64(*r))
                .collect::<Vec<T>>()
        } else {
            let mut wts = Vec::with_capacity(num_roots);

            for k in (1..=num_roots as u8).rev() {
                let weight = Self::beta(n, k);
                wts.push(T::from(2.0) / weight);
            }
            wts
        };

        weights
    }

    pub fn theta(n: u8, k: u8) -> T {
        let v_n = T::one() / (T::from_u8(n) + T::from_f32(0.5));
        let alpha_n_k = Self::alpha(v_n, k);
        let cosi = alpha_n_k.cos() / alpha_n_k.sin();
        let v_n_2 = v_n * v_n;

        Self::F
            .iter()
            .fold(T::zero(), |s, f| s * v_n_2 + f(alpha_n_k, cosi))
    }

    const F: [fn(T, T) -> T; 6] = [
        Self::f_5,
        Self::f_4,
        Self::f_3,
        Self::f_2,
        Self::f_1,
        Self::f_0,
    ];

    fn f_0(x: T, _u: T) -> T {
        x
    }

    fn f_1(x: T, u: T) -> T {
        (u * x - T::one()) / (T::from_u8(8) * x)
    }

    fn f_2(x: T, u: T) -> T {
        let u_sq = u * u;
        let x_sq = x * x;
        let x_q = x_sq * x;
        (T::from_u8(6) * x_sq * (T::one() + u_sq) + T::from_u8(25)
            - u * (T::from_u8(31) * u_sq + T::from_u8(33)) * x_q)
            / (T::from_u16(384) * x_q)
    }

    fn f_3(x: T, u: T) -> T {
        let x_inv = T::one() / x;
        let x_inv_5 = x_inv * x_inv * x_inv * x_inv * x_inv;

        Self::r_30(u)
            + Self::r_35(u) * x_inv_5
            + (T::one() + u * u) * Self::R_3.iter().fold(T::zero(), |s, r| s * x_inv + r(u))
    }

    fn f_4(x: T, u: T) -> T {
        let x_inv = T::one() / x;
        let x_inv_2 = x_inv * x_inv;
        let x_inv_7 = x_inv_2 * x_inv_2 * x_inv_2 * x_inv;

        Self::r_40(u)
            + Self::r_47(u) * x_inv_7
            + (T::one() + u * u) * Self::R_4.iter().fold(T::zero(), |s, r| s * x_inv + r(u))
    }

    fn f_5(x: T, u: T) -> T {
        let x_inv = T::one() / x;
        let x_inv_3 = x_inv * x_inv * x_inv;
        let x_inv_9 = x_inv_3 * x_inv_3 * x_inv_3;

        Self::r_50(u)
            + Self::r_59(u) * x_inv_9
            + (T::one() + u * u) * Self::R_5.iter().fold(T::zero(), |s, r| s * x_inv + r(u))
    }

    fn alpha(v_n: T, k: u8) -> T {
        v_n * Self::bessel_0(k)
    }

    pub fn bessel_0(k: u8) -> T {
        return if k > 20 {
            let a_k = T::pi() * (T::from_u8(k) - T::from_f32(0.25));
            let r = T::from_f64(1.0 / 8.0) / a_k;
            let d = r * r;

            let s = a_k
                + r * (T::one()
                    - d * (T::from_f64(124.0 / 3.0)
                        + d * (T::from_f64(120928.0 / 15.0)
                            - d * (T::from_f64(401743168.0 / 105.0)
                                + T::from_f64(1071187749376.0 / 315.0) * d))));
            s
        } else {
            T::from_f64(Self::JZ[(k - 1) as usize])
        };
    }

    const JZ: [f64; 20] = [
        2.40482555769577276862163187933,
        5.5200781102863106495966041128,
        8.65372791291101221695419871266,
        11.7915344390142816137430449119,
        14.9309177084877859477625939974,
        18.0710639679109225431478829756,
        21.2116366298792589590783933505,
        24.3524715307493027370579447632,
        27.4934791320402547958772882346,
        30.6346064684319751175495789269,
        33.7758202135735686842385463467,
        36.9170983536640439797694930633,
        40.0584257646282392947993073740,
        43.1997917131767303575240727287,
        46.3411883716618140186857888791,
        49.4826098973978171736027615332,
        52.6240518411149960292512853804,
        55.7655107550199793116834927735,
        58.9069839260809421328344066346,
        62.0484691902271698828525002646,
    ];

    const R_3: [fn(T) -> T; 3] = [Self::r_33, Self::r_32, Self::r_31];

    fn r_30(u: T) -> T {
        let u_sq = u * u;

        u * (T::from_u16(2595) + u_sq * (T::from_u16(6350) + T::from_u16(3779) * u_sq))
            / T::from_u16(15360)
    }

    fn r_31(u: T) -> T {
        let u_sq = u * u;
        -(T::from_u8(31) * u_sq + T::from_u8(11)) / T::from_u16(1024)
    }

    fn r_32(u: T) -> T {
        u / T::from_u16(512)
    }

    fn r_33(_u: T) -> T {
        T::from_i8(-25) / T::from_u16(3072)
    }

    fn r_35(_u: T) -> T {
        T::from_i16(-1073) / T::from_u16(5120)
    }

    const R_4: [fn(T) -> T; 5] = [Self::r_45, Self::r_44, Self::r_43, Self::r_42, Self::r_41];

    fn r_40(u: T) -> T {
        let u_sq = u * u;

        -(((T::from_u32(6277237) * u_sq + T::from_u32(14682157)) * u_sq + T::from_u32(10808595))
            * u_sq
            + T::from_u32(2407755))
            * u
            / T::from_u32(3440640)
    }

    fn r_41(u: T) -> T {
        let u_sq = u * u;

        ((T::from_u16(3779) * u_sq + T::from_u16(3810)) * u_sq + T::from_u16(519))
            / T::from_u16(24576)
    }

    fn r_45(_u: T) -> T {
        T::from_u16(1073) / T::from_u32(40960)
    }

    fn r_44(u: T) -> T {
        T::from_i8(-25) / T::from_u16(12288) * u
    }

    fn r_43(u: T) -> T {
        (T::from_u16(787) * u * u + T::from_u16(279)) / T::from_u32(49152)
    }

    fn r_42(u: T) -> T {
        let u_sq = u * u;
        -(T::from_i8(21) + T::from_u8(31) * u_sq) * u / T::from_u16(4096)
    }

    fn r_47(_u: T) -> T {
        T::from_u32(375733) / T::from_u32(229376)
    }

    const R_5: [fn(T) -> T; 7] = [
        Self::r_57,
        Self::r_56,
        Self::r_55,
        Self::r_54,
        Self::r_53,
        Self::r_52,
        Self::r_51,
    ];

    fn r_50(u: T) -> T {
        let u_sq = u * u;
        ((((T::from_u32(2092163573) * u_sq + T::from_u64(6282767956)) * u_sq
            + T::from_u64(6710945598))
            * u_sq
            + T::from_u32(2935744980))
            * u_sq
            + T::from_u32(415542645))
            * u
            / T::from_u32(82575360)
    }

    fn r_51(u: T) -> T {
        let u_sq = u * u;
        let d = ((T::from_u32(6277237) * u_sq + T::from_u32(10487255)) * u_sq
            + T::from_u32(4632255))
            * u_sq
            + T::from_u32(343965);

        -d / T::from_u32(3932160)
    }

    fn r_52(u: T) -> T {
        let u_sq = u * u;
        ((T::from_u16(15178) + T::from_u16(11337) * u_sq) * u_sq + T::from_u16(4329)) * u
            / T::from_u32(196608)
    }

    fn r_53(u: T) -> T {
        let u_sq = u * u;
        -((T::from_u32(96335) * u_sq + T::from_u32(97122)) * u_sq + T::from_u16(13227))
            / T::from_u32(1179648)
    }

    fn r_54(u: T) -> T {
        let u_sq = u * u;
        (T::from_u16(778) * u_sq + T::from_u16(527)) * u / T::from_u32(98304)
    }

    fn r_55(u: T) -> T {
        let u_sq = u * u;
        -(T::from_u32(100539) * u_sq + T::from_u16(35659)) / T::from_u32(1966080)
    }

    fn r_56(u: T) -> T {
        T::from_f64(41753.0 / 5898240.0) * u
    }

    fn r_57(_u: T) -> T {
        T::from_f64(-375733.0 / 1835008.0)
    }

    fn r_59(_u: T) -> T {
        T::from_f64(-55384775.0 / 2359296.0)
    }

    pub fn beta(n: u8, k: u8) -> T {
        let v_n = T::one() / (T::from_u8(n) + T::from_f32(0.5));
        let alpha_n_k = Self::alpha(v_n, k);
        let sin = alpha_n_k.sin();
        let cosi = alpha_n_k.cot();
        let v_n_2 = v_n * v_n;

        let j_k_2 = Self::bessel_1_sq(k);

        j_k_2 / v_n_2 * alpha_n_k / sin
            * Self::W.iter().fold(T::zero(), |s, w| {
                let value = w(alpha_n_k, cosi);
                s * v_n_2 + value
            })
    }

    fn bessel_1_sq(k: u8) -> T {
        return if k > 21 {
            let a_k = T::pi() * (T::from_u8(k) - T::from_f32(0.25));
            let r = T::from_f64(1.0 / 8.0) / a_k;
            let d = r * r;

            let s = T::from_f32(2.0)
                + d * d
                    * (T::from_f64(-7.0 / 24.0)
                        + d * (T::from_f64(151.0 / 80.0)
                            * d
                            * (T::from_f64(-172913.0 / 8064.0)
                                + d * (T::from_f64(461797.0 / 1152.0)
                                    + d * T::from_f64(-171497088497.0 / 15206400.0)))));
            s / (T::pi() * a_k)
        } else {
            T::from_f64(Self::J1[(k - 1) as usize])
        };
    }

    const W: [fn(T, T) -> T; 6] = [
        Self::w_5,
        Self::w_4,
        Self::w_3,
        Self::w_2,
        Self::w_1,
        Self::w_0,
    ];

    fn w_0(_x: T, _u: T) -> T {
        T::one()
    }

    fn w_1(x: T, u: T) -> T {
        (x * (u + x) - T::one()) / (T::from_u8(8) * x * x)
    }

    fn w_2(x: T, u: T) -> T {
        let u_sq = u * u;
        let x_sq = x * x;
        let x_4 = x_sq * x_sq;
        let a = T::from_u8(81) + (T::from_u8(6) * x_sq - T::from_u8(31)) * u * x
            - (T::from_u8(3) - T::from_u8(6) * u_sq) * x_sq
            - (T::from_u8(27) + (T::from_u8(84) + T::from_u8(56) * u_sq) * u_sq) * x_4;

        a / (T::from_u16(384) * x_4)
    }

    fn w_3(x: T, u: T) -> T {
        let x_inv = T::one() / x;
        Self::Q_3.iter().fold(T::zero(), |s, q| s * x_inv + q(u))
    }

    fn w_4(x: T, u: T) -> T {
        let x_inv = T::one() / x;
        Self::Q_4.iter().fold(T::zero(), |s, q| s * x_inv + q(u))
    }

    fn w_5(x: T, u: T) -> T {
        let x_inv = T::one() / x;
        Self::Q_5.iter().fold(T::zero(), |s, q| s * x_inv + q(u))
    }

    const J1: [f64; 21] = [
        0.269514123941916926139021992911,
        0.115780138582203695807812836182,
        0.0736863511364082151406476811985,
        0.0540375731981162820417749182758,
        0.0426614290172430912655106063495,
        0.0352421034909961013587473033648,
        0.0300210701030546726750888157688,
        0.0261473914953080885904584675399,
        0.0231591218246913922652676382178,
        0.0207838291222678576039808057297,
        0.0188504506693176678161056800214,
        0.0172461575696650082995240053542,
        0.0158935181059235978027065594287,
        0.0147376260964721895895742982592,
        0.0137384651453871179182880484134,
        0.0128661817376151328791406637228,
        0.0120980515486267975471075438497,
        0.0114164712244916085168627222986,
        0.0108075927911802040115547286830,
        0.0102603729262807628110423992790,
        0.00976589713979105054059846736696,
    ];

    const Q_3: [fn(T) -> T; 7] = [
        Self::q_36,
        Self::q_35,
        Self::q_34,
        Self::q_33,
        Self::q_32,
        Self::q_31,
        Self::q_30,
    ];

    fn q_30(u: T) -> T {
        let u_sq = u * u;
        ((T::from_f64(151.0 / 160.0) * u_sq + T::from_f64(187.0 / 96.0)) * u_sq
            + T::from_f64(295.0 / 256.0))
            * u_sq
            + T::from_f64(153.0 / 1024.0)
    }

    fn q_31(u: T) -> T {
        let u_sq = u * u;
        (((T::from_f64(-35.0 / 384.0) * u_sq - T::from_f64(119.0 / 768.0)) * u_sq)
            - T::from_f64(65.0 / 1024.0))
            * u
    }

    fn q_32(u: T) -> T {
        let u_sq = u * u;
        (T::from_f64(7.0 / 384.0) * u_sq + T::from_f64(15.0 / 512.0)) * u_sq
            + T::from_f64(5.0 / 512.0)
    }

    fn q_33(u: T) -> T {
        let u_sq = u * u;
        (T::from_f64(1.0 / 512.0) * u_sq - T::from_f64(13.0 / 1536.0)) * u
    }

    fn q_34(u: T) -> T {
        let u_sq = u * u;
        T::from_f64(-7.0 / 384.0) * u_sq + T::from_f64(53.0 / 3072.0)
    }
    fn q_35(u: T) -> T {
        T::from_f64(3749.0 / 15360.0) * u
    }

    fn q_36(_u: T) -> T {
        T::from_f64(-1125.0 / 1024.0)
    }

    const Q_4: [fn(T) -> T; 9] = [
        Self::q_48,
        Self::q_47,
        Self::q_46,
        Self::q_45,
        Self::q_44,
        Self::q_43,
        Self::q_42,
        Self::q_41,
        Self::q_40,
    ];

    fn q_40(u: T) -> T {
        let u_sq = u * u;

        (((T::from_f64(-3626248438009.0 / 338228674560.0) * u_sq
            - T::from_f64(669667.0 / 23040.0))
            * u_sq
            - T::from_f64(27351.0 / 1024.0))
            * u_sq
            - T::from_f64(36941.0 / 4096.0))
            * u_sq
            - T::from_f64(21429.0 / 32768.0)
    }

    fn q_41(u: T) -> T {
        let u_sq = u * u;

        (((T::from_f64(997510355.0 / 1207959552.0) * u_sq + T::from_f64(7393.0 / 3840.9)) * u_sq
            + T::from_f64(8639.0 / 6144.0))
            * u_sq
            + T::from_f64(2513.0 / 8192.0))
            * u
    }

    fn q_42(u: T) -> T {
        let u_sq = u * u;

        ((T::from_f64(-1837891769.0 / 12079595520.0) * u_sq - T::from_f64(1909.0 / 6144.0)) * u_sq
            - T::from_f64(1483.0 / 8192.0))
            * u_sq
            - T::from_f64(371.0 / 16384.0)
    }

    fn q_43(u: T) -> T {
        let u_sq = u * u;
        ((T::from_f64(355532953.0 / 6039797760.0) * u_sq + T::from_f64(1849.0 / 18432.0)) * u_sq
            + T::from_f64(675.0 / 16384.0))
            * u
    }

    fn q_44(u: T) -> T {
        let u_sq = u * u;
        (T::from_f64(-147456121.0 / 4831838208.0) * u_sq - T::from_f64(1183.0 / 24576.0)) * u_sq
            - T::from_f64(1565.0 / 98304.0)
    }

    fn q_45(u: T) -> T {
        (T::from_f64(-19906471.0 / 6039797760.0) * u * u + T::from_f64(6823.0 / 245760.0)) * u
    }

    fn q_46(u: T) -> T {
        let u_sq = u * u;
        T::from(149694043.0 / 2415919104.0) * u_sq - T::from_f64(156817.0 / 1474560.0)
    }

    fn q_47(u: T) -> T {
        T::from_f64(-76749336551.0 / 42278584320.0) * u
    }

    fn q_48(_u: T) -> T {
        T::from_f64(568995840001.0 / 48318382080.0)
    }

    const Q_5: [fn(T) -> T; 11] = [
        Self::q_510,
        Self::q_59,
        Self::q_58,
        Self::q_57,
        Self::q_56,
        Self::q_55,
        Self::q_54,
        Self::q_53,
        Self::q_52,
        Self::q_51,
        Self::q_50,
    ];

    fn q_50(u: T) -> T {
        let u_sq = u * u;
        ((((T::from_f64(97620617026363819.0 / 487049291366400.0) * u_sq
            + T::from_f64(202966472595331.0 / 300647710720.0))
            * u_sq
            + T::from_f64(17266857.0 / 20480.0))
            * u_sq
            + T::from_f64(22973795.0 / 49152.0))
            * u_sq
            + T::from_f64(3401195.0 / 32768.0))
            * u_sq
            + T::from_f64(1268343.0 / 262144.0)
    }

    fn q_51(u: T) -> T {
        let u_sq = u * u;
        ((((T::from_f64(-65272472659909.0 / 5411658792960.0) * u_sq
            - T::from_f64(2717368577869.0 / 75161927680.0))
            * u_sq
            - T::from(4729993.0 / 122880.0))
            * u_sq
            - T::from_f64(548439.0 / 32768.0))
            * u_sq
            - T::from_f64(612485.0 / 262144.0))
            * u
    }
    fn q_52(u: T) -> T {
        let u_sq = u * u;
        (((T::from_f64(6324614896949.0 / 3607772528640.0) * u_sq
            + T::from_f64(45578037433.0 / 9663676416.0))
            * u_sq
            + T::from_f64(52739.0 / 12288.0))
            * u_sq
            + T::from_f64(93673.0 / 65536.0))
            * u_sq
            + T::from_f64(26455.0 / 262144.0)
    }

    fn q_53(u: T) -> T {
        let u_sq = u * u;
        (((T::from_f64(-63001776779.0 / 115964116992.0) * u_sq
            - T::from_f64(40779010513.0 / 32212254720.0))
            * u_sq
            - T::from_f64(181651.0 / 196608.0))
            * u_sq
            - T::from_f64(19795.0 / 98304.0))
            * u
    }

    fn q_54(u: T) -> T {
        let u_sq = u * u;
        ((T::from_f64(184730261873.0 / 773094113280.0) * u_sq
            + T::from_f64(2101713105.0 / 4294967296.0))
            * u_sq
            + T::from_f64(56281.0 / 196608.0))
            * u_sq
            + T::from_f64(9477.0 / 262144.0)
    }

    fn q_55(u: T) -> T {
        let u_sq = u * u;
        ((T::from_f64(-38212677741.0 / 214748364800.0) * u_sq
            - T::from_f64(29273066033.0 / 96636764160.0))
            * u_sq
            - T::from_f64(488659.0 / 3932160.0))
            * u
    }

    fn q_56(u: T) -> T {
        let u_sq = u * u;
        (T::from_f64(370339107271.0 / 2319282339840.0) * u_sq
            + T::from_f64(996334037.0 / 4026531840.0))
            * u_sq
            + T::from_f64(39817.0 / 491520.0)
    }

    fn q_57(u: T) -> T {
        let u_sq = u * u;
        (T::from_f64(16514308061.0 / 1352914698240.0) * u_sq
            - T::from_f64(3258170891.0 / 15032385536.0))
            * u
    }

    fn q_58(u: T) -> T {
        T::from_f64(3354565639447.0 / 2705829396480.0)
            - T::from_f64(335149450411.0 / 721554505728.0) * u * u
    }

    fn q_59(u: T) -> T {
        T::from_f64(1230657354291011.0 / 48704929136640.0) * u
    }
    fn q_510(_u: T) -> T {
        T::from_f64(-553063956480229.0 / 2576980377600.0)
    }
}
