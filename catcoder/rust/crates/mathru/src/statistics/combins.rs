/// Factorial
///
/// <https://en.wikipedia.org/wiki/Factorial>
pub fn factorial(f: u32) -> u64 {
    let mut n: u64 = f as u64;
    let mut result: u64 = 1;
    while n > 0 {
        result *= n;
        n -= 1;
    }
    result
}

/// Binomial coefficient
///
/// <https://en.wikipedia.org/wiki/Binomial_coefficient>
pub fn binom(n: u32, k: u32) -> u32 {
    (factorial(n) / (factorial(k) * factorial(n - k))) as u32
}

pub fn perm(n: u32, r: u32) -> u32 {
    let n_fact: u64 = factorial(n);
    (n_fact / factorial(n - r)) as u32
}

pub fn comb<'a>(n: u32, r: u32) -> u32 {
    let p = perm(n, r);
    p / (factorial(r) as u32)
}
