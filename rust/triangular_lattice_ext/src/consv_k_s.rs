use num_complex::Complex;
use fnv::FnvHashMap;

use common::*;
use blochfunc::{BlochFunc, BlochFuncSet};
use super::PI;

fn decompose(i: i32, n: i32) -> Vec<i32> {
    // n is the maximum power of 2 that is needed to describe the system--
    // n = N - 1
    let mut v = Vec::with_capacity(n as usize);
    match i {
        -1 => (),
        o  => {
            let ord = 2_i32.pow(o as u32);
            v.push(i / ord);
            decompose(i % ord, o - 1);
        }
    }
    v
}

fn permute(i: u32, n: u32) -> u32 {
    // rewrite this inner function to use slice destruction pattern matching
    // when it becomes available in Rust stable
    fn aux(mut l: Vec<i32>, prev: i32, len: i32, n: i32) -> Vec<i32> {
        if l.len() == 0 { (n - len + 1..n).collect::<Vec<i32>>() }
        else {
            let hd = l.pop().unwrap();
            let mut tl = l;
            if hd - prev == 1 { aux(tl, hd, len, n) }
            else {
                let rl = len - tl.len() as i32 - 1;
                let nhd = hd - 1;
                tl.push(nhd);
                let mut pl = (nhd - rl..nhd).collect::<Vec<i32>>();
                pl.append(&mut tl);
                pl
            }
        }
    };
    let l = decompose(i as i32, n as i32);
    let len = l.len() as i32;
    let pows = aux(l, -1, len, n as i32);
    pows.into_iter()
        .map(|x| 2_u32.pow(x as u32))
        .sum()
}

pub fn bloch_states<'a>(nx: u32, ny: u32, kx: u32, ky: u32) -> BlochFuncSet {
    let n = nx * ny;
    let mut sieve = vec![true; 2_usize.pow(n)];
    let mut bfuncs: Vec<BlochFunc> = Vec::new();
    let phase = |i, j| {
        let r = 1.;
        let ang1 = 2. * PI * (i * kx) as f64 / nx as f64;
        let ang2 = 2. * PI * (j * ky) as f64 / ny as f64;
        Complex::from_polar(&r, &(ang1 + ang2))
    };

    for dec in 0..2_usize.pow(n) {
        if sieve[dec]
        {   // if the corresponding entry of dec in "sieve" is not false,
            // we find all translations of dec and put them in a BlochFunc
            // then mark all corresponding entries in "sieve" as false.

            // "decs" is a hashtable that holds vectors whose entries
            // correspond to Bloch function constituent configurations which
            // are mapped to single decimals that represent the leading states.
            let mut decs: FnvHashMap<u32, Complex<f64>> = FnvHashMap::default();
            // "new_dec" represents the configuration we are currently iterating
            // over.
            let mut new_dec = dec as u32;
            for j in 0..ny {
                for i in 0..nx {
                    sieve[new_dec as usize] = false;
                    let new_p = match decs.get(&new_dec) {
                        Some(&p) => p + phase(i, j),
                        None     => phase(i, j)
                    };
                    decs.insert(new_dec, new_p);
                    new_dec = translate_x(new_dec, nx, ny);
                }
                new_dec = translate_y(new_dec, nx, ny);
            }

            let lead = dec as u32;
            let norm = decs.values()
                .into_iter()
                .map(|&x| x.norm_sqr())
                .sum::<f64>()
                .sqrt();

            if norm > 1e-8 {
                let mut bfunc = BlochFunc { lead, decs, norm };
                bfuncs.push(bfunc);
            }
        }
    }

    let mut table = BlochFuncSet::create(bfuncs);
    table.sort();
    table
}
