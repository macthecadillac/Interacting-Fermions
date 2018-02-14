use fnv::FnvHashMap;
use num_complex::Complex;

use blochfunc::{BlochFunc, BlochFuncSet};
use common::*;

use super::PI;

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

pub fn ss_z_elements(sites: &(Vec<u32>, Vec<u32>), orig_state: &BlochFunc) -> f64 {
    let (ref site1, ref site2) = *sites;
    let mut same_dir = 0_i32;
    for (&s1, &s2) in site1.iter().zip(site2.iter()) {
        let (upup, downdown) = repeated_spins(orig_state.lead, s1, s2);
        if upup { same_dir += 1 };
        if downdown { same_dir += 1};
    }
    let diff_dir = site1.len() as i32 - same_dir;
    0.25 * (same_dir - diff_dir) as f64
}

#[allow(non_snake_case)]
pub fn ss_pm_elements(J: f64, sites: &(Vec<u32>, Vec<u32>),
                      orig_state: &BlochFunc,
                      dec_to_ind: &FnvHashMap<u32, u32>,
                      hashtable: &FnvHashMap<&u32, &BlochFunc>
                      ) -> FnvHashMap<u32, Complex<f64>> {
    let J = Complex::new(J, 0.);
    let mut j_element = FnvHashMap::default();
    let (ref site1, ref site2) = *sites;
    for (&s1, &s2) in site1.iter().zip(site2.iter()) {
        let (updown, downup) = exchange_spin_flips(orig_state.lead, s1, s2);
        let mut new_dec: u32;
        match (updown, downup) {
            (true, false) => new_dec = orig_state.lead - s1 + s2,
            (false, true) => new_dec = orig_state.lead + s1 - s2,
            _ => continue
        }
        match find_leading_state(new_dec, &hashtable) {
            None => (),
            Some((cntd_state, phase)) => {
                let j = *(dec_to_ind.get(&(cntd_state.lead)).unwrap());
                let coeff = phase * coeff(&orig_state, &cntd_state);

                let element = match j_element.get(&j) {
                    Some(&c) => c + J * coeff,
                    None     => J * coeff
                };
                j_element.insert(j, element);
            }
        }
    }
    j_element
}

#[allow(non_snake_case)]
pub fn ss_ppmm_elements(nx: u32, ny: u32, J: f64,
                        sites: &(Vec<u32>, Vec<u32>),
                        orig_state: &BlochFunc,
                        dec_to_ind: &FnvHashMap<u32, u32>,
                        hashtable: &FnvHashMap<&u32, &BlochFunc>
                        ) -> FnvHashMap<u32, Complex<f64>> {
    let J = Complex::new(J, 0.);
    let mut j_element = FnvHashMap::default();
    let (ref site1, ref site2) = *sites;
    for (&s1, &s2) in site1.iter().zip(site2.iter()) {
        let (upup, downdown) = repeated_spins(orig_state.lead, s1, s2);
        let mut new_dec: u32;
        let mut _gamma = Complex::new(0., 0.);
        match (upup, downdown) {
            (true, false) => {
                new_dec = orig_state.lead - s1 - s2;
                _gamma += gamma(nx, ny, s1, s2).conj();
            }
            (false, true) => {
                new_dec = orig_state.lead + s1 + s2;
                _gamma += gamma(nx, ny, s1, s2);
            }
            _ => continue
        }
        match find_leading_state(new_dec, &hashtable) {
            None => (),
            Some((cntd_state, phase)) => {
                let j = *(dec_to_ind.get(&(cntd_state.lead)).unwrap());
                let coeff = phase * coeff(&orig_state, &cntd_state);

                let element = match j_element.get(&j) {
                    Some(&c) => c + J * coeff * _gamma,
                    None     => J * coeff * _gamma
                };
                j_element.insert(j, element);
            }
        }
    }
    j_element
}

#[allow(non_snake_case)]
pub fn ss_pmz_elements(nx: u32, ny: u32, J: f64,
                       sites: &(Vec<u32>, Vec<u32>),
                       orig_state: &BlochFunc,
                       dec_to_ind: &FnvHashMap<u32, u32>,
                       hashtable: &FnvHashMap<&u32, &BlochFunc>,
                       ) -> FnvHashMap<u32, Complex<f64>> {
    let J = Complex::new(0., J);  // the entire operator was multiplied by i
    let mut j_element = FnvHashMap::default();
    let (ref site1, ref site2) = *sites;
    for (&s_1, &s_2) in site1.iter().zip(site2.iter()) {
        for &(s1, s2) in [(s_1, s_2), (s_2, s_1)].iter() {
            let z_contrib =
                if orig_state.lead | s1 == orig_state.lead { 0.5 } else { -0.5 };

            let mut new_dec: u32;
            let mut _gamma = Complex::new(0., 0.);
            if orig_state.lead | s2 == orig_state.lead {
                new_dec = orig_state.lead - s2;
                _gamma += gamma(nx, ny, s1, s2).conj();
            } else {
                new_dec = orig_state.lead + s2;
                _gamma -= gamma(nx, ny, s1, s2);
            }

            match find_leading_state(new_dec, &hashtable) {
                None => (),
                Some((cntd_state, phase)) => {
                    let j = *(dec_to_ind.get(&(cntd_state.lead)).unwrap());
                    let coeff = phase * coeff(&orig_state, &cntd_state);

                    let element = match j_element.get(&j) {
                        Some(&c) => c + J * z_contrib * coeff * _gamma,
                        None     => J * z_contrib * coeff * _gamma
                    };
                    j_element.insert(j, element);
                }
            }
        }
    }
    j_element
}
