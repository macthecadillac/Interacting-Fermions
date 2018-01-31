use fnv::FnvHashMap;
use num_complex::Complex;

use blochfunc::BlochFunc;
use common::*;

pub fn ss_z_elements(sites: &(Vec<u32>, Vec<u32>), orig_state: &BlochFunc) -> f64 {
    let (ref site1, ref site2) = *sites;
    let mut same_dir = 0_i32;
    for (&s1, &s2) in site1.iter().zip(site2.iter()) {
        let (upup, downdown) = _repeated_spins(orig_state.lead, s1, s2);
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
        let (updown, downup) = _exchange_spin_flips(orig_state.lead, s1, s2);
        let mut new_dec: u32;
        match (updown, downup) {
            (true, false) => new_dec = orig_state.lead - s1 + s2,
            (false, true) => new_dec = orig_state.lead + s1 - s2,
            _ => continue
        }
        match _find_leading_state(new_dec, &hashtable) {
            Some((cntd_state, phase)) => {
                let j = *(dec_to_ind.get(&(cntd_state.lead)).unwrap());
                let coeff = phase * _coeff(&orig_state, &cntd_state);

                if j_element.contains_key(&j) {
                    *j_element.get_mut(&j).unwrap() += J * coeff;
                } else {
                    j_element.insert(j, J * coeff);
                }
            },
            None => ()
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
        let (upup, downdown) = _repeated_spins(orig_state.lead, s1, s2);
        let mut new_dec: u32;
        let mut gamma = Complex::new(0., 0.);
        match (upup, downdown) {
            (true, false) => {
                new_dec = orig_state.lead - s1 - s2;
                gamma += _gamma(nx, ny, s1, s2).conj();
            }
            (false, true) => {
                new_dec = orig_state.lead + s1 + s2;
                gamma += _gamma(nx, ny, s1, s2);
            }
            _ => continue
        }
        match _find_leading_state(new_dec, &hashtable) {
            Some((cntd_state, phase)) => {
                let j = *(dec_to_ind.get(&(cntd_state.lead)).unwrap());
                let coeff = phase * _coeff(&orig_state, &cntd_state);

                if j_element.contains_key(&j) {
                    *j_element.get_mut(&j).unwrap() += J * coeff * gamma;
                } else {
                    j_element.insert(j, J * coeff * gamma);
                }
            },
            None => ()
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
            let mut gamma = Complex::new(0., 0.);
            if orig_state.lead | s2 == orig_state.lead {
                new_dec = orig_state.lead - s2;
                gamma += _gamma(nx, ny, s1, s2).conj();
            } else {
                new_dec = orig_state.lead + s2;
                gamma -= _gamma(nx, ny, s1, s2);
            }

            match _find_leading_state(new_dec, &hashtable) {
                Some((cntd_state, phase)) => {
                    let j = *(dec_to_ind.get(&(cntd_state.lead)).unwrap());
                    let coeff = phase * _coeff(&orig_state, &cntd_state);

                    if j_element.contains_key(&j) {
                        *j_element.get_mut(&j).unwrap() += J * z_contrib * coeff * gamma;
                    } else {
                        j_element.insert(j, J * z_contrib * coeff * gamma);
                    }
                },
                None => ()
            }
        }
    }
    j_element
}

