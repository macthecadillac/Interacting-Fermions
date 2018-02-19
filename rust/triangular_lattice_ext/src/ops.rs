use num_complex::Complex;
use fnv::FnvHashMap;
use blochfunc::{BlochFunc, BlochFuncSet};
use common::*;

pub fn ss_z_elements(sites: &(Vec<u64>, Vec<u64>), orig_state: &BlochFunc) -> f64 {
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
#[allow(unused)]
pub fn ss_pm_elements(nx: u32, ny: u32,
                      sites: &(Vec<u64>, Vec<u64>),
                      orig_state: &BlochFunc,
                      dec_to_ind: &FnvHashMap<u64, u32>,
                      hashtable: &FnvHashMap<&u64, &BlochFunc>
                      ) -> FnvHashMap<u32, Complex<f64>> {
    let J = Complex::new(0.5, 0.);
    let mut j_element = FnvHashMap::default();
    let (ref site1, ref site2) = *sites;
    for (&s1, &s2) in site1.iter().zip(site2.iter()) {
        let (updown, downup) = exchange_spin_flips(orig_state.lead, s1, s2);
        let mut new_dec: u64;
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
pub fn ss_ppmm_elements(nx: u32, ny: u32,
                        sites: &(Vec<u64>, Vec<u64>),
                        orig_state: &BlochFunc,
                        dec_to_ind: &FnvHashMap<u64, u32>,
                        hashtable: &FnvHashMap<&u64, &BlochFunc>
                        ) -> FnvHashMap<u32, Complex<f64>> {
    let J = Complex::new(1., 0.);
    let mut j_element = FnvHashMap::default();
    let (ref site1, ref site2) = *sites;
    for (&s1, &s2) in site1.iter().zip(site2.iter()) {
        let (upup, downdown) = repeated_spins(orig_state.lead, s1, s2);
        let mut new_dec: u64;
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
pub fn ss_pmz_elements(nx: u32, ny: u32,
                       sites: &(Vec<u64>, Vec<u64>),
                       orig_state: &BlochFunc,
                       dec_to_ind: &FnvHashMap<u64, u32>,
                       hashtable: &FnvHashMap<&u64, &BlochFunc>,
                       ) -> FnvHashMap<u32, Complex<f64>> {
    let J = Complex::new(0., 1.);  // the entire operator was multiplied by i
    let mut j_element = FnvHashMap::default();
    let (ref site1, ref site2) = *sites;
    for (&s_1, &s_2) in site1.iter().zip(site2.iter()) {
        for &(s1, s2) in [(s_1, s_2), (s_2, s_1)].iter() {
            let z_contrib =
                if orig_state.lead | s1 == orig_state.lead { 0.5 } else { -0.5 };

            let mut new_dec: u64;
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

pub fn ss_z(sites: &(Vec<u64>, Vec<u64>), bfuncs: &BlochFuncSet)
    -> CoordMatrix<CComplex<f64>> {
    let dims = bfuncs.nonzero;
    let (ind_to_dec, _) = gen_ind_dec_conv_dicts(&bfuncs);

    let mut data: Vec<CComplex<f64>> = Vec::with_capacity(dims as usize);
    let cols = (0..dims as u32).collect::<Vec<u32>>();
    let rows = (0..dims as u32).collect::<Vec<u32>>();
    for i in 0..dims as u32 {
        let orig_state = ind_to_dec.get(&i).unwrap();
        let i_element = ss_z_elements(&sites, &orig_state);
        let re = i_element;
        let im = 0.;
        data.push(CComplex { re, im });
    }
    CoordMatrix::new(data, cols, rows, dims, dims)
}

fn off_diag_ops(element_f: fn(nx: u32, ny: u32,
                              sites: &(Vec<u64>, Vec<u64>),
                              orig_state: &BlochFunc,
                              dec_to_ind: &FnvHashMap<u64, u32>,
                              hashtable: &FnvHashMap<&u64, &BlochFunc>
                              ) -> FnvHashMap<u32, Complex<f64>>,
                sites: &(Vec<u64>, Vec<u64>), bfuncs: &BlochFuncSet)
    -> CoordMatrix<CComplex<f64>> {
    let dims = bfuncs.nonzero;
    let hashtable = BlochFuncSet::build_dict(&bfuncs);
    let (ind_to_dec, dec_to_ind) = gen_ind_dec_conv_dicts(&bfuncs);

    let alloc_size = dims * (1 + 8 * bfuncs.nx * bfuncs.ny);
    let mut data: Vec<CComplex<f64>> = Vec::with_capacity(alloc_size as usize);
    let mut cols: Vec<u32> = Vec::with_capacity(alloc_size as usize);
    let mut rows: Vec<u32> = Vec::with_capacity(alloc_size as usize);
    for i in 0..dims as u32 {
        let orig_state = ind_to_dec.get(&i).unwrap();
        let ij_elements = element_f(bfuncs.nx, bfuncs.ny, sites, &orig_state,
                                    &dec_to_ind, &hashtable);
        for (j, entry) in ij_elements.into_iter() {
            rows.push(i);
            cols.push(j);
            data.push(CComplex::from_num_complex(entry));
        }
    }
    CoordMatrix::new(data, cols, rows, dims, dims)
}

pub fn ss_pm(sites: &(Vec<u64>, Vec<u64>), bfuncs: &BlochFuncSet)
    -> CoordMatrix<CComplex<f64>> {
    off_diag_ops(ss_pm_elements, &sites, &bfuncs)
}

pub fn ss_ppmm(sites: &(Vec<u64>, Vec<u64>), bfuncs: &BlochFuncSet)
    -> CoordMatrix<CComplex<f64>> {
    off_diag_ops(ss_ppmm_elements, &sites, &bfuncs)
}

pub fn ss_pmz(sites: &(Vec<u64>, Vec<u64>), bfuncs: &BlochFuncSet)
    -> CoordMatrix<CComplex<f64>> {
    off_diag_ops(ss_pmz_elements, &sites, &bfuncs)
}
