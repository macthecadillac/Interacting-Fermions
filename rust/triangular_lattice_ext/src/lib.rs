extern crate num_complex;
use num_complex::Complex;

extern crate fnv;
use fnv::FnvHashMap;

extern crate libc;

mod ops;

mod common;
use common::*;

mod sitevector;
use sitevector::SiteVector;

mod blochfunc;

#[cfg(test)]
mod tests;

const PI: f64 = 3.14159265358979323846;

fn h_elements(i: u32, nx: u32, ny: u32, j_z: f64, j_pm: f64,
              j_ppmm: f64, j_pmz: f64, j2: f64, j3: f64,
              sites: &InteractingSites,
              ind_to_dec: &FnvHashMap<u32, &blochfunc::BlochFunc>,
              dec_to_ind: &FnvHashMap<u32, u32>,
              hashtable: &FnvHashMap<&u32, &blochfunc::BlochFunc>,
              ) -> FnvHashMap<u32, Complex<f64>> {
    let orig_state = ind_to_dec.get(&i).unwrap();

    let hz_1_el = j_z * ops::ss_z_elements(&sites.first, &orig_state);
    let hz_2_el = j2 * j_z / j_pm * ops::ss_z_elements(&sites.second, &orig_state);
    let hz_3_el = j3 * j_z / j_pm * ops::ss_z_elements(&sites.third, &orig_state);
    let hpm_1_el = ops::ss_pm_elements(j_pm, &sites.first, &orig_state,
                                       &dec_to_ind, &hashtable);
    let hpm_2_el = ops::ss_pm_elements(j2, &sites.second, &orig_state,
                                       &dec_to_ind, &hashtable);
    let hpm_3_el = ops::ss_pm_elements(j3, &sites.third, &orig_state,
                                       &dec_to_ind, &hashtable);
    let hppmm_el = ops::ss_ppmm_elements(nx, ny, j_ppmm, &sites.first,
                                         &orig_state, &dec_to_ind, &hashtable);
    let hpmz_el = ops::ss_pmz_elements(nx, ny, j_pmz, &sites.first, &orig_state,
                                       &dec_to_ind, &hashtable);

    let mut elements = FnvHashMap::default();
    elements.insert(i, Complex::new(hz_1_el + hz_2_el + hz_3_el, 0.));
    for op in [hpm_1_el, hpm_2_el, hpm_3_el, hppmm_el, hpmz_el].into_iter() {
        for (&j, el) in op.into_iter() {
            let j = j as u32;
            let e = match elements.get(&j) {
                Some(&e) => e + el,
                None     => *el
            };
            elements.insert(j, e);
        }
    }
    elements
}

#[no_mangle]
pub extern fn hamiltonian(nx: u32, ny: u32, kx: u32, ky: u32,
                          j_z: f64, j_pm: f64, j_ppmm: f64, j_pmz: f64,
                          j2: f64, j3: f64
                          ) -> CoordMatrix<CComplex<f64>> {
    let bfuncs = bloch_states(nx, ny, kx, ky);
    let dims = bfuncs.nonzero;
    let hashtable = blochfunc::BlochFuncSet::build_dict(&bfuncs);
    let (ind_to_dec, dec_to_ind) = gen_ind_dec_conv_dicts(&bfuncs);

    let first = interacting_sites(nx, ny, 1);
    let second = interacting_sites(nx, ny, 2);
    let third = interacting_sites(nx, ny, 3);
    let sites = InteractingSites { first, second, third };

    let alloc_size = dims * (1 + 8 * nx * ny);
    let mut data: Vec<CComplex<f64>> = Vec::with_capacity(alloc_size as usize);
    let mut cols: Vec<u32> = Vec::with_capacity(alloc_size as usize);
    let mut rows: Vec<u32> = Vec::with_capacity(alloc_size as usize);
    for i in 0..dims as u32 {
        let ij_elements = h_elements(i, nx, ny, j_z, j_pm, j_ppmm, j_pmz, j2, j3,
                                     &sites, &ind_to_dec, &dec_to_ind,
                                     &hashtable);
        for (j, entry) in ij_elements.into_iter() {
            rows.push(i);
            cols.push(j);
            data.push(CComplex::from_num_complex(entry));
        }
    }
    CoordMatrix::new(data, cols, rows)
}

fn _sites(nx: u32, ny: u32, l: u32) -> (Vec<u32>, Vec<u32>) {
    let mut vec = SiteVector::new((0, 0), nx, ny );
    let xstride = (l % nx) as i32;
    let ystride = (l / nx) as i32;
    let mut site1 = Vec::new();
    let mut site2 = Vec::new();
    for _ in 0..ny {
        for _ in 0..nx {
            let s1 = vec.lattice_index();
            let s2 = vec.xhop(xstride).yhop(ystride).lattice_index();
            site1.push(s1);
            site2.push(s2);
            vec = vec.xhop(1);
        }
        vec = vec.yhop(1);
    }

    (
    site1.into_iter().map(|s| 2_u32.pow(s)).collect::<Vec<u32>>(),
    site2.into_iter().map(|s| 2_u32.pow(s)).collect::<Vec<u32>>()
    )
}

#[no_mangle]
pub extern fn ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    let bfuncs = bloch_states(nx, ny, kx, ky);
    let dims = bfuncs.nonzero;
    let (ind_to_dec, _) = gen_ind_dec_conv_dicts(&bfuncs);
    let sites = _sites(nx, ny, l);
    let mut data: Vec<CComplex<f64>> = Vec::with_capacity(dims as usize);
    let mut cols: Vec<u32> = Vec::with_capacity(dims as usize);
    let mut rows: Vec<u32> = Vec::with_capacity(dims as usize);
    for i in 0..dims as u32 {
        let orig_state = ind_to_dec.get(&i).unwrap();
        let ij_elements = ops::ss_z_elements(&sites, &orig_state);
        let re = ij_elements;
        let im = 0.;
        data.push(CComplex { re, im });
        cols.push(i);
        rows.push(i);
    }
    CoordMatrix::new(data, cols, rows)
}

#[no_mangle]
pub extern fn ss_pm(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    let bfuncs = bloch_states(nx, ny, kx, ky);
    let dims = bfuncs.nonzero;
    let hashtable = blochfunc::BlochFuncSet::build_dict(&bfuncs);
    let (ind_to_dec, dec_to_ind) = gen_ind_dec_conv_dicts(&bfuncs);
    let sites = _sites(nx, ny, l);

    let mut data: Vec<CComplex<f64>> = Vec::with_capacity(dims as usize);
    let mut cols: Vec<u32> = Vec::with_capacity(dims as usize);
    let mut rows: Vec<u32> = Vec::with_capacity(dims as usize);
    for i in 0..dims as u32 {
        let orig_state = ind_to_dec.get(&i).unwrap();
        let ij_elements = ops::ss_pm_elements(1.0, &sites, &orig_state,
                                              &dec_to_ind, &hashtable);
        for (j, entry) in ij_elements.into_iter() {
            rows.push(i);
            cols.push(j);
            data.push(CComplex::from_num_complex(entry));
        }
    }
    CoordMatrix::new(data, cols, rows)
}
