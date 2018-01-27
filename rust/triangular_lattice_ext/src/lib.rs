use std::mem;

extern crate num_complex;
use num_complex::Complex;

extern crate fnv;
use fnv::FnvHashMap;

extern crate libc;
use libc::size_t;

mod sitevector;
use sitevector::*;

mod blochfunc;
use blochfunc::*;

const PI: f64 = 3.14159265358979323846;

// c compatible complex type for export to numpy at the end
#[repr(C)]
pub struct CComplex<T> {
    re: T,
    im: T
}

impl<T> CComplex<T> {
    fn from_num_complex(c: Complex<T>) -> CComplex<T> {
        let re = c.re;
        let im = c.im;
        CComplex { re, im }
    }
}

#[repr(C)]
pub struct Vector<T> {
    ptr: *const T,
    len: size_t
}

impl<T> Vector<T> {
    fn new(ptr: *const T, len: size_t) -> Vector<T> {
        Vector { ptr, len }
    }
}

#[repr(C)]
pub struct CoordMatrix<T> {
    data: Vector<T>,
    col: Vector<u32>,
    row: Vector<u32>
}

impl<T> CoordMatrix<T> {
    fn new(mut data: Vec<T>, mut col: Vec<u32>, mut row: Vec<u32>) -> CoordMatrix<T> {
        let data_ptr = data.as_mut_ptr();
        let data_len = data.len() as size_t;

        let col_ptr = col.as_mut_ptr();
        let col_len = col.len() as size_t;

        let row_ptr = row.as_mut_ptr();
        let row_len = row.len() as size_t;

        mem::forget(data);
        mem::forget(col);
        mem::forget(row);
        let data = Vector::new(data_ptr, data_len);
        let col = Vector::new(col_ptr, col_len);
        let row = Vector::new(row_ptr, row_len);
        CoordMatrix { data, col, row }
    }
}

struct InteractingSites {
    first: (Vec<u32>, Vec<u32>),
    second: (Vec<u32>, Vec<u32>),
    third: (Vec<u32>, Vec<u32>)
}

fn translate_x(dec: u32, nx: u32, ny: u32) -> u32 {
    let n = (0..ny).map(|x| x * nx).collect::<Vec<u32>>();
    let s = n.iter()
        .map(|&x| dec % 2_u32.pow(x + nx) / 2_u32.pow(x))
        .map(|x| (x * 2_u32) % 2_u32.pow(nx) + x / 2_u32.pow(nx - 1));

    n.iter().map(|&x| 2_u32.pow(x))
        .zip(s)
        .map(|(a, b)| a * b)  // basically a dot product here
        .sum()
}

fn translate_y(dec: u32, nx: u32, ny: u32) -> u32 {
    let xdim = 2_u32.pow(nx);
    let pred_totdim = 2_u32.pow(nx * (ny - 1));
    let tail = dec % xdim;
    dec / xdim + tail * pred_totdim
}

fn _exchange_spin_flips(dec: u32, s1: u32, s2: u32) -> (bool, bool) {
    let updown = (dec | s1 == dec) && (dec | s2 != dec);
    let downup = (dec | s1 != dec) && (dec | s2 == dec);
    (updown, downup)
}

fn _repeated_spins(dec: u32, s1: u32, s2: u32) -> (bool, bool) {
    let upup = (dec | s1 == dec) && (dec | s2 == dec);
    let downdown = (dec | s1 != dec) && (dec | s2 != dec);
    (upup, downdown)
}

fn _generate_bonds(nx: u32, ny: u32) -> Vec<Vec<Vec<SiteVector>>> {
    let n = nx * ny;
    let mut vec = SiteVector::new((0, 0), nx, ny);
    let mut bonds_by_range = vec![Vec::new(); 3];
    for _ in 0..n {
        let nearest_neighbor = vec.nearest_neighboring_sites(false);
        let second_neighbor = vec.second_neighboring_sites(false);
        let third_neighbor = vec.third_neighboring_sites(false);
        let neighbors = vec![nearest_neighbor, second_neighbor, third_neighbor];
        for (leap, bonds) in bonds_by_range.iter_mut().enumerate() {
            for n in neighbors[leap].iter() {
                let mut bond = vec![vec.clone(), n.clone()];
                bond.sort();
                bonds.push(bond);
            }
        }
        vec = vec.next_site();
    }
    bonds_by_range
}

fn _phase_arr(nx: u32, ny: u32, kx: u32, ky: u32) -> Vec<Complex<f64>> {
    let xphase = (0..nx)
        .map(|m| 2. * PI * (kx as f64) * (m as f64) / (nx as f64))
        .map(|x| Complex::new(0., x))
        .map(|x| x.exp())
        .collect::<Vec<_>>();

    let yphase = (0..ny)
        .map(|n| 2. * PI * (ky as f64) * (n as f64) / (ny as f64))
        .map(|x| Complex::new(0., x))
        .map(|x| x.exp());

    let mut phase_arr: Vec<Complex<f64>> = Vec::new();
    for iy in yphase {
        for ix in &xphase {
            phase_arr.push(ix * iy);
        }
    }

    phase_arr
}

fn _norm_coeff(bfunc: &BlochFunc, nx: u32, ny: u32, kx: u32, ky: u32) -> f64 {
    let n = (nx * ny) as usize;
    let phase_arr = _phase_arr(nx, ny, kx, ky);
    let mut phases: Vec<Complex<f64>> = Vec::with_capacity(n);
    for locs in bfunc.decs.values() {
        let mut tot_phase = Complex::new(0., 0.);
        for &loc in locs.iter() {
            tot_phase += phase_arr[loc as usize];
        }
        phases.push(tot_phase);
    }

    phases.into_iter()
        .map(|x| x.norm().powf(2.))
        .sum::<f64>()
        .sqrt()
}

fn _gamma(nx: u32, ny: u32, s1: u32, s2: u32) -> Complex<f64> {
    let m = (s1 as f64).log2().round() as u32;
    let n = (s2 as f64).log2().round() as u32;
    let vec1 = SiteVector::from_index(m, nx, ny);
    let vec2 = SiteVector::from_index(n, nx, ny );
    let ang = vec1.angle_with(&vec2);

    Complex::from_polar(&1.0, &ang)
}

fn _interacting_sites(nx: u32, ny: u32, l: u32) -> (Vec<u32>, Vec<u32>) {
    let mut site1 = Vec::new();
    let mut site2 = Vec::new();
    let bonds_by_range = _generate_bonds(nx, ny);
    let bonds = &bonds_by_range[l as usize - 1];
    for bond in bonds.iter() {
        site1.push(bond[0].lattice_index());
        site2.push(bond[1].lattice_index());
    }

    (
    site1.into_iter().map(|x| 2_u32.pow(x)).collect::<Vec<_>>(),
    site2.into_iter().map(|x| 2_u32.pow(x)).collect::<Vec<_>>(),
    )
}

fn zero_momentum_states<'a>(nx: u32, ny: u32) -> BlochFuncSet {
    let n = nx * ny;
    let mut sieve = vec![true; 2_usize.pow(n)];
    let mut bfuncs: Vec<BlochFunc> = Vec::new();

    for dec in 0..2_usize.pow(n) {
        if sieve[dec]
        {   // if the corresponding entry of dec in "sieve" is not false,
            // we find all translations of dec and put them in a BlochFunc
            // then mark all corresponding entries in "sieve" as false.

            // "decs" is a hashtable that holds vectors whose entries
            // correspond to Bloch function constituent configurations which
            // are mapped to single decimals that represent the leading states.
            let mut decs: FnvHashMap<u32, Vec<u32>> = FnvHashMap::default();
            // "new_dec" represents the configuration we are currently iterating
            // over.
            let mut new_dec = dec as u32;
            for n in 0..ny {
                for m in 0..nx {
                    // "loc" represents the number of translations from 0
                    // (the leading state) to the current configuration.
                    let loc = n * nx + m;
                    // set sieve entry to false since we have iterated over it
                    sieve[new_dec as usize] = false;
                    // if "decs" already contains our key (the leading state)
                    // then we simply add the current "loc" to the vector
                    // corresponding to the key.
                    if decs.contains_key(&new_dec) {
                        match decs.get_mut(&new_dec) {
                            Some(v) => v.push(loc),
                            None => ()  // cannot put the insert here because
                                        // of double mutable borrowing
                        }
                    } else {
                        // since the hashtable does not contain our key, we add
                        // the key then add "loc" to it
                        let mut v = vec![loc];
                        decs.insert(new_dec, v);
                    }
                    new_dec = translate_x(new_dec, nx, ny);
                }
                new_dec = translate_y(new_dec, nx, ny);
            }

            let lead = dec as u32;
            let norm = None;
            let mut bfunc = BlochFunc { lead, decs, norm };
            bfuncs.push(bfunc)
        }
    }

    let mut table = BlochFuncSet::create(bfuncs);
    table.sort();
    table
}

fn _bloch_states(nx: u32, ny: u32, kx: u32, ky: u32) -> BlochFuncSet {
    let mut bfuncs = zero_momentum_states(nx, ny);
    let mut nonzero = 0;
    for bfunc in bfuncs.data.iter_mut() {
        let norm = _norm_coeff(bfunc, nx, ny, kx, ky);
        bfunc.norm = Some(norm);
        if norm > 1e-8 {
            nonzero += 1;
        }
    }
    bfuncs.nonzero = Some(nonzero);
    bfuncs
}

fn _find_leading_state<'a>(dec: u32,
                           hashtable: &'a FnvHashMap<&u32, &BlochFunc>,
                           phase_arr: &Vec<Complex<f64>>
                           ) -> Option<(&'a BlochFunc, Complex<f64>)> {
    let leading_state = match hashtable.get(&dec) {
        Some(&cntd_state) => 
            if cntd_state.norm < Some(1e-8) { None }
            else {
                let mut phase = Complex::new(0., 0.);
                if let Some(locs) = cntd_state.decs.get(&dec) {
                    unsafe {
                        for &loc in locs.iter() {
                            phase += phase_arr.get_unchecked(loc as usize);
                        }
                    }
                }
                phase = phase.conj();
                phase /= phase.norm();
                Some((cntd_state, phase))
            },
        None => panic!("Leading state not found!")
    };
    leading_state
}

fn _gen_ind_dec_conv_dicts<'a>(bfuncs: &'a BlochFuncSet)
    -> (FnvHashMap<u32, &'a BlochFunc>, FnvHashMap<u32, u32>) {
    let nonzero_states = bfuncs.iter()
        .filter(|s| s.norm.unwrap() > 1e-8)
        .collect::<Vec<_>>();
    let dec = nonzero_states.iter()
        .map(|x| x.lead)
        .collect::<Vec<_>>();
    let nstates = dec.len();
    let inds = (0..nstates as u32).collect::<Vec<u32>>();

    // build the hashtables
    let dec_to_ind = dec.into_iter()
        .zip(inds.clone())
        .collect::<FnvHashMap<u32, u32>>();
    let ind_to_dec = inds.into_iter()
        .zip(nonzero_states)
        .collect::<FnvHashMap<u32, &BlochFunc>>();

    (ind_to_dec, dec_to_ind)
}

fn _coeff(orig_state: &BlochFunc, cntd_state: &BlochFunc) -> f64 {
    cntd_state.norm.unwrap() / orig_state.norm.unwrap()
}

fn h_z_elements(sites: &(Vec<u32>, Vec<u32>), orig_state: &BlochFunc) -> f64 {
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
fn h_pm_elements(J: f64, sites: &(Vec<u32>, Vec<u32>),
                 orig_state: &BlochFunc,
                 dec_to_ind: &FnvHashMap<u32, u32>,
                 hashtable: &FnvHashMap<&u32, &BlochFunc>,
                 phase_arr: &Vec<Complex<f64>>
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
        match _find_leading_state(new_dec, &hashtable, &phase_arr) {
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
fn h_ppmm_elements(nx: u32, ny: u32, J: f64,
                   sites: &(Vec<u32>, Vec<u32>),
                   orig_state: &BlochFunc,
                   dec_to_ind: &FnvHashMap<u32, u32>,
                   hashtable: &FnvHashMap<&u32, &BlochFunc>,
                   phase_arr: &Vec<Complex<f64>>
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
        match _find_leading_state(new_dec, &hashtable, &phase_arr) {
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
fn h_pmz_elements(nx: u32, ny: u32, J: f64,
                  sites: &(Vec<u32>, Vec<u32>),
                  orig_state: &BlochFunc,
                  dec_to_ind: &FnvHashMap<u32, u32>,
                  hashtable: &FnvHashMap<&u32, &BlochFunc>,
                  phase_arr: &Vec<Complex<f64>>
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

            match _find_leading_state(new_dec, &hashtable, &phase_arr) {
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

fn h_elements(i: u32, nx: u32, ny: u32, j_z: f64, j_pm: f64,
              j_ppmm: f64, j_pmz: f64, j2: f64, j3: f64,
              sites: &InteractingSites,
              ind_to_dec: &FnvHashMap<u32, &BlochFunc>,
              dec_to_ind: &FnvHashMap<u32, u32>,
              hashtable: &FnvHashMap<&u32, &BlochFunc>,
              phase_arr: &Vec<Complex<f64>>
              ) -> FnvHashMap<u32, Complex<f64>> {
    let orig_state = ind_to_dec.get(&i).unwrap();

    let hz_1_el = j_z * h_z_elements(&sites.first, &orig_state);
    let hz_2_el = j2 * j_z / j_pm * h_z_elements(&sites.second, &orig_state);
    let hz_3_el = j3 * j_z / j_pm * h_z_elements(&sites.third, &orig_state);

    let hpm_1_el = h_pm_elements(j_pm, &sites.first, &orig_state, &dec_to_ind,
                                 &hashtable, &phase_arr);

    let hpm_2_el = h_pm_elements(j2, &sites.second, &orig_state, &dec_to_ind,
                                 &hashtable, &phase_arr);

    let hpm_3_el = h_pm_elements(j3, &sites.third, &orig_state, &dec_to_ind,
                                 &hashtable, &phase_arr);

    let hppmm_el = h_ppmm_elements(nx, ny, j_ppmm, &sites.first, &orig_state,
                                   &dec_to_ind, &hashtable, &phase_arr);

    let hpmz_el = h_pmz_elements(nx, ny, j_pmz, &sites.first, &orig_state,
                                 &dec_to_ind, &hashtable, &phase_arr);

    let mut elements = FnvHashMap::default();
    elements.insert(i, Complex::new(hz_1_el + hz_2_el + hz_3_el, 0.));
    for op in [hpm_1_el, hpm_2_el, hpm_3_el, hppmm_el, hpmz_el].into_iter() {
        for (&j, el) in op.into_iter() {
            let j = j as u32;
            if elements.contains_key(&j) {
                *elements.get_mut(&j).unwrap() += el;
            } else {
                elements.insert(j, *el);
            }
        }
    }
    elements
}

#[no_mangle]
pub extern fn hamiltonian(nx: u32, ny: u32, kx: u32, ky: u32,
                          j_z: f64, j_pm: f64, j_ppmm: f64, j_pmz: f64,
                          j2: f64, j3: f64
                          ) -> CoordMatrix<CComplex<f64>> {
    let bfuncs = _bloch_states(nx, ny, kx, ky);
    let dims = bfuncs.nonzero.unwrap();
    let hashtable = BlochFuncSet::build_dict(&bfuncs);
    let phase_arr = _phase_arr(nx, ny, kx, ky);
    let (ind_to_dec, dec_to_ind) = _gen_ind_dec_conv_dicts(&bfuncs);

    let first = _interacting_sites(nx, ny, 1);
    let second = _interacting_sites(nx, ny, 2);
    let third = _interacting_sites(nx, ny, 3);
    let sites = InteractingSites { first, second, third };

    let alloc_size = dims * (1 + 8 * nx * ny);
    let mut data: Vec<CComplex<f64>> = Vec::with_capacity(alloc_size as usize);
    let mut cols: Vec<u32> = Vec::with_capacity(alloc_size as usize);
    let mut rows: Vec<u32> = Vec::with_capacity(alloc_size as usize);
    for i in 0..dims as u32 {
        let ij_elements = h_elements(i, nx, ny, j_z, j_pm, j_ppmm, j_pmz, j2, j3,
                                     &sites, &ind_to_dec, &dec_to_ind,
                                     &hashtable, &phase_arr);
        for (j, entry) in ij_elements.into_iter() {
            rows.push(i);
            cols.push(j);
            data.push(CComplex::from_num_complex(entry));
        }
    }
    CoordMatrix::new(data, cols, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_x() {
        let d1 = 10;
        let d2 = 5;
        assert_eq!(translate_x(d1, 4, 6), d2);
    }

    #[test]
    fn translate_y() {
        let d1 = 2;
        let d2 = 8192;
        assert_eq!(translate_y(d1, 4, 4), d2);
    }

    #[test]
    fn exchange_spin_flips() {
        assert_eq!(_exchange_spin_flips(10, 2, 8), (false, false));
        assert_eq!(_exchange_spin_flips(93, 1, 32), (true, false));
    }

    #[test]
    fn repeated_spins() {
        assert_eq!(_repeated_spins(93, 1, 32), (false, false));
        assert_eq!(_repeated_spins(93, 4, 64), (true, false));
        assert_eq!(_repeated_spins(93, 128, 256), (false, true));
    }

    #[test]
    fn generate_bonds() {
        let bonds = _generate_bonds(4, 6);
        assert_eq!(bonds[0].len(), 72);
        assert_eq!(bonds[1].len(), 72);
        assert_eq!(bonds[2].len(), 60);

        let bonds = _generate_bonds(6, 6);
        assert_eq!(bonds[0].len(), 108);
        assert_eq!(bonds[1].len(), 108);
        assert_eq!(bonds[2].len(), 108);
    }

    #[test]
    fn norm_coeff() {
        let nx = 4;
        let ny = 3;
        let kx = 0;
        let ky = 1;
        let bfuncs = _bloch_states(nx, ny, kx, ky);
        let bfunc = &bfuncs.data[34];
        let norm = _norm_coeff(bfunc, nx, ny, kx, ky);
        assert!((norm - 3.46410161514).abs() < 1e-8);

        let nx = 2;
        let ny = 4;
        let kx = 1;
        let ky = 0;
        let bfuncs = _bloch_states(nx, ny, kx, ky);
        let bfunc = &bfuncs.data[34];
        let norm = _norm_coeff(bfunc, nx, ny, kx, ky);
        assert!((norm - 2.82842712475).abs() < 1e-8);
    }

    #[test]
    fn gamma() {
        let nx = 4;
        let ny = 3;
        let s1 = 32;
        let s2 = 256;
        let gamma = _gamma(nx, ny, s1, s2);
        println!("{}", gamma);
        assert!((gamma - Complex::new(-0.5, 0.866025403784)).norm() < 1e-8);
    }

    #[test]
    fn zero_momentum_states() {
        let nx = 4;
        let ny = 4;
        let bfuncs = zero_momentum_states(nx, ny);
        assert_eq!(bfuncs.data.len(), 4156);

        let nx = 4;
        let ny = 3;
        let bfuncs = zero_momentum_states(nx, ny);
        assert_eq!(bfuncs.data.len(), 352);

        let nx = 3;
        let ny = 4;
        let bfuncs = zero_momentum_states(nx, ny);
        assert_eq!(bfuncs.data.len(), 352);
    }

    #[test]
    fn _bloch_states() {
        let nx = 4;
        let ny = 4;
        let kx = 1;
        let ky = 3;
        let bfuncs = _bloch_states(nx, ny, kx, ky);
        assert_eq!(bfuncs.nonzero, Some(4080));
    }
}
