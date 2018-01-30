use std::mem;
use libc::size_t;
use num_complex::Complex;
use fnv::FnvHashMap;

use super::PI;
use sitevector::SiteVector;
use blochfunc::{BlochFunc, BlochFuncSet};

// c compatible complex type for export to numpy at the end
#[repr(C)]
pub struct CComplex<T> {
    pub re: T,
    pub im: T
}

impl<T> CComplex<T> {
    pub fn from_num_complex(c: Complex<T>) -> CComplex<T> {
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
    pub fn new(mut data: Vec<T>, mut col: Vec<u32>, mut row: Vec<u32>) -> CoordMatrix<T> {
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

pub struct InteractingSites {
    pub first: (Vec<u32>, Vec<u32>),
    pub second: (Vec<u32>, Vec<u32>),
    pub third: (Vec<u32>, Vec<u32>)
}

pub fn translate_x(dec: u32, nx: u32, ny: u32) -> u32 {
    let n = (0..ny).map(|x| x * nx).collect::<Vec<u32>>();
    let s = n.iter()
        .map(|&x| dec % 2_u32.pow(x + nx) / 2_u32.pow(x))
        .map(|x| (x * 2_u32) % 2_u32.pow(nx) + x / 2_u32.pow(nx - 1));

    n.iter().map(|&x| 2_u32.pow(x))
        .zip(s)
        .map(|(a, b)| a * b)  // basically a dot product here
        .sum()
}

pub fn translate_y(dec: u32, nx: u32, ny: u32) -> u32 {
    let xdim = 2_u32.pow(nx);
    let pred_totdim = 2_u32.pow(nx * (ny - 1));
    let tail = dec % xdim;
    dec / xdim + tail * pred_totdim
}

pub fn _exchange_spin_flips(dec: u32, s1: u32, s2: u32) -> (bool, bool) {
    let updown = (dec | s1 == dec) && (dec | s2 != dec);
    let downup = (dec | s1 != dec) && (dec | s2 == dec);
    (updown, downup)
}

pub fn _repeated_spins(dec: u32, s1: u32, s2: u32) -> (bool, bool) {
    let upup = (dec | s1 == dec) && (dec | s2 == dec);
    let downdown = (dec | s1 != dec) && (dec | s2 != dec);
    (upup, downdown)
}

pub fn _generate_bonds(nx: u32, ny: u32) -> Vec<Vec<Vec<SiteVector>>> {
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

pub fn _phase_arr(nx: u32, ny: u32, kx: u32, ky: u32) -> Vec<Complex<f64>> {
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

pub fn _norm_coeff(bfunc: &BlochFunc, nx: u32, ny: u32, kx: u32, ky: u32) -> f64 {
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

pub fn _gamma(nx: u32, ny: u32, s1: u32, s2: u32) -> Complex<f64> {
    let m = (s1 as f64).log2().round() as u32;
    let n = (s2 as f64).log2().round() as u32;
    let vec1 = SiteVector::from_index(m, nx, ny);
    let vec2 = SiteVector::from_index(n, nx, ny );
    let ang = vec1.angle_with(&vec2);

    Complex::from_polar(&1.0, &ang)
}

pub fn _interacting_sites(nx: u32, ny: u32, l: u32) -> (Vec<u32>, Vec<u32>) {
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

pub fn zero_momentum_states<'a>(nx: u32, ny: u32) -> BlochFuncSet {
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

pub fn _bloch_states(nx: u32, ny: u32, kx: u32, ky: u32) -> BlochFuncSet {
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

pub fn _find_leading_state<'a>(dec: u32,
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

pub fn _gen_ind_dec_conv_dicts<'a>(bfuncs: &'a BlochFuncSet)
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

pub fn _coeff(orig_state: &BlochFunc, cntd_state: &BlochFunc) -> f64 {
    cntd_state.norm.unwrap() / orig_state.norm.unwrap()
}

