use std::mem;
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign,
               Rem, RemAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign,
               Neg};
use std::cmp::Ordering;
use libc::size_t;
use num_complex::Complex;
use fnv::FnvHashMap;

use sitevector::SiteVector;
use blochfunc::{BlochFunc, BlochFuncSet};

pub const PI: f64 = 3.1415926535897932384626433832795028841971;
pub const POW2: [BinaryBasis; 63] = [
    BinaryBasis(1), BinaryBasis(2), BinaryBasis(4), BinaryBasis(8),
    BinaryBasis(16), BinaryBasis(32), BinaryBasis(64), BinaryBasis(128),
    BinaryBasis(256), BinaryBasis(512), BinaryBasis(1024), BinaryBasis(2048),
    BinaryBasis(4096), BinaryBasis(8192), BinaryBasis(16384),
    BinaryBasis(32768), BinaryBasis(65536), BinaryBasis(131072),
    BinaryBasis(262144), BinaryBasis(524288), BinaryBasis(1048576),
    BinaryBasis(2097152), BinaryBasis(4194304), BinaryBasis(8388608),
    BinaryBasis(16777216), BinaryBasis(33554432), BinaryBasis(67108864),
    BinaryBasis(134217728), BinaryBasis(268435456), BinaryBasis(536870912),
    BinaryBasis(1073741824), BinaryBasis(2147483648), BinaryBasis(4294967296),
    BinaryBasis(8589934592), BinaryBasis(17179869184), BinaryBasis(34359738368),
    BinaryBasis(68719476736), BinaryBasis(137438953472),
    BinaryBasis(274877906944), BinaryBasis(549755813888),
    BinaryBasis(1099511627776), BinaryBasis(2199023255552),
    BinaryBasis(4398046511104), BinaryBasis(8796093022208),
    BinaryBasis(17592186044416), BinaryBasis(35184372088832),
    BinaryBasis(70368744177664), BinaryBasis(140737488355328),
    BinaryBasis(281474976710656), BinaryBasis(562949953421312),
    BinaryBasis(1125899906842624), BinaryBasis(2251799813685248),
    BinaryBasis(4503599627370496), BinaryBasis(9007199254740992),
    BinaryBasis(18014398509481984), BinaryBasis(36028797018963968),
    BinaryBasis(72057594037927936), BinaryBasis(144115188075855872),
    BinaryBasis(288230376151711744), BinaryBasis(576460752303423488),
    BinaryBasis(1152921504606846976), BinaryBasis(2305843009213693952),
    BinaryBasis(4611686018427387904)
];

make_int_type!(BinaryBasis, u64);
make_int_type!(Dim, u32);
make_int_type!(I, i32);

impl Neg for I {
    type Output = Self;

    fn neg(self) -> Self {
        I(-self.0)
    }
}

impl Add<Dim> for I {
    type Output = Self;

    fn add(self, rhs: Dim) -> Self {
        I(self.0 + rhs.0 as i32)
    }
}

impl AddAssign<Dim> for I {
    fn add_assign(&mut self, rhs: Dim) {
        *self = I(self.0 + rhs.0 as i32)
    }
}

impl Mul<Dim> for I {
    type Output = Self;

    fn mul(self, rhs: Dim) -> Self {
        I(self.0 * rhs.0 as i32)
    }
}

impl Rem<Dim> for I {
    type Output = Self;

    fn rem(self, rhs: Dim) -> Self {
        I(self.0 % rhs.0 as i32)
    }
}

impl RemAssign<Dim> for I {
    fn rem_assign(&mut self, rhs: Dim) {
        *self = I(self.0 % rhs.0 as i32)
    }
}

impl Div<Dim> for I {
    type Output = Self;

    fn div(self, rhs: Dim) -> Self {
        I(self.0 / rhs.0 as i32)
    }
}

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
    pub ptr: *mut T,
    pub len: size_t
}

impl<T> Vector<T> {
    fn new(ptr: *mut T, len: size_t) -> Vector<T> {
        Vector { ptr, len }
    }
}

#[repr(C)]
pub struct CoordMatrix<T> {
    pub data: Vector<T>,
    pub col: Vector<u32>,
    pub row: Vector<u32>,
    pub ncols: u32,
    pub nrows: u32
}

impl<T> CoordMatrix<T> {
    pub fn new(mut data: Vec<T>, mut col: Vec<u32>, mut row: Vec<u32>,
               ncols: u32, nrows: u32) -> CoordMatrix<T> {
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
        CoordMatrix { data, col, row, ncols, nrows }
    }
}

pub fn translate_x(dec: BinaryBasis, nx: Dim, ny: Dim) -> BinaryBasis {
    let n = (0..ny.raw_int()).map(|x| x * nx.raw_int()).collect::<Vec<u32>>();
    let s = n.iter()
             .map(|&x| dec % POW2[(x + nx.raw_int()) as usize] / POW2[x as usize])
             .map(|x| (x * BinaryBasis(2)) % POW2[nx.raw_int() as usize] +
                       x / POW2[nx.raw_int() as usize - 1]);

    n.iter().map(|&x| POW2[x as usize])
     .zip(s)
     .map(|(a, b)| a * b)  // basically a dot product here
     .fold(BinaryBasis(0), |acc, x| x + acc) // sum over vector
}

pub fn translate_y(dec: BinaryBasis, nx: Dim, ny: Dim) -> BinaryBasis {
    let xdim = POW2[nx.raw_int() as usize];
    let pred_totdim = POW2[nx.raw_int() as usize * (ny.raw_int() - 1) as usize];
    let tail = dec % xdim;
    dec / xdim + tail * pred_totdim
}

pub fn exchange_spin_flips(dec: BinaryBasis, s1: BinaryBasis, s2: BinaryBasis) -> (bool, bool) {
    let updown = (dec | s1 == dec) && (dec | s2 != dec);
    let downup = (dec | s1 != dec) && (dec | s2 == dec);
    (updown, downup)
}

pub fn repeated_spins(dec: BinaryBasis, s1: BinaryBasis, s2: BinaryBasis) -> (bool, bool) {
    let upup = (dec | s1 == dec) && (dec | s2 == dec);
    let downdown = (dec | s1 != dec) && (dec | s2 != dec);
    (upup, downdown)
}

pub fn generate_bonds(nx: Dim, ny: Dim) -> Vec<Vec<Vec<SiteVector>>> {
    let n = nx * ny;
    let mut vec = SiteVector::new((I(0), I(0)), nx, ny);
    let mut bonds_by_range = vec![Vec::new(); 3];
    for _ in 0..n.raw_int() {
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

pub fn gamma(nx: Dim, ny: Dim, s1: BinaryBasis, s2: BinaryBasis) -> Complex<f64> {
    let m = (s1.raw_int() as f64).log2().round() as i32;
    let n = (s2.raw_int() as f64).log2().round() as i32;
    let vec1 = SiteVector::from_index(I(m), nx, ny);
    let vec2 = SiteVector::from_index(I(n), nx, ny);
    let ang = vec1.angle_with(&vec2);

    Complex::from_polar(&1.0, &ang)
}

/// Generate all possible pairs of interacting sites on the lattice according to
/// the stride l
pub fn interacting_sites(nx: Dim, ny: Dim, l: I) -> (Vec<BinaryBasis>, Vec<BinaryBasis>) {
    let mut site1 = Vec::new();
    let mut site2 = Vec::new();
    let bonds_by_range = generate_bonds(nx, ny);
    let bonds = &bonds_by_range[l.raw_int() as usize - 1];
    for bond in bonds.iter() {
        site1.push(bond[0].lattice_index());
        site2.push(bond[1].lattice_index());
    }

    let f = |s: Vec<I>| s.into_iter()
                           .map(|s| POW2[s.raw_int() as usize])
                           .collect::<Vec<BinaryBasis>>();

    (f(site1), f(site2))
}

pub fn triangular_vert_sites(nx: Dim, ny: Dim) -> (Vec<BinaryBasis>, Vec<BinaryBasis>, Vec<BinaryBasis>) {
    let mut site1 = Vec::new();
    let mut site2 = Vec::new();
    let mut site3 = Vec::new();
    let mut vec = SiteVector::new((I(0), I(0)), nx, ny);
    let i = I(1);

    for _ in 0..ny.raw_int() {
        for _ in 0..nx.raw_int() {
            // For ijk in clockwise direction in upright triangle
            let s1 = vec.lattice_index();
            let s2 = vec.xhop(i).lattice_index();
            let s3 = vec.xhop(i).yhop(i).lattice_index();
            site1.push(s1);
            site2.push(s2);
            site3.push(s3);

            // For ijk in clockwise direction in inverted triangle
            let s4 = vec.lattice_index();
            let s5 = vec.xhop(i).lattice_index();
            let s6 = vec.xhop(i).yhop(-i).lattice_index();
            site1.push(s4);
            site2.push(s5);
            site3.push(s6);

            vec = vec.xhop(i);
        }
        vec = vec.yhop(i);
    }

    let f = |s: Vec<I>| s.into_iter()
                         .map(|s| POW2[s.raw_int() as usize])
                         .collect::<Vec<BinaryBasis>>();

    (f(site1), f(site2), f(site3))
}

/// Generate all permutations of the combination of any two sites on the lattice
/// where l = |i - j| for sites i and j
pub fn all_sites(nx: Dim, ny: Dim, l: I) -> (Vec<BinaryBasis>, Vec<BinaryBasis>) {
    let mut vec = SiteVector::new((I(0), I(0)), nx, ny);
    let xstride = l % nx;
    let ystride = l / nx;
    let mut site1 = Vec::new();
    let mut site2 = Vec::new();
    for _ in 0..ny.raw_int() {
        for _ in 0..nx.raw_int() {
            let s1 = vec.lattice_index();
            let s2 = vec.xhop(xstride).yhop(ystride).lattice_index();
            site1.push(s1);
            site2.push(s2);
            vec = vec.xhop(I(1));
        }
        vec = vec.yhop(I(1));
    }

    let f = |s: Vec<I>| s.into_iter()
                           .map(|s| POW2[s.raw_int() as usize])
                           .collect::<Vec<BinaryBasis>>();

    (f(site1), f(site2))
}

pub fn find_leading_state<'a>(dec: BinaryBasis,
                              hashtable: &'a FnvHashMap<&BinaryBasis, &BlochFunc>
                              ) -> Option<(&'a BlochFunc, Complex<f64>)> {

    match hashtable.get(&dec) {
        None => None,
        Some(&cntd_state) => match cntd_state.decs.get(&dec) {
            None     => None,
            Some(&p) => {
                let mut phase = p.conj();
                phase /= phase.norm();
                Some((cntd_state, phase))
            },
        }
    }
}

pub fn gen_ind_dec_conv_dicts<'a>(bfuncs: &'a BlochFuncSet)
    -> (FnvHashMap<u32, &'a BlochFunc>, FnvHashMap<BinaryBasis, u32>) {
    let dec = bfuncs.iter()
        .map(|x| x.lead)
        .collect::<Vec<_>>();
    let nstates = dec.len();
    let inds = (0..nstates as u32).collect::<Vec<u32>>();

    // build the hashtables
    let dec_to_ind = dec.into_iter()
        .zip(inds.clone())
        .collect::<FnvHashMap<BinaryBasis, u32>>();
    let ind_to_dec = inds.into_iter()
        .zip(bfuncs.iter())
        .collect::<FnvHashMap<u32, &BlochFunc>>();

    (ind_to_dec, dec_to_ind)
}

pub fn coeff(orig_state: &BlochFunc, cntd_state: &BlochFunc) -> f64 {
    cntd_state.norm / orig_state.norm
}
