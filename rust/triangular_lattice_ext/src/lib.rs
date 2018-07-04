extern crate fnv;
extern crate libc;
extern crate num_bigint;
extern crate num_complex;
extern crate num_traits;

#[macro_use]
mod buildtype;

mod blochfunc;
pub mod common;
pub mod consv;
mod ops;
mod sitevector;

use common::{CComplex, CoordMatrix, Dim, I, K};

// The following functions wrap functions in child modules so they could be
// exported via the FFI without namespace collisions (the FFI follows C
// convention so namespace doesn't exist.)
#[no_mangle]
pub extern "C" fn k_h_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
                           -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_z(Dim(nx), Dim(ny), K(kx), K(ky), I(l as i32))
}

#[no_mangle]
pub extern "C" fn k_h_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
                            -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_xy(Dim(nx), Dim(ny), K(kx), K(ky), I(l as i32))
}

#[no_mangle]
pub extern "C" fn k_h_ss_ppmm(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
                              -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_ppmm(Dim(nx), Dim(ny), K(kx), K(ky), I(l as i32))
}

#[no_mangle]
pub extern "C" fn k_h_ss_pmz(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
                             -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_pmz(Dim(nx), Dim(ny), K(kx), K(ky), I(l as i32))
}

#[no_mangle]
pub extern "C" fn k_h_sss_chi(nx: u32, ny: u32, kx: u32, ky: u32)
                              -> CoordMatrix<CComplex<f64>> {
    consv::k::h_sss_chi(Dim(nx), Dim(ny), K(kx), K(ky))
}

#[no_mangle]
pub extern "C" fn k_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
                         -> CoordMatrix<CComplex<f64>> {
    consv::k::ss_z(Dim(nx), Dim(ny), K(kx), K(ky), I(l as i32))
}

#[no_mangle]
pub extern "C" fn k_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
                          -> CoordMatrix<CComplex<f64>> {
    consv::k::ss_xy(Dim(nx), Dim(ny), K(kx), K(ky), I(l as i32))
}

#[no_mangle]
pub extern "C" fn ks_h_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
                            -> CoordMatrix<CComplex<f64>> {
    consv::ks::h_ss_z(Dim(nx), Dim(ny), K(kx), K(ky), nup, I(l as i32))
}

#[no_mangle]
pub extern "C" fn ks_h_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
                             -> CoordMatrix<CComplex<f64>> {
    consv::ks::h_ss_xy(Dim(nx), Dim(ny), K(kx), K(ky), nup, I(l as i32))
}

#[no_mangle]
pub extern "C" fn ks_h_sss_chi(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32)
                               -> CoordMatrix<CComplex<f64>> {
    consv::ks::h_sss_chi(Dim(nx), Dim(ny), K(kx), K(ky), nup)
}

#[no_mangle]
pub extern "C" fn ks_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
                          -> CoordMatrix<CComplex<f64>> {
    consv::ks::ss_z(Dim(nx), Dim(ny), K(kx), K(ky), nup, I(l as i32))
}

#[no_mangle]
pub extern "C" fn ks_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
                           -> CoordMatrix<CComplex<f64>> {
    consv::ks::ss_xy(Dim(nx), Dim(ny), K(kx), K(ky), nup, I(l as i32))
}

// accepts a pointer from external callers so Rust can dispose of the objects
// passed to the caller
#[no_mangle]
pub unsafe extern "C" fn request_free(mat: CoordMatrix<CComplex<f64>>) {
    Box::from_raw(mat.data.ptr);
    Box::from_raw(mat.col.ptr);
    Box::from_raw(mat.row.ptr);
}
