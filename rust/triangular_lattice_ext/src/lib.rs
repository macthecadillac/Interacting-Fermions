extern crate num_complex;
extern crate num_bigint;
extern crate num_traits;
extern crate fnv;
extern crate libc;

mod blochfunc;
mod consv;
mod common;
mod ops;
mod sitevector;

#[cfg(test)]
mod tests;

use common::{CoordMatrix, CComplex};

// The following functions wrap functions in child modules so they could be
// exported via the FFI without namespace collisions (the FFI follows C
// convention so namespace doesn't exist.)
#[no_mangle]
pub extern fn k_h_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_z(nx, ny, kx, ky, l)
}

#[no_mangle]
pub extern fn k_h_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_xy(nx, ny, kx, ky, l)
}

#[no_mangle]
pub extern fn k_h_ss_ppmm(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_ppmm(nx, ny, kx, ky, l)
}

#[no_mangle]
pub extern fn k_h_ss_pmz(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_pmz(nx, ny, kx, ky, l)
}

#[no_mangle]
pub extern fn k_h_ss_chi(nx: u32, ny: u32, kx: u32, ky: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::h_ss_chi(nx, ny, kx, ky)
}

#[no_mangle]
pub extern fn k_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::ss_z(nx, ny, kx, ky, l)
}

#[no_mangle]
pub extern fn k_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::k::ss_xy(nx, ny, kx, ky, l)
}

#[no_mangle]
pub extern fn ks_h_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::ks::h_ss_z(nx, ny, kx, ky, nup, l)
}

#[no_mangle]
pub extern fn ks_h_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::ks::h_ss_xy(nx, ny, kx, ky, nup, l)
}

#[no_mangle]
pub extern fn ks_h_ss_chi(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::ks::h_ss_chi(nx, ny, kx, ky, nup)
}

#[no_mangle]
pub extern fn ks_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::ks::ss_z(nx, ny, kx, ky, nup, l)
}

#[no_mangle]
pub extern fn ks_ss_xy(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
    -> CoordMatrix<CComplex<f64>> {
    consv::ks::ss_xy(nx, ny, kx, ky, nup, l)
}

// accepts a pointer from external callers so Rust can dispose of the objects
// passed to the caller
#[no_mangle]
pub unsafe extern fn request_free(mat: CoordMatrix<CComplex<f64>>) {
    Box::from_raw(mat.data.ptr);
    Box::from_raw(mat.col.ptr);
    Box::from_raw(mat.row.ptr);
}
