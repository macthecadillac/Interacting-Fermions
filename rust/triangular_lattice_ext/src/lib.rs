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

const PI: f64 = 3.1415926535897932384626433832795028841971;
const POW2: [u64; 63] = [1, 2, 4,
                         8, 16, 32,
                        64, 128, 256,
                       512, 1024, 2048,
                      4096, 8192, 16384,
                     32768, 65536, 131072,
                    262144, 524288, 1048576,
                   2097152, 4194304, 8388608,
                  16777216, 33554432, 67108864,
                 134217728, 268435456, 536870912,
                1073741824, 2147483648, 4294967296,
                8589934592, 17179869184, 34359738368,
               68719476736, 137438953472, 274877906944,
              549755813888, 1099511627776, 2199023255552,
             4398046511104, 8796093022208, 17592186044416,
            35184372088832, 70368744177664, 140737488355328,
           281474976710656, 562949953421312, 1125899906842624,
          2251799813685248, 4503599627370496, 9007199254740992,
         18014398509481984, 36028797018963968, 72057594037927936,
        144115188075855872, 288230376151711744, 576460752303423488,
       1152921504606846976, 2305843009213693952, 4611686018427387904];

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
