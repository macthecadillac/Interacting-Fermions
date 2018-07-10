/// This module contains the following sub-modules:
///     k
///     ks
///     ksl

/// This module contains functions that work under the assumption that lattice
/// momentum is conserved.
pub mod k {
    use fnv::FnvHashMap;
    use num_complex::Complex;

    use blochfunc::{BlochFunc, BlochFuncSet};
    use common::*;
    use ops;

    fn bloch_states<'a>(nx: Dim, ny: Dim, kx: K, ky: K) -> BlochFuncSet {
        let n = nx * ny;
        let mut sieve = vec![true; 2_usize.pow(n.raw_int())];
        let mut bfuncs: Vec<BlochFunc> = Vec::new();
        let phase = |i, j| {
            let r = 1.;
            let ang1 = 2. * PI * (i * kx.raw_int()) as f64 / nx.raw_int() as f64;
            let ang2 = 2. * PI * (j * ky.raw_int()) as f64 / ny.raw_int() as f64;
            Complex::from_polar(&r, &(ang1 + ang2))
        };

        for dec in 0..2_usize.pow(n.raw_int()) {
            if sieve[dec] {
                // if the corresponding entry of dec in "sieve" is not false,
                // we find all translations of dec and put them in a BlochFunc
                // then mark all corresponding entries in "sieve" as false.

                // "decs" is a hashtable that holds vectors whose entries
                // correspond to Bloch function constituent configurations which
                // are mapped to single decimals that represent the leading states.
                let mut decs: FnvHashMap<BinaryBasis, Complex<f64>> =
                    FnvHashMap::default();
                // "new_dec" represents the configuration we are currently iterating
                // over.
                let mut new_dec = BinaryBasis(dec as u64);
                for j in 0..ny.raw_int() {
                    for i in 0..nx.raw_int() {
                        sieve[new_dec.raw_int() as usize] = false;
                        let new_p = match decs.get(&new_dec) {
                            Some(&p) => p + phase(i, j),
                            None => phase(i, j)
                        };
                        decs.insert(new_dec, new_p);
                        new_dec = translate_x(new_dec, nx, ny);
                    }
                    new_dec = translate_y(new_dec, nx, ny);
                }

                let lead = BinaryBasis(dec as u64);
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

        let mut table = BlochFuncSet::create(nx, ny, bfuncs);
        table.sort();
        table
    }

    pub fn h_ss_z(nx: Dim, ny: Dim, kx: K, ky: K, l: I)
                  -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn h_ss_xy(nx: Dim, ny: Dim, kx: K, ky: K, l: I)
                   -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_xy(&sites, &bfuncs)
    }

    pub fn h_ss_ppmm(nx: Dim, ny: Dim, kx: K, ky: K, l: I)
                     -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_ppmm(&sites, &bfuncs)
    }

    pub fn h_ss_pmz(nx: Dim, ny: Dim, kx: K, ky: K, l: I)
                    -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_pmz(&sites, &bfuncs)
    }

    pub fn h_sss_chi(nx: Dim, ny: Dim, kx: K, ky: K) -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = triangular_vert_sites(nx, ny);
        ops::sss_chi(&sites, &bfuncs)
    }

    pub fn ss_z(nx: Dim, ny: Dim, kx: K, ky: K, l: I) -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = all_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn ss_xy(nx: Dim, ny: Dim, kx: K, ky: K, l: I)
                 -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = all_sites(nx, ny, l);
        ops::ss_xy(&sites, &bfuncs)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bloch_states_test() {
            let nx = Dim(4);
            let ny = Dim(4);
            let kx = K(1);
            let ky = K(3);
            let bfuncs = bloch_states(nx, ny, kx, ky);
            assert_eq!(bfuncs.nonzero, 4080);
        }
    }
}

/// This module contains functions that work under the assumption that lattice
/// momentum and total Sz are conserved.
pub mod ks {
    use fnv::FnvHashMap;
    use num_complex::Complex;

    use blochfunc::{BlochFunc, BlochFuncSet};
    use common::*;
    use ops;

    fn bloch_states<'a>(nx: Dim, ny: Dim, kx: K, ky: K, nup: u32) -> BlochFuncSet {
        let n = nx * ny;

        let sz_basis_states = sz_basis(n, nup);
        let mut szdec_to_ind: FnvHashMap<BinaryBasis, usize> = FnvHashMap::default();
        let mut ind_to_szdec: FnvHashMap<usize, BinaryBasis> = FnvHashMap::default();
        for (i, &bs) in sz_basis_states.iter().enumerate() {
            ind_to_szdec.insert(i, bs);
            szdec_to_ind.insert(bs, i);
        }

        let mut sieve = vec![true; sz_basis_states.len()];
        let mut bfuncs: Vec<BlochFunc> = Vec::new();
        let phase = |i, j| {
            let r = 1.;
            let ang1 = 2. * PI * (i * kx.raw_int()) as f64 / nx.raw_int() as f64;
            let ang2 = 2. * PI * (j * ky.raw_int()) as f64 / ny.raw_int() as f64;
            Complex::from_polar(&r, &(ang1 + ang2))
        };

        for ind in 0..sieve.len() {
            if sieve[ind] {
                // if the corresponding entry of dec in "sieve" is not false,
                // we find all translations of dec and put them in a BlochFunc
                // then mark all corresponding entries in "sieve" as false.

                // "decs" is a hashtable that holds vectors whose entries
                // correspond to Bloch function constituent configurations which
                // are mapped to single decimals that represent the leading states.
                let mut decs: FnvHashMap<BinaryBasis, Complex<f64>> =
                    FnvHashMap::default();
                // "new_dec" represents the configuration we are currently iterating
                // over.
                let dec = *ind_to_szdec.get(&ind).unwrap();
                let mut new_dec = dec;
                let mut new_ind = ind;
                for j in 0..ny.raw_int() {
                    for i in 0..nx.raw_int() {
                        sieve[new_ind as usize] = false;
                        let new_p = match decs.get(&new_dec) {
                            Some(&p) => p + phase(i, j),
                            None => phase(i, j)
                        };
                        decs.insert(new_dec, new_p);
                        new_dec = translate_x(new_dec, nx, ny);
                        new_ind = *szdec_to_ind.get(&new_dec).unwrap() as usize;
                    }
                    new_dec = translate_y(new_dec, nx, ny);
                }

                let lead = dec;
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

        let mut table = BlochFuncSet::create(nx, ny, bfuncs);
        table.sort();
        table
    }

    pub fn h_ss_z(nx: Dim, ny: Dim, kx: K, ky: K, nup: u32, l: I)
                  -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn h_ss_xy(nx: Dim, ny: Dim, kx: K, ky: K, nup: u32, l: I)
                   -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_xy(&sites, &bfuncs)
    }

    pub fn h_sss_chi(nx: Dim, ny: Dim, kx: K, ky: K, nup: u32)
                     -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = triangular_vert_sites(nx, ny);
        ops::sss_chi(&sites, &bfuncs)
    }

    pub fn ss_z(nx: Dim, ny: Dim, kx: K, ky: K, nup: u32, l: I)
                -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = all_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn ss_xy(nx: Dim, ny: Dim, kx: K, ky: K, nup: u32, l: I)
                 -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = all_sites(nx, ny, l);
        ops::ss_xy(&sites, &bfuncs)
    }
}
