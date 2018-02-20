pub mod k {
    // in this specific case crystal momentum is conserved
    use fnv::FnvHashMap;
    use num_complex::Complex;

    use blochfunc::{BlochFunc, BlochFuncSet};
    use common::*;
    use ops;
    use super::super::PI;

    pub fn bloch_states<'a>(nx: u32, ny: u32, kx: u32, ky: u32) -> BlochFuncSet {
        let n = nx * ny;
        let mut sieve = vec![true; 2_usize.pow(n)];
        let mut bfuncs: Vec<BlochFunc> = Vec::new();
        let phase = |i, j| {
            let r = 1.;
            let ang1 = 2. * PI * (i * kx) as f64 / nx as f64;
            let ang2 = 2. * PI * (j * ky) as f64 / ny as f64;
            Complex::from_polar(&r, &(ang1 + ang2))
        };

        for dec in 0..2_usize.pow(n) {
            if sieve[dec]
            {   // if the corresponding entry of dec in "sieve" is not false,
                // we find all translations of dec and put them in a BlochFunc
                // then mark all corresponding entries in "sieve" as false.

                // "decs" is a hashtable that holds vectors whose entries
                // correspond to Bloch function constituent configurations which
                // are mapped to single decimals that represent the leading states.
                let mut decs: FnvHashMap<u64, Complex<f64>> = FnvHashMap::default();
                // "new_dec" represents the configuration we are currently iterating
                // over.
                let mut new_dec = dec as u64;
                for j in 0..ny {
                    for i in 0..nx {
                        sieve[new_dec as usize] = false;
                        let new_p = match decs.get(&new_dec) {
                            Some(&p) => p + phase(i, j),
                            None     => phase(i, j)
                        };
                        decs.insert(new_dec, new_p);
                        new_dec = translate_x(new_dec, nx, ny);
                    }
                    new_dec = translate_y(new_dec, nx, ny);
                }

                let lead = dec as u64;
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

    pub fn h_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn h_ss_pm(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_pm(&sites, &bfuncs)
    }

    pub fn h_ss_ppmm(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_ppmm(&sites, &bfuncs)
    }

    pub fn h_ss_pmz(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_pmz(&sites, &bfuncs)
    }

    pub fn ss_z(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = all_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn ss_pm(nx: u32, ny: u32, kx: u32, ky: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky);
        let sites = all_sites(nx, ny, l);
        ops::ss_pm(&sites, &bfuncs)
    }
}


pub mod ks {
    // in this specific case crystal momentum and total spin are conserved
    use num_complex::Complex;
    use num_bigint::*;
    use fnv::FnvHashMap;

    use common::*;
    use ops;
    use blochfunc::{BlochFunc, BlochFuncSet};
    use super::super::PI;
    use super::super::POW2;

    pub fn permute(mut v: Vec<i32>, lmax: u32) -> Vec<i32> {
        if v[0] > 0 { v[0] -= 1; v }
        else {
            let mut i = 1;
            while i < v.len() {
                if v[i] - v[i - 1] > 1 {
                    v[i] -= 1;
                    break
                }
                else { i += 1; }
            }
            let mut j = i;
            if i == v.len() { v[i - 1] = lmax as i32; j -= 1; }
            while j > 0 {
                v[j - 1] = v[j] - 1;
                j -= 1;
            }
            v
        }
    }

    pub fn compose(v: &Vec<i32>) -> u64 {
        v.iter().fold(0, |acc, &x| POW2[x as usize] + acc)
    }

    pub fn fac(n: BigUint) -> BigUint {
        if n == 0_u64.to_biguint().unwrap() { 1_u64.to_biguint().unwrap() }
        else { n.clone() * fac(n.clone() - 1_u64.to_biguint().unwrap()) }
    }

    pub fn choose(n: u32, c: u32) -> u64 {
        let n = n.to_biguint().unwrap();
        let c = c.to_biguint().unwrap();
        let ncr = fac(n.clone()) / (fac(c.clone()) * fac(n.clone() - c.clone()));
        ncr.to_bytes_le()
           .iter()
           .enumerate()
           .map(|(i, &x)| x as u64 * POW2[i as usize * 8])
           .sum()
    }

    pub fn sz_basis(n: u32, nup: u32) -> Vec<u64> {
        let mut l = (0..nup as i32).collect::<Vec<i32>>();
        let l_size = choose(n, nup);
        let mut sz_basis_states: Vec<u64> = Vec::with_capacity(l_size as usize);
        for _ in 0..l_size {
            l = permute(l, n - 1);
            let i = compose(&l);
            sz_basis_states.push(i);
        }
        sz_basis_states
    }

    pub fn bloch_states<'a>(nx: u32, ny: u32, kx: u32, ky: u32,
                        nup: u32) -> BlochFuncSet {
        let n = nx * ny;

        let sz_basis_states = sz_basis(n, nup);
        let mut szdec_to_ind: FnvHashMap<u64, usize> = FnvHashMap::default();
        let mut ind_to_szdec: FnvHashMap<usize, u64> = FnvHashMap::default();
        for (i, &bs) in sz_basis_states.iter().enumerate() {
            ind_to_szdec.insert(i, bs);
            szdec_to_ind.insert(bs, i);
        }

        let mut sieve = vec![true; sz_basis_states.len()];
        let mut bfuncs: Vec<BlochFunc> = Vec::new();
        let phase = |i, j| {
            let r = 1.;
            let ang1 = 2. * PI * (i * kx) as f64 / nx as f64;
            let ang2 = 2. * PI * (j * ky) as f64 / ny as f64;
            Complex::from_polar(&r, &(ang1 + ang2))
        };

        for ind in 0..sieve.len() {
            if sieve[ind]
            {   // if the corresponding entry of dec in "sieve" is not false,
                // we find all translations of dec and put them in a BlochFunc
                // then mark all corresponding entries in "sieve" as false.

                // "decs" is a hashtable that holds vectors whose entries
                // correspond to Bloch function constituent configurations which
                // are mapped to single decimals that represent the leading states.
                let mut decs: FnvHashMap<u64, Complex<f64>> = FnvHashMap::default();
                // "new_dec" represents the configuration we are currently iterating
                // over.
                let dec = *ind_to_szdec.get(&ind).unwrap() as u64;
                let mut new_dec = dec;
                let mut new_ind = ind;
                for j in 0..ny {
                    for i in 0..nx {
                        sieve[new_ind as usize] = false;
                        let new_p = match decs.get(&new_dec) {
                            Some(&p) => p + phase(i, j),
                            None     => phase(i, j)
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
        println!("Bloch table ready!");
        table
    }

    pub fn h_ss_z(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn h_ss_pm(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = interacting_sites(nx, ny, l);
        ops::ss_pm(&sites, &bfuncs)
    }

    pub fn ss_z(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = all_sites(nx, ny, l);
        ops::ss_z(&sites, &bfuncs)
    }

    pub fn ss_pm(nx: u32, ny: u32, kx: u32, ky: u32, nup: u32, l: u32)
        -> CoordMatrix<CComplex<f64>> {
        let bfuncs = bloch_states(nx, ny, kx, ky, nup);
        let sites = all_sites(nx, ny, l);
        ops::ss_pm(&sites, &bfuncs)
    }
}
