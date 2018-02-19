pub mod k {
    // in this specific case crystal momentum is conserved
    use fnv::FnvHashMap;
    use num_complex::Complex;

    use blochfunc::{BlochFunc, BlochFuncSet};
    use common::*;
    use ops;

    const PI: f64 = 3.14159265358979323846;

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
    use common::List::{Nil, Node};
    use ops;
    use blochfunc::{BlochFunc, BlochFuncSet};

    const PI: f64 = 3.14159265358979323846;

    pub fn permute(l: List<i32>, n: u32) -> List<i32> {
        fn retr(apdx: List<i32>, acc: List<i32>, n: i32) -> List<i32> {
            match (apdx, acc.clone()) {
                (Nil, Node(_, _)) | (Nil, Nil) => acc,
                (Node(_, rmdr), Nil)           => retr(*rmdr, Nil.push(n), n),
                (Node(_, rmdr), Node(hd, _))   => retr(*rmdr, acc.push(hd - 1), n)
            }
        };

        fn aux(apdx: List<i32>, l: List<i32>, prev: i32, n: i32) -> List<i32> {
            match l {
                Nil          => retr(apdx, l, n),
                Node(hd, tl) => {
                    if hd - prev == 1 { aux(apdx.push(hd), *tl, hd, n) }
                    else { retr(apdx, tl.push(hd - 1), n) }
                }
            }
        };

        aux(Nil, l, -1, n as i32)
    }

    pub fn compose(l: List<i32>) -> u64 {
        fn aux(l: List<i32>, acc: u64) -> u64 {
            match l {
                List::Nil          => acc,
                List::Node(hd, tl) => aux(*tl, acc + 2_u64.pow(hd as u32))
            }
        };
        aux(l, 0)
    }

    pub fn fac(n: BigUint) -> BigUint {
        if n == 0_u64.to_biguint().unwrap() { 1_u64.to_biguint().unwrap() }
        else { n.clone() * fac(n.clone() - 1_u64.to_biguint().unwrap()) }
    }

    pub fn choose(n: u32, c: u32) -> u64 {
        let n = n.to_biguint().unwrap();
        let c = c.to_biguint().unwrap();
        let ncr = fac(n.clone()) / (fac(c.clone()) * fac(n.clone() - c.clone()));
        ncr.to_bytes_le().iter()
            .enumerate()
            .map(|(i, &x)| x as u64 * 2_u64.pow(i as u32 * 8))
            .sum()
    }

    pub fn sz_basis(n: u32, nup: u32) -> Vec<u64> {
        let mut l = (0..nup as i32).fold(List::Nil, |acc, x| acc.push(x))
                                   .rev();
        let l_size = choose(n, nup);
        let mut l_vec: Vec<List<i32>> = Vec::with_capacity(l_size as usize);
        for _ in 0..l_size {
            l = permute(l, n - 1);
            l_vec.push(l.clone());
        }
        l_vec.into_iter()
             .map(|v| compose(v))
             .collect::<Vec<u64>>()
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
