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
