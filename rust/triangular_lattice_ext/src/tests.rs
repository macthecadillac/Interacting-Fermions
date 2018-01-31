use common;
use num_complex::Complex;

#[test]
fn translate_x() {
    let d1 = 10;
    let d2 = 5;
    assert_eq!(common::translate_x(d1, 4, 6), d2);
}

#[test]
fn translate_y() {
    let d1 = 2;
    let d2 = 8192;
    assert_eq!(common::translate_y(d1, 4, 4), d2);
}

#[test]
fn exchange_spin_flips() {
    assert_eq!(common::exchange_spin_flips(10, 2, 8), (false, false));
    assert_eq!(common::exchange_spin_flips(93, 1, 32), (true, false));
}

#[test]
fn repeated_spins() {
    assert_eq!(common::repeated_spins(93, 1, 32), (false, false));
    assert_eq!(common::repeated_spins(93, 4, 64), (true, false));
    assert_eq!(common::repeated_spins(93, 128, 256), (false, true));
}

#[test]
fn generate_bonds() {
    let bonds = common::generate_bonds(4, 6);
    assert_eq!(bonds[0].len(), 72);
    assert_eq!(bonds[1].len(), 72);
    assert_eq!(bonds[2].len(), 72);

    let bonds = common::generate_bonds(6, 6);
    assert_eq!(bonds[0].len(), 108);
    assert_eq!(bonds[1].len(), 108);
    assert_eq!(bonds[2].len(), 108);
}

#[test]
fn gamma() {
    let nx = 4;
    let ny = 3;
    let s1 = 32;
    let s2 = 256;
    let gamma = common::gamma(nx, ny, s1, s2);
    println!("{}", gamma);
    assert!((gamma - Complex::new(-0.5, 0.866025403784)).norm() < 1e-8);
}

#[test]
fn _bloch_states() {
    let nx = 4;
    let ny = 4;
    let kx = 1;
    let ky = 3;
    let bfuncs = common::bloch_states(nx, ny, kx, ky);
    assert_eq!(bfuncs.nonzero, 4080);
}
