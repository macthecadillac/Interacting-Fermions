use common;
use consv;
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
fn k_bloch_states() {
    let nx = 4;
    let ny = 4;
    let kx = 1;
    let ky = 3;
    let bfuncs = consv::k::bloch_states(nx, ny, kx, ky);
    assert_eq!(bfuncs.nonzero, 4080);
}

#[test]
fn choose() {
    let n = 6;
    let nup = 3;
    assert_eq!(consv::ks::choose(n, nup), 20);

    let n = 24;
    let nup = 12;
    assert_eq!(consv::ks::choose(n, nup), 2704156);
}

#[test]
fn permute() {
    use common::List;
    let l = [6, 5, 4, 3].iter().fold(List::Nil, |acc, &x| acc.push(x));
    let ans = [6, 5, 4, 2].iter().fold(List::Nil, |acc, &x| acc.push(x));
    assert_eq!(consv::ks::permute(l, 7), ans);

    let l = [6, 5, 4, 1, 0].iter().fold(List::Nil, |acc, &x| acc.push(x));
    let ans = [6, 5, 3, 2, 1].iter().fold(List::Nil, |acc, &x| acc.push(x));
    assert_eq!(consv::ks::permute(l, 9), ans);

    let l = [2, 1, 0].iter().fold(List::Nil, |acc, &x| acc.push(x));
    let ans = [5, 4, 3].iter().fold(List::Nil, |acc, &x| acc.push(x));
    assert_eq!(consv::ks::permute(l, 5), ans);
}

#[test]
fn compose() {
    use common::List;
    let l = [0, 1, 2, 3].iter().fold(List::Nil, |acc, &x| acc.push(x));
    assert_eq!(consv::ks::compose(l), 15);

    let l = [1, 3, 5].iter().fold(List::Nil, |acc, &x| acc.push(x));
    assert_eq!(consv::ks::compose(l), 42);
}

#[test]
fn sz_basis() {
    let n = 6;
    let nup = 3;
    assert_eq!(consv::ks::sz_basis(n, nup).len(), 20);
}

// #[test]
// fn ks_bloch_states() {
//     let nx = 4;
//     let ny = 4;
//     let kx = 1;
//     let ky = 3;
//     let bfuncs = consv::ks::bloch_states(nx, ny, kx, ky);
//     assert_eq!(bfuncs.nonzero, 4080);
// }
