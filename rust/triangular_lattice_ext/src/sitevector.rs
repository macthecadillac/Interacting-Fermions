use common::{PI, I, Dim};

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct SiteVector {
    x: I,
    y: I,
    nx: Dim,
    ny: Dim,
}

impl SiteVector {
    pub fn lattice_index(&self) -> I {
        self.x + self.y * self.nx
    }

    pub fn next_site(&self) -> SiteVector {
        let new_index = self.lattice_index() + I(1);
        let new_x = new_index % self.nx;
        let new_y = new_index / self.nx;
        SiteVector { x: new_x, y: new_y, .. *self }
    }

    pub fn new(ordered_pair: (I, I), nx: Dim, ny: Dim)
        -> SiteVector {
        let x = ordered_pair.0;
        let y = ordered_pair.1;
        SiteVector { x, y, nx, ny }
    }

    pub fn from_index(index: I, nx: Dim, ny: Dim)
        -> SiteVector {
        let x = index % nx;
        let y = index / nx;
        SiteVector { x, y, nx, ny }
    }
}

// periodic boundary conidtions
impl SiteVector {
    pub fn xhop(&self, stride: I) -> SiteVector {
        let mut new_x = self.x + stride;
        if new_x < I(0) {
            new_x += self.nx;
        }
        new_x %= self.nx;
        SiteVector { x: new_x, .. *self }
    }

    pub fn yhop(&self, stride: I) -> SiteVector {
        let mut new_y = self.y + stride;
        if new_y < I(0) {
            new_y += self.ny;
        }
        new_y %= self.ny;
        SiteVector { y: new_y, .. *self }
    }
}

// for this specific model
impl SiteVector {
    pub fn angle_with(&self, other: &SiteVector) -> f64 {
        // type casting is necessary or else subtraction might cause overflow
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        if dx == I(0) && dy != I(0) {
            -2. * PI / 3.
        } else if dx != I(0) && dy == I(0) {
            0.
        } else {
            2. * PI / 3.
        }
    }

    pub fn a1_hop(&self, stride: I) -> Option<SiteVector> {
        let vec = self.xhop(stride);
        match vec == *self {
            true  => None,
            false => Some(vec)
        }
    }

    pub fn a2_hop(&self, stride: I) -> Option<SiteVector> {
        let vec = self.xhop(-stride).yhop(stride);
        match vec == *self {
            true  => None,
            false => Some(vec)
        }
    }

    pub fn a3_hop(&self, stride: I) -> Option<SiteVector> {
        let vec = self.yhop(-stride);
        match vec == *self {
            true  => None,
            false => Some(vec)
        }
    }

    pub fn b1_hop(&self, stride: I) -> Option<SiteVector> {
        let vec = self.xhop(stride).yhop(stride);
        match vec == *self {
            true  => None,
            false => Some(vec)
        }
    }

    pub fn b2_hop(&self, stride: I) -> Option<SiteVector> {
        let vec = self.xhop(I(-2) * stride).yhop(stride);
        match vec == *self {
            true  => None,
            false => Some(vec)
        }
    }

    // this is a bit ugly. Perhaps clean this up a little when you have time
    pub fn b3_hop(&self, stride: I) -> Option<SiteVector> {
        let v = self.b1_hop(-stride);
        let vec = match v {
            None      => None,
            Some(vec) => match vec.b2_hop(-stride) {
                None      => None,
                Some(vec) => match vec == *self {
                    true  => None,
                    false => Some(vec)
                }
            }
        };
        vec
    }

    pub fn _neighboring_sites(&self,
                              strides: Vec<I>,
                              funcs: Vec<fn(&SiteVector, I) -> Option<SiteVector>>
                              ) -> Vec<SiteVector> {
        let mut neighbors = Vec::new();
        for &stride in strides.iter() {
            for &func in funcs.iter() {
                match func(self, stride) {
                    Some(vec) => neighbors.push(vec),
                    None      => ()
                }
            }
        }
        neighbors
    }

    pub fn nearest_neighboring_sites(&self, all: bool) -> Vec<SiteVector> {
        let strides = if all { vec![I(1), I(-1)] } else { vec![I(1)] };
        let funcs: Vec<fn(&SiteVector, I) -> Option<SiteVector>> = vec![
            SiteVector::a1_hop,
            SiteVector::a2_hop,
            SiteVector::a3_hop
        ];
        SiteVector::_neighboring_sites(&self, strides, funcs)
    }

    pub fn second_neighboring_sites(&self, all: bool) -> Vec<SiteVector> {
        let strides = if all { vec![I(1), I(-1)] } else { vec![I(1)] };
        let funcs: Vec<fn(&SiteVector, I) -> Option<SiteVector>> = vec![
            SiteVector::b1_hop,
            SiteVector::b2_hop,
            SiteVector::b3_hop
        ];
        SiteVector::_neighboring_sites(&self, strides, funcs)
    }

    pub fn third_neighboring_sites(&self, all: bool) -> Vec<SiteVector> {
        let strides = if all { vec![I(2), I(-2)] } else { vec![I(2)] };
        let funcs: Vec<fn(&SiteVector, I) -> Option<SiteVector>> = vec![
            SiteVector::a1_hop,
            SiteVector::a2_hop,
            SiteVector::a3_hop
        ];
        SiteVector::_neighboring_sites(&self, strides, funcs)
    }
}

