const PI: f64 = 3.14159265358979323846;

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct SiteVector {
    x: u32,
    y: u32,
    nx: u32,
    ny: u32,
}

impl SiteVector {
    pub fn lattice_index(&self) -> u32 {
        self.x + self.nx * self.y
    }

    pub fn next_site(&self) -> SiteVector {
        let new_index = self.lattice_index() + 1;
        let new_x = new_index % self.nx;
        let new_y = new_index / self.nx;
        SiteVector { x: new_x, y: new_y, .. *self }
    }

    pub fn new(ordered_pair: (u32, u32), nx: u32, ny: u32) -> SiteVector {
        let x = ordered_pair.0;
        let y = ordered_pair.1;
        SiteVector { x, y, nx, ny }
    }

    pub fn from_index(index: u32, nx: u32, ny: u32) -> SiteVector {
        let x = index % nx;
        let y = index / nx;
        SiteVector { x, y, nx, ny }
    }
}

// periodic boundary conidtions
impl SiteVector {
    pub fn xhop(&self, stride: i32) -> SiteVector {
        let mut new_x = (self.x as i32) + stride;
        if new_x < 0 {
            new_x += self.nx as i32;
        }
        new_x %= self.nx as i32;
        SiteVector { x: new_x as u32, .. *self }
    }

    pub fn yhop(&self, stride: i32) -> SiteVector {
        let mut new_y = self.y as i32 + stride;
        if new_y < 0 {
            new_y += self.ny as i32;
        }
        new_y %= self.ny as i32;
        SiteVector { y: new_y as u32, .. *self }
    }
}

// for this specific model
impl SiteVector {
    pub fn angle_with(&self, other: &SiteVector) -> f64 {
        // type casting is necessary or else subtraction might cause overflow
        let dx = self.x as i32 - other.x as i32;
        let dy = self.y as i32 - other.y as i32;
        if dx == 0 && dy != 0 {
            -2. * PI / 3.
        } else if dx != 0 && dy == 0 {
            0.
        } else {
            2. * PI / 3.
        }
    }

    pub fn a1_hop(&self, stride: i32) -> Option<SiteVector> {
        let vec = self.xhop(stride);
        match vec == *self {
            true => None,
            false => Some(vec)
        }
    }

    pub fn a2_hop(&self, stride: i32) -> Option<SiteVector> {
        let vec = self.xhop(-stride).yhop(stride);
        match vec == *self {
            true => None,
            false => Some(vec)
        }
    }

    pub fn a3_hop(&self, stride: i32) -> Option<SiteVector> {
        let vec = self.yhop(-stride);
        match vec == *self {
            true => None,
            false => Some(vec)
        }
    }

    pub fn b1_hop(&self, stride: i32) -> Option<SiteVector> {
        let vec = self.xhop(stride).yhop(stride);
        match vec == *self {
            true => None,
            false => Some(vec)
        }
    }

    pub fn b2_hop(&self, stride: i32) -> Option<SiteVector> {
        let vec = self.xhop(-2 * stride).yhop(stride);
        match vec == *self {
            true => None,
            false => Some(vec)
        }
    }

    // this is a bit ugly. Perhaps clean this up a little when you have time
    pub fn b3_hop(&self, stride: i32) -> Option<SiteVector> {
        let v = self.b1_hop(-stride);
        let vec = match v {
            None => None,
            Some(vec) => match vec.b2_hop(-stride) {
                None => None,
                Some(vec) => match vec == *self {
                    true => None,
                    false => Some(vec)
                }
            }
        };
        vec
    }

    pub fn _neighboring_sites(&self,
                              strides: Vec<i32>,
                              funcs: Vec<fn(&SiteVector, i32) -> Option<SiteVector>>
                              ) -> Vec<SiteVector> {
        let mut neighbors = Vec::new();
        for &stride in strides.iter() {
            for &func in funcs.iter() {
                match func(self, stride) {
                    Some(vec) => neighbors.push(vec),
                    None => ()
                }
            }
        }
        neighbors
    }

    pub fn nearest_neighboring_sites(&self, all: bool) -> Vec<SiteVector> {
        let strides = if all { vec![1, -1] } else { vec![1] };
        let funcs: Vec<fn(&SiteVector, i32) -> Option<SiteVector>> = vec![
            SiteVector::a1_hop,
            SiteVector::a2_hop,
            SiteVector::a3_hop
        ];
        SiteVector::_neighboring_sites(&self, strides, funcs)
    }

    pub fn second_neighboring_sites(&self, all: bool) -> Vec<SiteVector> {
        let strides = if all { vec![1, -1] } else { vec![1] };
        let funcs: Vec<fn(&SiteVector, i32) -> Option<SiteVector>> = vec![
            SiteVector::b1_hop,
            SiteVector::b2_hop,
            SiteVector::b3_hop
        ];
        SiteVector::_neighboring_sites(&self, strides, funcs)
    }

    pub fn third_neighboring_sites(&self, all: bool) -> Vec<SiteVector> {
        let strides = if all { vec![2, -2] } else { vec![2] };
        let funcs: Vec<fn(&SiteVector, i32) -> Option<SiteVector>> = vec![
            SiteVector::a1_hop,
            SiteVector::a2_hop,
            SiteVector::a3_hop
        ];
        SiteVector::_neighboring_sites(&self, strides, funcs)
    }
}

