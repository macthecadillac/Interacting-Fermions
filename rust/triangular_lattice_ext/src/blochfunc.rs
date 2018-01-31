use std::cmp::Ordering;
use fnv::FnvHashMap;
use num_complex::Complex;

#[derive(Clone, Debug)]
pub struct BlochFunc {
    pub lead: u32,
    pub decs: FnvHashMap<u32, Complex<f64>>,
    pub norm: f64,
}

impl Ord for BlochFunc {
    fn cmp(&self, other: &BlochFunc) -> Ordering {
        self.lead.cmp(&other.lead)
    }
}

impl PartialOrd for BlochFunc {
    fn partial_cmp(&self, other: &BlochFunc) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BlochFunc {
    fn eq(&self, other: &BlochFunc) -> bool{
        self.lead == other.lead
    }
}

impl Eq for BlochFunc {}

#[derive(Clone, Debug)]
pub struct BlochFuncSet {
    pub data: Vec<BlochFunc>,
    pub nonzero: u32,
}

impl<'a> BlochFuncSet {
    pub fn create(bfuncs: Vec<BlochFunc>) -> BlochFuncSet {
        let data = bfuncs;
        let nonzero = data.len() as u32;
        BlochFuncSet{ data, nonzero }
    }

    pub fn sort(&mut self) {
        self.data.sort();
    }
    
    pub fn iter(&self) -> BlochFuncSetIterator {
        BlochFuncSetIterator::new(&self.data)
    }

    pub fn build_dict(bfuncs: &BlochFuncSet) -> FnvHashMap<&u32, &BlochFunc> {
        let mut hashtable = FnvHashMap::default();
        for bfunc in bfuncs.data.iter() {
            for dec in bfunc.decs.keys() {
                hashtable.insert(dec, bfunc);
            }
        }
        hashtable
    }
}

pub struct BlochFuncSetIterator<'a> {
    pub ptr: usize,
    pub len: usize,
    pub data: &'a Vec<BlochFunc>
}

impl<'a> BlochFuncSetIterator<'a> {
    pub fn new(data: &'a Vec<BlochFunc>) -> BlochFuncSetIterator<'a> {
        let ptr = 0;
        let len = data.len();
        BlochFuncSetIterator { ptr, len, data }
    }
}

impl<'a> Iterator for BlochFuncSetIterator<'a> {
    type Item = &'a BlochFunc;

    fn next(&mut self) -> Option<Self::Item> {
        self.ptr += 1;
        if self.ptr <= self.len {
            Some(&self.data[self.ptr - 1])
        } else {
            None
        }
    }
}
