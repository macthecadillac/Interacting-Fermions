#[macro_export]
macro_rules! make_int_type {
    ($n:ident, $t:ty) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        pub struct $n(pub $t);

        impl $n {
            pub fn raw_int(self) -> $t { self.0 }
        }

        impl BitAnd for $n {
            type Output = Self;

            fn bitand(self, rhs: Self) -> Self { $n(self.0 & rhs.0) }
        }

        impl BitAndAssign for $n {
            fn bitand_assign(&mut self, rhs: Self) { *self = $n(self.0 & rhs.0) }
        }

        impl BitOr for $n {
            type Output = Self;

            fn bitor(self, rhs: Self) -> Self { $n(self.0 | rhs.0) }
        }

        impl BitOrAssign for $n {
            fn bitor_assign(&mut self, rhs: Self) { *self = $n(self.0 | rhs.0) }
        }

        impl Add for $n {
            type Output = Self;

            fn add(self, rhs: Self) -> Self { $n(self.0 + rhs.0) }
        }

        impl AddAssign for $n {
            fn add_assign(&mut self, rhs: Self) { *self = $n(self.0 + rhs.0) }
        }

        impl Sub for $n {
            type Output = Self;

            fn sub(self, rhs: Self) -> Self { $n(self.0 - rhs.0) }
        }

        impl SubAssign for $n {
            fn sub_assign(&mut self, rhs: Self) { *self = $n(self.0 - rhs.0) }
        }

        impl Mul for $n {
            type Output = Self;

            fn mul(self, rhs: Self) -> Self { $n(self.0 * rhs.0) }
        }

        impl MulAssign for $n {
            fn mul_assign(&mut self, rhs: Self) { *self = $n(self.0 * rhs.0) }
        }

        impl Div for $n {
            type Output = Self;

            fn div(self, rhs: Self) -> Self { $n(self.0 / rhs.0) }
        }

        impl DivAssign for $n {
            fn div_assign(&mut self, rhs: Self) { *self = $n(self.0 / rhs.0) }
        }

        impl Rem for $n {
            type Output = Self;

            fn rem(self, rhs: Self) -> Self { $n(self.0 % rhs.0) }
        }

        impl RemAssign for $n {
            fn rem_assign(&mut self, rhs: Self) { *self = $n(self.0 % rhs.0) }
        }

        impl Ord for $n {
            fn cmp(&self, rhs: &Self) -> Ordering { self.0.cmp(&rhs.0) }
        }

        impl PartialOrd for $n {
            fn partial_cmp(&self, rhs: &Self) -> Option<Ordering> {
                Some(self.cmp(rhs))
            }
        }
    };
}
