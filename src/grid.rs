#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape<const N: usize>(pub [usize; N]);

impl<const N: usize> Shape<N> {
    pub fn size(self) -> usize {
        self.0.iter().product()
    }

    pub fn flatten(self, index: [usize; N]) -> usize {
        let mut stride = 1;
        let mut flat = 0;

        for axis in (0..N).rev() {
            flat += index[axis] * stride;
            stride *= self.0[axis];
        }

        flat
    }

    pub fn unflatten(self, mut flat: usize) -> [usize; N] {
        let mut index = [0; N];

        for axis in (0..N).rev() {
            let extent = self.0[axis];
            index[axis] = flat % extent;
            flat /= extent;
        }

        index
    }
}
