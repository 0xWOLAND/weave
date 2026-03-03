use crate::{GridLike, Representable, grid::Shape};

#[derive(Clone)]
pub struct Grid<T: Copy, const N: usize> {
    pub shape: Shape<N>,
    pub data: Vec<T>,
}

pub type Image<T> = Grid<T, 2>;

impl<T: Copy, const N: usize> Grid<T, N> {
    pub fn new(shape: [usize; N], data: Vec<T>) -> Self {
        let shape = Shape(shape);
        assert_eq!(data.len(), shape.size());
        Self { shape, data }
    }
}

impl<G: GridLike<N>, const N: usize> From<&G> for Grid<G::Elem, N> {
    fn from(grid: &G) -> Self {
        Self {
            shape: grid.shape(),
            data: grid.iter().collect(),
        }
    }
}

impl<T: Copy, const N: usize> GridLike<N> for Grid<T, N> {
    type Elem = T;

    fn shape(&self) -> Shape<N> {
        self.shape
    }

    fn at(&self, index: [usize; N]) -> T {
        self.data[self.shape.flatten(index)]
    }
}

impl<T: Copy, const N: usize> Representable<N> for Grid<T, N> {
    fn tabulate(shape: [usize; N], f: impl Fn([usize; N]) -> Self::Elem) -> Self {
        let shape = Shape(shape);
        let len = shape.size();
        let mut data = Vec::with_capacity(len);

        for flat in 0..len {
            data.push(f(shape.unflatten(flat)));
        }

        Self { shape, data }
    }
}
