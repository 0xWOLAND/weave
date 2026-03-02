use crate::{Grid, Representable};

#[derive(Clone)]
pub struct Image<T: Copy> {
    width: usize,
    height: usize,
    pub data: Vec<T>,
}

impl<T: Copy> Image<T> {
    pub fn new(width: usize, height: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), width * height);
        Self { width, height, data }
    }
}

impl<G: Grid> From<&G> for Image<G::Elem> {
    fn from(grid: &G) -> Self {
        let width = grid.width();
        let height = grid.height();
        let data: Vec<_> = grid.iter().collect();

        Self::new(width, height, data)
    }
}

impl<T: Copy> Grid for Image<T> {
    type Elem = T;

    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }

    fn at(&self, x: usize, y: usize) -> T {
        self.data[y * self.width + x]
    }
}

#[derive(Clone)]
pub struct Field<const W: usize, const H: usize, T: Copy> {
    data: Vec<T>,
}

impl<const W: usize, const H: usize, T: Copy> Grid for Field<W, H, T> {
    type Elem = T;

    fn width(&self) -> usize { W }
    fn height(&self) -> usize { H }

    fn at(&self, x: usize, y: usize) -> T {
        self.data[y * W + x]
    }
}

impl<const W: usize, const H: usize, T: Copy> Representable for Field<W, H, T> {
    type Index = (usize, usize);

    fn index(&self, (x, y): Self::Index) -> Self::Elem {
        <Self as Grid>::at(self, x, y)
    }

    fn tabulate(f: impl Fn(Self::Index) -> Self::Elem) -> Self {
        let mut data = Vec::with_capacity(W * H);

        for y in 0..H {
            for x in 0..W {
                data.push(f((x, y)));
            }
        }

        Self { data }
    }
}
