use std::marker::PhantomData;

pub trait Grid {
    type Elem: Copy;

    fn width(&self) -> usize;
    fn height(&self) -> usize;

    fn at(&self, x: usize, y: usize) -> Self::Elem;

    fn map<F, B>(self, f: F) -> Map<Self, F, B>
    where
        Self: Sized,
        F: Fn(Self::Elem) -> B,
        B: Copy,
    {
        Map { grid: self, f, _marker: PhantomData }
    }

    fn extend<F, B>(self, f: F) -> Extend<Self, F, B>
    where
        Self: Sized,
        F: Fn(View<Self>) -> B,
        B: Copy,
    {
        Extend { grid: self, f, _marker: PhantomData }
    }

    fn zip<H>(self, other: H) -> Zip<Self, H>
    where
        Self: Sized,
        H: Grid,
    {
        assert_eq!(self.width(), other.width());
        assert_eq!(self.height(), other.height());
        Zip { left: self, right: other }
    }
}

pub trait Representable: Grid {
    type Index: Copy;

    fn index(&self, i: Self::Index) -> Self::Elem;

    fn tabulate(f: impl Fn(Self::Index) -> Self::Elem) -> Self;
}

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


pub struct View<'a, G: Grid> {
    grid: &'a G,
    x: usize,
    y: usize,
}

impl<'a, G: Grid> View<'a, G> {
    pub fn x(&self) -> usize {
        self.x
    }

    pub fn y(&self) -> usize {
        self.y
    }

    pub fn width(&self) -> usize {
        self.grid.width()
    }

    pub fn height(&self) -> usize {
        self.grid.height()
    }

    pub fn extract(&self) -> G::Elem {
        self.grid.at(self.x, self.y)
    }

    pub fn get(&self, dx: isize, dy: isize) -> Option<G::Elem> {
        let nx = self.x as isize + dx;
        let ny = self.y as isize + dy;

        if nx >= 0
            && ny >= 0
            && nx < self.grid.width() as isize
            && ny < self.grid.height() as isize
        {
            Some(self.grid.at(nx as usize, ny as usize))
        } else {
            None
        }
    }
}


pub struct Map<G, F, B> {
    grid: G,
    f: F,
    _marker: PhantomData<B>,
}

impl<G, F, B> Grid for Map<G, F, B>
where
    G: Grid,
    F: Fn(G::Elem) -> B,
    B: Copy,
{
    type Elem = B;

    fn width(&self) -> usize { self.grid.width() }
    fn height(&self) -> usize { self.grid.height() }

    fn at(&self, x: usize, y: usize) -> B {
        (self.f)(self.grid.at(x, y))
    }
}


pub struct Extend<G, F, B> {
    grid: G,
    f: F,
    _marker: PhantomData<B>,
}

pub struct Zip<G, H> {
    left: G,
    right: H,
}

impl<G, H> Grid for Zip<G, H>
where
    G: Grid,
    H: Grid,
{
    type Elem = (G::Elem, H::Elem);

    fn width(&self) -> usize { self.left.width() }
    fn height(&self) -> usize { self.left.height() }

    fn at(&self, x: usize, y: usize) -> Self::Elem {
        (self.left.at(x, y), self.right.at(x, y))
    }
}

impl<G, F, B> Grid for Extend<G, F, B>
where
    G: Grid,
    F: Fn(View<G>) -> B,
    B: Copy,
{
    type Elem = B;

    fn width(&self) -> usize { self.grid.width() }
    fn height(&self) -> usize { self.grid.height() }

    fn at(&self, x: usize, y: usize) -> B {
        let view = View { grid: &self.grid, x, y };
        (self.f)(view)
    }
}


pub fn materialize<G: Grid>(grid: G) -> Image<G::Elem> {
    let mut data = Vec::with_capacity(grid.width() * grid.height());

    for y in 0..grid.height() {
        for x in 0..grid.width() {
            data.push(grid.at(x, y));
        }
    }

    Image::new(grid.width(), grid.height(), data)
}
