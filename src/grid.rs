use std::{iter::FusedIterator, marker::PhantomData};

use crate::{Reader, View};

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

pub trait GridLike<const N: usize> {
    type Elem: Copy;

    fn shape(&self) -> Shape<N>;
    fn at(&self, index: [usize; N]) -> Self::Elem;

    fn iter(&self) -> GridIter<'_, Self, N>
    where
        Self: Sized,
    {
        GridIter::new(self)
    }

    fn map<F, B>(self, f: F) -> Map<Self, F, B, N>
    where
        Self: Sized,
        F: Fn(Self::Elem) -> B,
        B: Copy,
    {
        Map {
            grid: self,
            f,
            _marker: PhantomData,
        }
    }

    fn duplicate(&self) -> Duplicate<'_, Self, N>
    where
        Self: Sized,
    {
        Duplicate { grid: self }
    }

    fn extend<F, B>(&self, f: F) -> Map<Duplicate<'_, Self, N>, F, B, N>
    where
        Self: Sized,
        F: for<'a> Fn(View<&'a Self, N>) -> B,
        B: Copy,
    {
        self.duplicate().map(f)
    }

    fn zip<H>(self, other: H) -> Zip<Self, H, N>
    where
        Self: Sized,
        H: GridLike<N>,
    {
        assert_eq!(self.shape(), other.shape());
        Zip {
            left: self,
            right: other,
        }
    }

    fn remap<F>(&self, shape: [usize; N], fill: Self::Elem, f: F) -> Remap<'_, Self, F, N>
    where
        Self: Sized,
        F: Fn([usize; N]) -> Option<[usize; N]>,
    {
        Remap {
            grid: self,
            shape: Shape(shape),
            fill,
            f,
        }
    }
}

pub trait Representable<const N: usize>: GridLike<N> {
    fn tabulate(shape: [usize; N], f: impl FnMut([usize; N]) -> Self::Elem) -> Self;
}

pub struct GridIter<'a, G: GridLike<N>, const N: usize> {
    grid: &'a G,
    shape: Shape<N>,
    next: usize,
    len: usize,
}

impl<'a, G: GridLike<N>, const N: usize> GridIter<'a, G, N> {
    pub fn new(grid: &'a G) -> Self {
        let shape = grid.shape();
        let len = shape.size();

        Self {
            grid,
            shape,
            next: 0,
            len,
        }
    }
}

impl<G: GridLike<N>, const N: usize> Iterator for GridIter<'_, G, N> {
    type Item = G::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }

        let index = self.shape.unflatten(self.next);
        self.next += 1;
        Some(self.grid.at(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len();
        (remaining, Some(remaining))
    }
}

impl<G: GridLike<N>, const N: usize> ExactSizeIterator for GridIter<'_, G, N> {
    fn len(&self) -> usize {
        self.len - self.next
    }
}

impl<G: GridLike<N>, const N: usize> FusedIterator for GridIter<'_, G, N> {}

impl<'a, G: GridLike<N>, const N: usize> From<&'a G> for Reader<'a, [usize; N], G::Elem> {
    fn from(grid: &'a G) -> Self {
        Reader::new(move |index| grid.at(*index))
    }
}

impl<G: GridLike<N>, const N: usize> GridLike<N> for &G {
    type Elem = G::Elem;

    fn shape(&self) -> Shape<N> {
        (*self).shape()
    }

    fn at(&self, index: [usize; N]) -> Self::Elem {
        (*self).at(index)
    }
}

pub struct Map<G, F, B, const N: usize> {
    grid: G,
    f: F,
    _marker: PhantomData<B>,
}

impl<G, F, B, const N: usize> GridLike<N> for Map<G, F, B, N>
where
    G: GridLike<N>,
    F: Fn(G::Elem) -> B,
    B: Copy,
{
    type Elem = B;

    fn shape(&self) -> Shape<N> {
        self.grid.shape()
    }

    fn at(&self, index: [usize; N]) -> B {
        (self.f)(self.grid.at(index))
    }
}

pub struct Duplicate<'a, G: GridLike<N>, const N: usize> {
    grid: &'a G,
}

impl<'a, G: GridLike<N>, const N: usize> GridLike<N> for Duplicate<'a, G, N> {
    type Elem = View<&'a G, N>;

    fn shape(&self) -> Shape<N> {
        self.grid.shape()
    }

    fn at(&self, index: [usize; N]) -> Self::Elem {
        View::new(self.grid, index)
    }
}

pub struct Zip<G, H, const N: usize> {
    left: G,
    right: H,
}

impl<G, H, const N: usize> GridLike<N> for Zip<G, H, N>
where
    G: GridLike<N>,
    H: GridLike<N>,
{
    type Elem = (G::Elem, H::Elem);

    fn shape(&self) -> Shape<N> {
        self.left.shape()
    }

    fn at(&self, index: [usize; N]) -> Self::Elem {
        (self.left.at(index), self.right.at(index))
    }
}

pub struct Remap<'a, G: GridLike<N>, F, const N: usize> {
    grid: &'a G,
    shape: Shape<N>,
    fill: G::Elem,
    f: F,
}

impl<'a, G, F, const N: usize> GridLike<N> for Remap<'a, G, F, N>
where
    G: GridLike<N>,
    F: Fn([usize; N]) -> Option<[usize; N]>,
{
    type Elem = G::Elem;

    fn shape(&self) -> Shape<N> {
        self.shape
    }

    fn at(&self, index: [usize; N]) -> Self::Elem {
        (self.f)(index)
            .map(|index| self.grid.at(index))
            .unwrap_or(self.fill)
    }
}
