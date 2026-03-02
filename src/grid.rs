use std::{iter::FusedIterator, marker::PhantomData};

use crate::{Reader, View};

pub trait Grid {
    type Elem: Copy;

    fn width(&self) -> usize;
    fn height(&self) -> usize;

    fn at(&self, x: usize, y: usize) -> Self::Elem;

    fn iter(&self) -> GridIter<'_, Self>
    where
        Self: Sized,
    {
        GridIter::new(self)
    }

    fn map<F, B>(self, f: F) -> Map<Self, F, B>
    where
        Self: Sized,
        F: Fn(Self::Elem) -> B,
        B: Copy,
    {
        Map { grid: self, f, _marker: PhantomData }
    }

    fn duplicate(&self) -> Duplicate<'_, Self>
    where
        Self: Sized,
    {
        Duplicate { grid: self }
    }

    fn extend<F, B>(&self, f: F) -> Map<Duplicate<'_, Self>, F, B>
    where
        Self: Sized,
        F: for<'a> Fn(View<&'a Self>) -> B,
        B: Copy,
    {
        self.duplicate().map(f)
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

pub struct GridIter<'a, G: Grid> {
    grid: &'a G,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
}

impl<'a, G: Grid> GridIter<'a, G> {
    pub fn new(grid: &'a G) -> Self {
        Self {
            grid,
            width: grid.width(),
            height: grid.height(),
            x: 0,
            y: 0,
        }
    }
}

impl<G: Grid> Iterator for GridIter<'_, G> {
    type Item = G::Elem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.y == self.height {
            return None;
        }

        let elem = self.grid.at(self.x, self.y);
        self.x += 1;

        if self.x == self.width {
            self.x = 0;
            self.y += 1;
        }

        Some(elem)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len();
        (remaining, Some(remaining))
    }
}

impl<G: Grid> ExactSizeIterator for GridIter<'_, G> {
    fn len(&self) -> usize {
        self.width * self.height - self.y * self.width - self.x
    }
}

impl<G: Grid> FusedIterator for GridIter<'_, G> {}

impl<'a, G: Grid> From<&'a G> for Reader<'a, (usize, usize), G::Elem> {
    fn from(grid: &'a G) -> Self {
        Reader::new(move |&(x, y)| grid.at(x, y))
    }
}

impl<G: Grid> Grid for &G {
    type Elem = G::Elem;

    fn width(&self) -> usize { (*self).width() }
    fn height(&self) -> usize { (*self).height() }

    fn at(&self, x: usize, y: usize) -> Self::Elem {
        (*self).at(x, y)
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

pub struct Duplicate<'a, G: Grid> {
    grid: &'a G,
}

impl<'a, G: Grid> Grid for Duplicate<'a, G> {
    type Elem = View<&'a G>;

    fn width(&self) -> usize { self.grid.width() }
    fn height(&self) -> usize { self.grid.height() }

    fn at(&self, x: usize, y: usize) -> Self::Elem {
        View::new(self.grid, x, y)
    }
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
