use std::iter::successors;

use crate::{GridLike, Store};

#[derive(Clone, Copy)]
pub struct View<G: GridLike<N>, const N: usize> {
    grid: G,
    position: [usize; N],
}

impl<G: GridLike<N>, const N: usize> View<G, N> {
    pub fn new(grid: G, position: [usize; N]) -> Self {
        Self { grid, position }
    }

    pub fn shape(&self) -> crate::grid::Shape<N> {
        self.grid.shape()
    }

    pub fn position(&self) -> [usize; N] {
        self.position
    }

    pub fn extract(&self) -> G::Elem {
        self.grid.at(self.position)
    }

    pub fn get(&self, offset: [isize; N]) -> Option<G::Elem> {
        let mut next = [0; N];
        let shape = self.grid.shape();

        for axis in 0..N {
            let coord = self.position[axis] as isize + offset[axis];

            if coord < 0 || coord >= shape.0[axis] as isize {
                return None;
            }

            next[axis] = coord as usize;
        }

        Some(self.grid.at(next))
    }

    pub fn iterate<A, F>(&self, seed: A, step: F) -> impl Iterator<Item = A>
    where
        A: Copy,
        F: Fn(A) -> A,
    {
        successors(Some(seed), move |&state| Some(step(state)))
    }
}

impl<'a, G: GridLike<N>, const N: usize> From<View<&'a G, N>> for Store<'a, [usize; N], G::Elem> {
    fn from(view: View<&'a G, N>) -> Self {
        let grid = view.grid;
        Store::new(view.position, move |position| grid.at(position))
    }
}
