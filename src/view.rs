use std::iter::successors;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::Image;

#[derive(Clone, Copy)]
pub struct View<I: Image<N>, const N: usize> {
    image: I,
    position: [usize; N],
}

impl<I: Image<N>, const N: usize> View<I, N> {
    pub fn new(image: I, position: [usize; N]) -> Self {
        Self { image, position }
    }

    pub fn shape(&self) -> crate::grid::Shape<N> {
        self.image.shape()
    }

    pub fn position(&self) -> [usize; N] {
        self.position
    }

    pub fn extract(&self) -> I::Pixel
    where
        I::Pixel: PartialOrd
            + Add<Output = I::Pixel>
            + Sub<Output = I::Pixel>
            + Mul<Output = I::Pixel>
            + Div<Output = I::Pixel>
            + Neg<Output = I::Pixel>,
    {
        self.image.sample(self.position)
    }

    pub fn get(&self, offset: [isize; N]) -> Option<I::Pixel>
    where
        I::Pixel: PartialOrd
            + Add<Output = I::Pixel>
            + Sub<Output = I::Pixel>
            + Mul<Output = I::Pixel>
            + Div<Output = I::Pixel>
            + Neg<Output = I::Pixel>,
    {
        let mut next = [0; N];
        let shape = self.image.shape();

        for axis in 0..N {
            let coord = self.position[axis] as isize + offset[axis];

            if coord < 0 || coord >= shape.0[axis] as isize {
                return None;
            }

            next[axis] = coord as usize;
        }

        Some(self.image.sample(next))
    }

    pub fn sample_at(&self, index: [usize; N]) -> Option<I::Pixel>
    where
        I::Pixel: PartialOrd
            + Add<Output = I::Pixel>
            + Sub<Output = I::Pixel>
            + Mul<Output = I::Pixel>
            + Div<Output = I::Pixel>
            + Neg<Output = I::Pixel>,
    {
        let shape = self.image.shape();

        for axis in 0..N {
            if index[axis] >= shape.0[axis] {
                return None;
            }
        }

        Some(self.image.sample(index))
    }

    pub fn collect<Pos>(&self, positions: Pos) -> Vec<I::Pixel>
    where
        Pos: IntoIterator<Item = [usize; N]>,
        I::Pixel: PartialOrd
            + Add<Output = I::Pixel>
            + Sub<Output = I::Pixel>
            + Mul<Output = I::Pixel>
            + Div<Output = I::Pixel>
            + Neg<Output = I::Pixel>,
    {
        positions
            .into_iter()
            .filter_map(|index| self.sample_at(index))
            .collect()
    }

    pub fn iterate<A, F>(&self, seed: A, step: F) -> impl Iterator<Item = A>
    where
        A: Copy,
        F: Fn(A) -> A,
    {
        successors(Some(seed), move |&state| Some(step(state)))
    }
}
