use std::path::Path;

use image::{RgbaImage, error::ParameterError, error::ParameterErrorKind};

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

impl Grid<[u8; 4], 2> {
    pub fn save_rgba(&self, path: impl AsRef<Path>) -> image::ImageResult<()> {
        let [w, h] = self.shape.0;
        let pixels = (0..h)
            .flat_map(|y| (0..w).map(move |x| self.at([x, y])))
            .flat_map(|rgba| rgba)
            .collect();

        RgbaImage::from_raw(w as u32, h as u32, pixels)
            .ok_or_else(|| {
                image::ImageError::Parameter(ParameterError::from_kind(
                    ParameterErrorKind::DimensionMismatch,
                ))
            })?
            .save(path)
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
