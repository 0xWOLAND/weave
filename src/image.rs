use std::{any::TypeId, path::Path};

use image::{RgbaImage, error::ParameterError, error::ParameterErrorKind};

use crate::{Image, LoweringError, Shape, backend::EvalContext};

#[derive(Clone)]
pub struct Grid<T: Copy, const N: usize> {
    pub shape: Shape<N>,
    pub data: Vec<T>,
}

pub type Raster<T> = Grid<T, 2>;

pub struct GridIter<'a, T: Copy, const N: usize> {
    grid: &'a Grid<T, N>,
    next: usize,
}

impl<'a, T: Copy, const N: usize> GridIter<'a, T, N> {
    pub fn new(grid: &'a Grid<T, N>) -> Self {
        Self { grid, next: 0 }
    }
}

impl<T: Copy, const N: usize> Iterator for GridIter<'_, T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.grid.shape.size() {
            return None;
        }

        let index = self.grid.shape.unflatten(self.next);
        self.next += 1;
        Some(self.grid.at(index))
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for GridIter<'_, T, N> {
    fn len(&self) -> usize {
        self.grid.shape.size() - self.next
    }
}

impl<T: Copy, const N: usize> Grid<T, N> {
    pub fn new(shape: [usize; N], data: Vec<T>) -> Self {
        let shape = Shape(shape);
        assert_eq!(data.len(), shape.size());
        Self { shape, data }
    }

    pub fn shape(&self) -> Shape<N> {
        self.shape
    }

    pub fn sample(&self, index: [usize; N]) -> T {
        self.at(index)
    }

    pub fn at(&self, index: [usize; N]) -> T {
        self.data[self.shape.flatten(index)]
    }

    pub fn iter(&self) -> GridIter<'_, T, N> {
        GridIter::new(self)
    }

    pub fn tabulate(shape: [usize; N], mut f: impl FnMut([usize; N]) -> T) -> Self {
        let shape = Shape(shape);
        let len = shape.size();
        let mut data = Vec::with_capacity(len);

        for flat in 0..len {
            data.push(f(shape.unflatten(flat)));
        }

        Self { shape, data }
    }
}

impl<T: Copy + 'static, const N: usize> Image<N> for Grid<T, N> {
    type Pixel = T;

    fn shape(&self) -> Shape<N> {
        self.shape
    }

    fn eval<B: EvalContext<Self::Pixel, N>>(
        &self,
        backend: &mut B,
        index: [usize; N],
    ) -> Result<Self::Pixel, LoweringError> {
        backend.load(self, index)
    }

    fn source_grid(&self) -> Option<&Grid<Self::Pixel, N>> {
        Some(self)
    }

    fn lower_mlir(
        &self,
        indices: &[String],
        next_id: &mut usize,
        body: &mut String,
    ) -> Result<String, LoweringError> {
        if TypeId::of::<T>() != TypeId::of::<f32>() {
            return Err(LoweringError::UnsupportedOperation(
                "melior backend currently supports only f32 arithmetic images",
            ));
        }

        let value_name = format!("%v{next_id}");
        *next_id += 1;
        let indices_text = indices.join(", ");
        let memref_ty = {
            let extents = self
                .shape
                .0
                .iter()
                .map(|extent| extent.to_string())
                .collect::<Vec<_>>()
                .join("x");
            format!("memref<{extents}xf32>")
        };

        body.push_str(&format!(
            "          {value_name} = memref.load %input[{indices_text}] : {memref_ty}\n"
        ));
        Ok(value_name)
    }
}

impl<I: Image<N>, const N: usize> From<&I> for Grid<I::Pixel, N>
where
    I::Pixel: Copy
        + PartialOrd
        + std::ops::Add<Output = I::Pixel>
        + std::ops::Sub<Output = I::Pixel>
        + std::ops::Mul<Output = I::Pixel>
        + std::ops::Div<Output = I::Pixel>
        + std::ops::Neg<Output = I::Pixel>,
{
    fn from(image: &I) -> Self {
        crate::ImageExt::materialize(image)
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
