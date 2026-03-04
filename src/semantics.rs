use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Backend, CpuBackend, Grid, LoweringError, Shape, backend::EvalContext};

mod private {
    use crate::Grid;

    pub trait Sealed {}

    #[derive(Clone, Copy)]
    pub enum UnaryOp {
        Neg,
        AddConst(f32),
        SubConst(f32),
        MulConst(f32),
        DivConst(f32),
        Threshold {
            cutoff: f32,
            if_true: f32,
            if_false: f32,
        },
    }

    pub struct ArithmeticImage<I, const N: usize> {
        pub image: I,
        pub op: UnaryOp,
    }

    impl<T: Copy, const N: usize> Sealed for Grid<T, N> {}
    impl<I: Sealed + ?Sized> Sealed for &I {}
    impl<I: Sealed, const N: usize> Sealed for ArithmeticImage<I, N> {}
}

use private::{ArithmeticImage, Sealed, UnaryOp};

pub trait Image<const N: usize>: Sealed {
    type Pixel: Copy;

    fn shape(&self) -> Shape<N>;

    #[doc(hidden)]
    fn eval<B: EvalContext<Self::Pixel, N>>(
        &self,
        backend: &mut B,
        index: [usize; N],
    ) -> Result<Self::Pixel, LoweringError>;

    #[doc(hidden)]
    fn source_grid(&self) -> Option<&Grid<Self::Pixel, N>>;

    #[doc(hidden)]
    fn lower_mlir(
        &self,
        indices: &[String],
        next_id: &mut usize,
        body: &mut String,
    ) -> Result<String, LoweringError>;

    fn sample(&self, index: [usize; N]) -> Self::Pixel
    where
        Self::Pixel: PartialOrd
            + Add<Output = Self::Pixel>
            + Sub<Output = Self::Pixel>
            + Mul<Output = Self::Pixel>
            + Div<Output = Self::Pixel>
            + Neg<Output = Self::Pixel>,
    {
        let mut backend = crate::backend::PureEval;
        self.eval(&mut backend, index)
            .expect("pure evaluation is infallible")
    }
}

impl<I: Image<N>, const N: usize> Image<N> for &I {
    type Pixel = I::Pixel;

    fn shape(&self) -> Shape<N> {
        (*self).shape()
    }

    fn eval<B: EvalContext<Self::Pixel, N>>(
        &self,
        backend: &mut B,
        index: [usize; N],
    ) -> Result<Self::Pixel, LoweringError> {
        (*self).eval(backend, index)
    }

    fn source_grid(&self) -> Option<&Grid<Self::Pixel, N>> {
        (*self).source_grid()
    }

    fn lower_mlir(
        &self,
        indices: &[String],
        next_id: &mut usize,
        body: &mut String,
    ) -> Result<String, LoweringError> {
        (*self).lower_mlir(indices, next_id, body)
    }
}

impl<I, const N: usize> Image<N> for ArithmeticImage<I, N>
where
    I: Image<N, Pixel = f32>,
{
    type Pixel = f32;

    fn shape(&self) -> Shape<N> {
        self.image.shape()
    }

    fn eval<B: EvalContext<Self::Pixel, N>>(
        &self,
        backend: &mut B,
        index: [usize; N],
    ) -> Result<Self::Pixel, LoweringError> {
        let value = self.image.eval(backend, index)?;

        match self.op {
            UnaryOp::Neg => backend.neg(value),
            UnaryOp::AddConst(rhs) => {
                let rhs = backend.literal(rhs)?;
                backend.add(value, rhs)
            }
            UnaryOp::SubConst(rhs) => {
                let rhs = backend.literal(rhs)?;
                backend.sub(value, rhs)
            }
            UnaryOp::MulConst(rhs) => {
                let rhs = backend.literal(rhs)?;
                backend.mul(value, rhs)
            }
            UnaryOp::DivConst(rhs) => {
                let rhs = backend.literal(rhs)?;
                backend.div(value, rhs)
            }
            UnaryOp::Threshold {
                cutoff,
                if_true,
                if_false,
            } => {
                let cutoff = backend.literal(cutoff)?;
                let condition = backend.gt(value, cutoff)?;
                let if_true = backend.literal(if_true)?;
                let if_false = backend.literal(if_false)?;
                backend.select(condition, if_true, if_false)
            }
        }
    }

    fn source_grid(&self) -> Option<&Grid<Self::Pixel, N>> {
        self.image.source_grid()
    }

    fn lower_mlir(
        &self,
        indices: &[String],
        next_id: &mut usize,
        body: &mut String,
    ) -> Result<String, LoweringError> {
        let value = self.image.lower_mlir(indices, next_id, body)?;

        match self.op {
            UnaryOp::Neg => {
                let zero = next_value(next_id);
                body.push_str(&format!("          {zero} = arith.constant 0.0 : f32\n"));
                let out = next_value(next_id);
                body.push_str(&format!("          {out} = arith.subf {zero}, {value} : f32\n"));
                Ok(out)
            }
            UnaryOp::AddConst(rhs) => lower_const_op("arith.addf", rhs, value, next_id, body),
            UnaryOp::SubConst(rhs) => lower_const_op("arith.subf", rhs, value, next_id, body),
            UnaryOp::MulConst(rhs) => lower_const_op("arith.mulf", rhs, value, next_id, body),
            UnaryOp::DivConst(rhs) => lower_const_op("arith.divf", rhs, value, next_id, body),
            UnaryOp::Threshold {
                cutoff,
                if_true,
                if_false,
            } => {
                let cutoff_name = next_value(next_id);
                body.push_str(&format!(
                    "          {cutoff_name} = arith.constant {cutoff:.9e} : f32\n"
                ));
                let pred = next_value(next_id);
                body.push_str(&format!(
                    "          {pred} = arith.cmpf ogt, {value}, {cutoff_name} : f32\n"
                ));
                let if_true_name = next_value(next_id);
                body.push_str(&format!(
                    "          {if_true_name} = arith.constant {if_true:.9e} : f32\n"
                ));
                let if_false_name = next_value(next_id);
                body.push_str(&format!(
                    "          {if_false_name} = arith.constant {if_false:.9e} : f32\n"
                ));
                let out = next_value(next_id);
                body.push_str(&format!(
                    "          {out} = arith.select {pred}, {if_true_name}, {if_false_name} : f32\n"
                ));
                Ok(out)
            }
        }
    }
}

fn next_value(next_id: &mut usize) -> String {
    let name = format!("%v{next_id}");
    *next_id += 1;
    name
}

fn lower_const_op(
    op: &str,
    rhs: f32,
    value: String,
    next_id: &mut usize,
    body: &mut String,
) -> Result<String, LoweringError> {
    let rhs_name = next_value(next_id);
    body.push_str(&format!(
        "          {rhs_name} = arith.constant {rhs:.9e} : f32\n"
    ));
    let out = next_value(next_id);
    body.push_str(&format!("          {out} = {op} {value}, {rhs_name} : f32\n"));
    Ok(out)
}

pub trait ImageExt<const N: usize>: Image<N> {
    fn neg(self) -> impl Image<N, Pixel = f32>
    where
        Self: Sized + Image<N, Pixel = f32>,
    {
        ArithmeticImage {
            image: self,
            op: UnaryOp::Neg,
        }
    }

    fn add_const(self, value: f32) -> impl Image<N, Pixel = f32>
    where
        Self: Sized + Image<N, Pixel = f32>,
    {
        ArithmeticImage {
            image: self,
            op: UnaryOp::AddConst(value),
        }
    }

    fn sub_const(self, value: f32) -> impl Image<N, Pixel = f32>
    where
        Self: Sized + Image<N, Pixel = f32>,
    {
        ArithmeticImage {
            image: self,
            op: UnaryOp::SubConst(value),
        }
    }

    fn mul_const(self, value: f32) -> impl Image<N, Pixel = f32>
    where
        Self: Sized + Image<N, Pixel = f32>,
    {
        ArithmeticImage {
            image: self,
            op: UnaryOp::MulConst(value),
        }
    }

    fn div_const(self, value: f32) -> impl Image<N, Pixel = f32>
    where
        Self: Sized + Image<N, Pixel = f32>,
    {
        ArithmeticImage {
            image: self,
            op: UnaryOp::DivConst(value),
        }
    }

    fn threshold(self, cutoff: f32, if_true: f32, if_false: f32) -> impl Image<N, Pixel = f32>
    where
        Self: Sized + Image<N, Pixel = f32>,
    {
        ArithmeticImage {
            image: self,
            op: UnaryOp::Threshold {
                cutoff,
                if_true,
                if_false,
            },
        }
    }

    fn materialize(&self) -> Grid<Self::Pixel, N>
    where
        Self::Pixel: PartialOrd
            + Add<Output = Self::Pixel>
            + Sub<Output = Self::Pixel>
            + Mul<Output = Self::Pixel>
            + Div<Output = Self::Pixel>
            + Neg<Output = Self::Pixel>,
        CpuBackend: Backend<N, Self::Pixel, Output = Grid<Self::Pixel, N>>,
    {
        CpuBackend::materialize(self).expect("cpu materialization is infallible")
    }

    fn materialize_with<B>(&self) -> Result<B::Output, LoweringError>
    where
        B: Backend<N, Self::Pixel>,
    {
        B::materialize(self)
    }
}

impl<I: Image<N>, const N: usize> ImageExt<N> for I {}

#[cfg(test)]
mod tests {
    use crate::{Grid, Image, ImageExt, MeliorBackend};

    #[test]
    fn sample_matches_existing_indexing() {
        let grid = Grid::new([2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]);

        assert_eq!(Image::sample(&grid, [1, 0]), 3.0);
        assert_eq!(ImageExt::materialize(&grid).data, grid.data);
    }

    #[test]
    fn melior_matches_cpu_for_arithmetic_chain() {
        let grid = Grid::new([2, 2], vec![0.25_f32, 0.75, 1.0, -0.5]);
        let image = (&grid).mul_const(2.0).add_const(0.5).threshold(1.0, 1.0, 0.0);

        let cpu = image.materialize();
        let mlir = image
            .materialize_with::<MeliorBackend>()
            .expect("mlir materialization");

        assert_eq!(cpu.data, mlir.data);
    }
}
