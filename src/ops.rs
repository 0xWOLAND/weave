use std::marker::PhantomData;

use crate::{Arithmetic, ExtendKernel, LowerableGrid, LoweringError, MapKernel, ZipMapKernel};

#[derive(Clone, Copy)]
pub struct Input<T>(PhantomData<T>);

#[derive(Clone, Copy)]
pub struct Constant<T>(pub T);

#[derive(Clone, Copy)]
pub struct Neg<T>(PhantomData<T>);

#[derive(Clone, Copy)]
pub struct AddConst<T>(pub T);

#[derive(Clone, Copy)]
pub struct SubConst<T>(pub T);

#[derive(Clone, Copy)]
pub struct MulConst<T>(pub T);

#[derive(Clone, Copy)]
pub struct DivConst<T>(pub T);

#[derive(Clone, Copy)]
pub struct Extract<T>(PhantomData<T>);

#[derive(Clone, Copy)]
pub struct OffsetOr<T, const N: usize> {
    offset: [isize; N],
    fallback: T,
}

#[derive(Clone, Copy)]
pub struct AddOffsetOr<T, const N: usize> {
    offset: [isize; N],
    fallback: T,
}

macro_rules! zip_op {
    ($name:ident, $ctor:ident, $method:ident) => {
        #[derive(Clone, Copy)]
        pub struct $name<T>(PhantomData<T>);

        pub const fn $ctor<T>() -> $name<T> {
            $name(PhantomData)
        }

        impl<T: Copy, const N: usize> ZipMapKernel<T, N> for $name<T> {
            fn apply<B: Arithmetic<N, T>>(
                &self,
                backend: &mut B,
                left: B::Scalar,
                right: B::Scalar,
            ) -> Result<B::Scalar, LoweringError> {
                backend.$method(left, right)
            }
        }
    };
}

macro_rules! map_const_op {
    ($name:ident, $ctor:ident, $method:ident) => {
        pub const fn $ctor<T: Copy>(value: T) -> $name<T> {
            $name(value)
        }

        impl<T: Copy, const N: usize> MapKernel<T, N> for $name<T> {
            fn apply<B: Arithmetic<N, T>>(
                &self,
                backend: &mut B,
                value: B::Scalar,
            ) -> Result<B::Scalar, LoweringError> {
                let rhs = backend.literal(self.0)?;
                backend.$method(value, rhs)
            }
        }
    };
}

zip_op!(AddOp, add, add);
zip_op!(SubOp, sub, sub);
zip_op!(MulOp, mul, mul);
zip_op!(DivOp, div, div);

map_const_op!(AddConst, add_const, add);
map_const_op!(SubConst, sub_const, sub);
map_const_op!(MulConst, mul_const, mul);
map_const_op!(DivConst, div_const, div);

pub const fn input<T>() -> Input<T> {
    Input(PhantomData)
}

pub const fn constant<T: Copy>(value: T) -> Constant<T> {
    Constant(value)
}

pub const fn neg<T>() -> Neg<T> {
    Neg(PhantomData)
}

pub const fn extract<T>() -> Extract<T> {
    Extract(PhantomData)
}

pub const fn offset_or<T: Copy, const N: usize>(offset: [isize; N], fallback: T) -> OffsetOr<T, N> {
    OffsetOr { offset, fallback }
}

pub const fn add_offset_or<T: Copy, const N: usize>(
    offset: [isize; N],
    fallback: T,
) -> AddOffsetOr<T, N> {
    AddOffsetOr { offset, fallback }
}

impl<T: Copy, const N: usize> MapKernel<T, N> for Input<T> {
    fn apply<B: Arithmetic<N, T>>(
        &self,
        _backend: &mut B,
        value: B::Scalar,
    ) -> Result<B::Scalar, LoweringError> {
        Ok(value)
    }
}

impl<T: Copy, const N: usize> MapKernel<T, N> for Constant<T> {
    fn apply<B: Arithmetic<N, T>>(
        &self,
        backend: &mut B,
        _value: B::Scalar,
    ) -> Result<B::Scalar, LoweringError> {
        backend.literal(self.0)
    }
}

impl<T: Copy, const N: usize> MapKernel<T, N> for Neg<T> {
    fn apply<B: Arithmetic<N, T>>(
        &self,
        backend: &mut B,
        value: B::Scalar,
    ) -> Result<B::Scalar, LoweringError> {
        backend.neg(value)
    }
}

impl<T: Copy, const N: usize> ExtendKernel<T, N> for Extract<T> {
    fn apply<B: Arithmetic<N, T>, G: LowerableGrid<N, Elem = T> + ?Sized>(
        &self,
        backend: &mut B,
        view: B::View<'_, G>,
    ) -> Result<B::Scalar, LoweringError> {
        backend.extract(&view)
    }
}

impl<T: Copy, const N: usize> ExtendKernel<T, N> for OffsetOr<T, N> {
    fn apply<B: Arithmetic<N, T>, G: LowerableGrid<N, Elem = T> + ?Sized>(
        &self,
        backend: &mut B,
        view: B::View<'_, G>,
    ) -> Result<B::Scalar, LoweringError> {
        let sampled = backend.get(&view, self.offset)?;
        let fallback = backend.literal(self.fallback)?;
        backend.unwrap_or(sampled, fallback)
    }
}

impl<T: Copy, const N: usize> ExtendKernel<T, N> for AddOffsetOr<T, N> {
    fn apply<B: Arithmetic<N, T>, G: LowerableGrid<N, Elem = T> + ?Sized>(
        &self,
        backend: &mut B,
        view: B::View<'_, G>,
    ) -> Result<B::Scalar, LoweringError> {
        let here = backend.extract(&view)?;
        let sampled = backend.get(&view, self.offset)?;
        let fallback = backend.literal(self.fallback)?;
        let near = backend.unwrap_or(sampled, fallback)?;
        backend.add(here, near)
    }
}
