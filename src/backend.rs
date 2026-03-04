use std::ops::{Add, Div, Mul, Neg, Sub};

use melior::{
    Context, ExecutionEngine,
    dialect::{DialectRegistry, arith, func, memref, scf},
    ir::{
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type, Value,
        attribute::{StringAttribute, TypeAttribute},
        operation::OperationLike,
        r#type::{FunctionType, MemRefType},
    },
    pass, utility,
};

use crate::{Grid, GridLike, Representable, Shape};

#[derive(Debug, Clone, PartialEq)]
pub enum LoweringError {
    UnsupportedOperation(&'static str),
    UnsupportedElementType(&'static str),
    BackendConstruction(String),
}

pub struct CpuBackend;
pub struct MeliorBackend;

pub trait MlirElement: Copy {
    fn mlir_type<'c>(context: &'c Context) -> Type<'c>;
    fn attr<'c>(context: &'c Context, value: Self) -> Result<Attribute<'c>, LoweringError>;
}

impl MlirElement for f32 {
    fn mlir_type<'c>(context: &'c Context) -> Type<'c> {
        Type::float32(context)
    }

    fn attr<'c>(context: &'c Context, value: Self) -> Result<Attribute<'c>, LoweringError> {
        Attribute::parse(context, &format!("{value:?} : f32"))
            .ok_or_else(|| LoweringError::BackendConstruction("invalid f32 literal".into()))
    }
}

pub trait Backend<const N: usize, G>
where
    G: LowerableGrid<N>,
{
    type Output;

    fn materialize(grid: &G) -> Result<Self::Output, LoweringError>;
}

pub trait KernelBackend<const N: usize, T: Copy> {
    type Scalar: Clone;
    type Bool: Clone;
    type MaybeScalar: Clone;
    type Index: Clone;
    type MaybeIndex: Clone;
    type View<'g, G>: Clone
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g;

    fn load_source(
        &mut self,
        source_ordinal: usize,
        grid: &Grid<T, N>,
        index: Self::Index,
    ) -> Result<Self::Scalar, LoweringError>;

    fn focus(&mut self, index: Self::Index) -> Self::Index;
    fn offset(
        &mut self,
        index: Self::Index,
        delta: [isize; N],
    ) -> Result<Self::Index, LoweringError>;
    fn some_index(&mut self, index: Self::Index) -> Result<Self::MaybeIndex, LoweringError>;
    fn none_index(&mut self) -> Result<Self::MaybeIndex, LoweringError>;

    fn make_view<'g, G>(
        &mut self,
        grid: &'g G,
        shape: Shape<N>,
        index: Self::Index,
        arg_offset: usize,
    ) -> Self::View<'g, G>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g;

    fn extract<'g, G>(&mut self, view: &Self::View<'g, G>) -> Result<Self::Scalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g;

    fn get<'g, G>(
        &mut self,
        view: &Self::View<'g, G>,
        offset: [isize; N],
    ) -> Result<Self::MaybeScalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g;

    fn unwrap_or(
        &mut self,
        value: Self::MaybeScalar,
        fallback: Self::Scalar,
    ) -> Result<Self::Scalar, LoweringError>;

    fn sample_or<G>(
        &mut self,
        grid: &G,
        mapped: Self::MaybeIndex,
        fill: T,
        arg_offset: usize,
    ) -> Result<Self::Scalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized;
}

pub trait Arithmetic<const N: usize, T: Copy>: KernelBackend<N, T> {
    fn literal(&mut self, value: T) -> Result<Self::Scalar, LoweringError>;
    fn gt(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Bool, LoweringError>;
    fn select(
        &mut self,
        condition: Self::Bool,
        if_true: Self::Scalar,
        if_false: Self::Scalar,
    ) -> Result<Self::Scalar, LoweringError>;
    fn add(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError>;
    fn sub(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError>;
    fn mul(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError>;
    fn div(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError>;
    fn neg(&mut self, value: Self::Scalar) -> Result<Self::Scalar, LoweringError>;
}

pub trait MapKernel<T: Copy, const N: usize> {
    fn apply<B: Arithmetic<N, T>>(
        &self,
        backend: &mut B,
        value: B::Scalar,
    ) -> Result<B::Scalar, LoweringError>;
}

pub trait ZipMapKernel<T: Copy, const N: usize> {
    fn apply<B: Arithmetic<N, T>>(
        &self,
        backend: &mut B,
        left: B::Scalar,
        right: B::Scalar,
    ) -> Result<B::Scalar, LoweringError>;
}

pub trait ExtendKernel<T: Copy, const N: usize> {
    fn apply<B: Arithmetic<N, T>, G: LowerableGrid<N, Elem = T> + ?Sized>(
        &self,
        backend: &mut B,
        view: B::View<'_, G>,
    ) -> Result<B::Scalar, LoweringError>;
}

pub trait RemapKernel<T: Copy, const N: usize> {
    fn apply<B: KernelBackend<N, T>>(
        &self,
        backend: &mut B,
        index: B::Index,
    ) -> Result<B::MaybeIndex, LoweringError>;
}

pub trait LowerableGrid<const N: usize> {
    type Elem: Copy;

    fn shape(&self) -> Shape<N>;
    fn input_arity(&self) -> usize;
    fn append_inputs<'a>(&'a self, inputs: &mut Vec<&'a Grid<Self::Elem, N>>);
    fn append_input_types<'c>(
        &self,
        context: &'c Context,
        types: &mut Vec<Type<'c>>,
    ) -> Result<(), LoweringError>;

    fn eval_with<B: Arithmetic<N, Self::Elem>>(
        &self,
        backend: &mut B,
        index: B::Index,
        cursor: &mut usize,
    ) -> Result<B::Scalar, LoweringError>;

    fn materialize_with<B>(&self) -> Result<B::Output, LoweringError>
    where
        Self: Sized,
        B: Backend<N, Self>,
    {
        B::materialize(self)
    }

    fn map<K>(self, kernel: K) -> MappedGrid<Self, K, N>
    where
        Self: Sized,
        K: MapKernel<Self::Elem, N>,
    {
        MappedGrid { grid: self, kernel }
    }

    fn zip<H>(self, other: H) -> ZippedGrid<Self, H, N>
    where
        Self: Sized,
        H: LowerableGrid<N, Elem = Self::Elem>,
    {
        assert_eq!(self.shape(), other.shape());
        ZippedGrid {
            left: self,
            right: other,
        }
    }

    fn extend<K>(self, kernel: K) -> ExtendGrid<Self, K, N>
    where
        Self: Sized,
        K: ExtendKernel<Self::Elem, N>,
    {
        ExtendGrid { grid: self, kernel }
    }

    fn remap<K>(self, shape: [usize; N], fill: Self::Elem, kernel: K) -> RemapGrid<Self, K, N>
    where
        Self: Sized,
        K: RemapKernel<Self::Elem, N>,
    {
        RemapGrid {
            grid: self,
            shape: Shape(shape),
            fill,
            kernel,
        }
    }
}

#[derive(Clone, Copy)]
pub struct SourceGrid<'a, T: Copy, const N: usize> {
    grid: &'a Grid<T, N>,
}

impl<T: Copy, const N: usize> Grid<T, N> {
    pub fn staged(&self) -> SourceGrid<'_, T, N> {
        SourceGrid { grid: self }
    }
}

impl<'a, T: Copy + MlirElement, const N: usize> LowerableGrid<N> for SourceGrid<'a, T, N> {
    type Elem = T;

    fn shape(&self) -> Shape<N> {
        self.grid.shape
    }

    fn input_arity(&self) -> usize {
        1
    }

    fn append_inputs<'b>(&'b self, inputs: &mut Vec<&'b Grid<Self::Elem, N>>) {
        inputs.push(self.grid);
    }

    fn append_input_types<'c>(
        &self,
        context: &'c Context,
        types: &mut Vec<Type<'c>>,
    ) -> Result<(), LoweringError> {
        types.push(memref_type::<T, N>(context, self.shape()).into());
        Ok(())
    }

    fn eval_with<B: Arithmetic<N, T>>(
        &self,
        backend: &mut B,
        index: B::Index,
        cursor: &mut usize,
    ) -> Result<B::Scalar, LoweringError> {
        let source_ordinal = *cursor;
        *cursor += 1;
        backend.load_source(source_ordinal, self.grid, index)
    }
}

pub struct MappedGrid<G, K, const N: usize> {
    grid: G,
    kernel: K,
}

impl<G, K, const N: usize> LowerableGrid<N> for MappedGrid<G, K, N>
where
    G: LowerableGrid<N>,
    K: MapKernel<G::Elem, N>,
{
    type Elem = G::Elem;

    fn shape(&self) -> Shape<N> {
        self.grid.shape()
    }

    fn input_arity(&self) -> usize {
        self.grid.input_arity()
    }

    fn append_inputs<'a>(&'a self, inputs: &mut Vec<&'a Grid<Self::Elem, N>>) {
        self.grid.append_inputs(inputs);
    }

    fn append_input_types<'c>(
        &self,
        context: &'c Context,
        types: &mut Vec<Type<'c>>,
    ) -> Result<(), LoweringError> {
        self.grid.append_input_types(context, types)
    }

    fn eval_with<B: Arithmetic<N, Self::Elem>>(
        &self,
        backend: &mut B,
        index: B::Index,
        cursor: &mut usize,
    ) -> Result<B::Scalar, LoweringError> {
        let value = self.grid.eval_with(backend, index, cursor)?;
        self.kernel.apply(backend, value)
    }
}

pub struct ZippedGrid<G, H, const N: usize> {
    left: G,
    right: H,
}

impl<G, H, const N: usize> ZippedGrid<G, H, N>
where
    G: LowerableGrid<N>,
    H: LowerableGrid<N, Elem = G::Elem>,
{
    pub fn map<K>(self, kernel: K) -> ZipMappedGrid<G, H, K, N>
    where
        K: ZipMapKernel<G::Elem, N>,
    {
        ZipMappedGrid {
            left: self.left,
            right: self.right,
            kernel,
        }
    }
}

pub struct ZipMappedGrid<G, H, K, const N: usize> {
    left: G,
    right: H,
    kernel: K,
}

impl<G, H, K, const N: usize> LowerableGrid<N> for ZipMappedGrid<G, H, K, N>
where
    G: LowerableGrid<N>,
    H: LowerableGrid<N, Elem = G::Elem>,
    K: ZipMapKernel<G::Elem, N>,
{
    type Elem = G::Elem;

    fn shape(&self) -> Shape<N> {
        self.left.shape()
    }

    fn input_arity(&self) -> usize {
        self.left.input_arity() + self.right.input_arity()
    }

    fn append_inputs<'a>(&'a self, inputs: &mut Vec<&'a Grid<Self::Elem, N>>) {
        self.left.append_inputs(inputs);
        self.right.append_inputs(inputs);
    }

    fn append_input_types<'c>(
        &self,
        context: &'c Context,
        types: &mut Vec<Type<'c>>,
    ) -> Result<(), LoweringError> {
        self.left.append_input_types(context, types)?;
        self.right.append_input_types(context, types)
    }

    fn eval_with<B: Arithmetic<N, Self::Elem>>(
        &self,
        backend: &mut B,
        index: B::Index,
        cursor: &mut usize,
    ) -> Result<B::Scalar, LoweringError> {
        let left = self.left.eval_with(backend, index.clone(), cursor)?;
        let right = self.right.eval_with(backend, index, cursor)?;
        self.kernel.apply(backend, left, right)
    }
}

pub struct ExtendGrid<G, K, const N: usize> {
    grid: G,
    kernel: K,
}

impl<G, K, const N: usize> LowerableGrid<N> for ExtendGrid<G, K, N>
where
    G: LowerableGrid<N>,
    K: ExtendKernel<G::Elem, N>,
{
    type Elem = G::Elem;

    fn shape(&self) -> Shape<N> {
        self.grid.shape()
    }

    fn input_arity(&self) -> usize {
        self.grid.input_arity()
    }

    fn append_inputs<'a>(&'a self, inputs: &mut Vec<&'a Grid<Self::Elem, N>>) {
        self.grid.append_inputs(inputs);
    }

    fn append_input_types<'c>(
        &self,
        context: &'c Context,
        types: &mut Vec<Type<'c>>,
    ) -> Result<(), LoweringError> {
        self.grid.append_input_types(context, types)
    }

    fn eval_with<B: Arithmetic<N, Self::Elem>>(
        &self,
        backend: &mut B,
        index: B::Index,
        cursor: &mut usize,
    ) -> Result<B::Scalar, LoweringError> {
        let arg_offset = *cursor;
        let focus = backend.focus(index);
        let view = backend.make_view(&self.grid, self.shape(), focus, arg_offset);
        let value = self.kernel.apply(backend, view)?;
        *cursor = arg_offset + self.grid.input_arity();
        Ok(value)
    }
}

pub struct RemapGrid<G, K, const N: usize>
where
    G: LowerableGrid<N>,
{
    grid: G,
    shape: Shape<N>,
    fill: G::Elem,
    kernel: K,
}

impl<G, K, const N: usize> LowerableGrid<N> for RemapGrid<G, K, N>
where
    G: LowerableGrid<N>,
    K: RemapKernel<G::Elem, N>,
{
    type Elem = G::Elem;

    fn shape(&self) -> Shape<N> {
        self.shape
    }

    fn input_arity(&self) -> usize {
        self.grid.input_arity()
    }

    fn append_inputs<'a>(&'a self, inputs: &mut Vec<&'a Grid<Self::Elem, N>>) {
        self.grid.append_inputs(inputs);
    }

    fn append_input_types<'c>(
        &self,
        context: &'c Context,
        types: &mut Vec<Type<'c>>,
    ) -> Result<(), LoweringError> {
        self.grid.append_input_types(context, types)
    }

    fn eval_with<B: Arithmetic<N, Self::Elem>>(
        &self,
        backend: &mut B,
        index: B::Index,
        cursor: &mut usize,
    ) -> Result<B::Scalar, LoweringError> {
        let arg_offset = *cursor;
        let mapped = self.kernel.apply(backend, index)?;
        let value = backend.sample_or(&self.grid, mapped, self.fill, arg_offset)?;
        *cursor = arg_offset + self.grid.input_arity();
        Ok(value)
    }
}

impl<const N: usize, G> Backend<N, G> for CpuBackend
where
    G: LowerableGrid<N>,
    G::Elem: Copy
        + PartialOrd
        + Add<Output = G::Elem>
        + Sub<Output = G::Elem>
        + Mul<Output = G::Elem>
        + Div<Output = G::Elem>
        + Neg<Output = G::Elem>,
{
    type Output = Grid<G::Elem, N>;

    fn materialize(grid: &G) -> Result<Self::Output, LoweringError> {
        let mut backend = CpuEval;
        Ok(Grid::tabulate(grid.shape().0, |index| {
            let mut cursor = 0;
            grid.eval_with(&mut backend, index, &mut cursor)
                .expect("cpu evaluation is infallible")
        }))
    }
}

impl<const N: usize, G> Backend<N, G> for MeliorBackend
where
    G: LowerableGrid<N, Elem = f32>,
    G::Elem: MlirElement,
{
    type Output = Grid<f32, N>;

    fn materialize(grid: &G) -> Result<Self::Output, LoweringError> {
        if N == 0 {
            return Err(LoweringError::UnsupportedOperation(
                "zero-dimensional lowering is not implemented",
            ));
        }

        let context = Context::new();
        let registry = DialectRegistry::new();
        utility::register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();

        let location = Location::unknown(&context);
        let mut module = Module::new(location);
        let output_type = memref_type::<G::Elem, N>(&context, grid.shape());

        let mut input_types = Vec::new();
        grid.append_input_types(&context, &mut input_types)?;
        let mut function_input_types = input_types.clone();
        function_input_types.push(output_type.into());

        let function_block = Block::new(
            &function_input_types
                .iter()
                .copied()
                .map(|ty| (ty, location))
                .collect::<Vec<_>>(),
        );
        let arguments = (0..input_types.len())
            .map(|i| {
                function_block.argument(i).map_err(|_| {
                    LoweringError::BackendConstruction("missing function argument".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(|value| value_to_context(value.into()))
            .collect::<Vec<_>>();

        let output = value_to_context(
            function_block
                .argument(input_types.len())
                .map_err(|_| {
                    LoweringError::BackendConstruction("missing output memref argument".into())
                })?
                .into(),
        );

        emit_loop_nest(
            grid,
            &function_block,
            &context,
            location,
            output,
            &arguments,
            0,
            &mut Vec::with_capacity(N),
        )?;
        function_block.append_operation(func::r#return(&[], location));

        let region = Region::new();
        region.append_block(function_block);
        module.body().append_operation(func::func(
            &context,
            StringAttribute::new(&context, "materialize"),
            TypeAttribute::new(FunctionType::new(&context, &function_input_types, &[]).into()),
            region,
            &[(
                Identifier::new(&context, "llvm.emit_c_interface"),
                Attribute::parse(&context, "unit").ok_or_else(|| {
                    LoweringError::BackendConstruction(
                        "failed to create llvm.emit_c_interface attribute".into(),
                    )
                })?,
            )],
            location,
        ));

        if !module.as_operation().verify() {
            return Err(LoweringError::BackendConstruction(
                "generated MLIR did not verify".into(),
            ));
        }

        utility::register_all_passes();
        utility::register_all_llvm_translations(&context);

        let pass_manager = pass::PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_to_llvm());
        pass_manager.add_pass(pass::conversion::create_finalize_mem_ref_to_llvm());
        pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());

        pass_manager.run(&mut module).map_err(|err| {
            LoweringError::BackendConstruction(format!("failed to lower module to LLVM: {err}"))
        })?;

        let engine = ExecutionEngine::new(&module, 2, &[], false);

        let mut inputs = Vec::with_capacity(grid.input_arity());
        grid.append_inputs(&mut inputs);
        let mut output_grid = Grid::new(grid.shape().0, vec![0.0; grid.shape().size()]);
        let mut input_descriptors = inputs
            .iter()
            .map(|input| memref_descriptor(*input))
            .collect::<Vec<_>>();
        let mut output_descriptor = memref_descriptor_mut(&mut output_grid);
        let ciface = engine.lookup("_mlir_ciface_materialize");
        if ciface.is_null() {
            return Err(LoweringError::BackendConstruction(
                "missing _mlir_ciface_materialize symbol".into(),
            ));
        }
        unsafe { invoke_memref_ciface(ciface, &mut input_descriptors, &mut output_descriptor)? };

        Ok(output_grid)
    }
}

fn emit_loop_nest<'c, G, const N: usize>(
    grid: &G,
    block: &Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    output: Value<'c, 'c>,
    arguments: &[Value<'c, 'c>],
    depth: usize,
    indices: &mut Vec<Value<'c, 'c>>,
) -> Result<(), LoweringError>
where
    G: LowerableGrid<N, Elem = f32>,
    G::Elem: MlirElement,
{
    let start = lower_index_constant(block, context, location, 0)?;
    let end = lower_index_constant(block, context, location, grid.shape().0[depth] as i64)?;
    let step = lower_index_constant(block, context, location, 1)?;

    let loop_block = Block::new(&[(Type::index(context), location)]);
    let index = value_to_context(
        loop_block
            .argument(0)
            .map_err(|_| LoweringError::BackendConstruction("missing induction variable".into()))?
            .into(),
    );
    indices.push(index);

    if depth + 1 == N {
        let mut backend = MlirEval {
            block: &loop_block,
            context,
            location,
            arguments,
            _marker: std::marker::PhantomData,
        };
        let mut cursor = 0;
        let value = grid.eval_with(&mut backend, MlirIndex::new(indices.clone()), &mut cursor)?;
        loop_block.append_operation(memref::store(value, output, indices.as_slice(), location));
    } else {
        emit_loop_nest(
            grid,
            &loop_block,
            context,
            location,
            output,
            arguments,
            depth + 1,
            indices,
        )?;
    }

    indices.pop();
    loop_block.append_operation(scf::r#yield(&[], location));

    let region = Region::new();
    region.append_block(loop_block);
    block.append_operation(scf::r#for(start, end, step, region, location));
    Ok(())
}

struct CpuEval;

struct CpuView<'g, G, T: Copy, const N: usize>
where
    G: LowerableGrid<N, Elem = T> + ?Sized,
{
    grid: &'g G,
    shape: Shape<N>,
    index: [usize; N],
    arg_offset: usize,
}

impl<G, T: Copy, const N: usize> Clone for CpuView<'_, G, T, N>
where
    G: LowerableGrid<N, Elem = T> + ?Sized,
{
    fn clone(&self) -> Self {
        Self {
            grid: self.grid,
            shape: self.shape,
            index: self.index,
            arg_offset: self.arg_offset,
        }
    }
}

impl<T, const N: usize> KernelBackend<N, T> for CpuEval
where
    T: Copy
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    type Scalar = T;
    type Bool = bool;
    type MaybeScalar = Option<T>;
    type Index = [usize; N];
    type MaybeIndex = Option<[usize; N]>;
    type View<'g, G>
        = CpuView<'g, G, T, N>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g;

    fn load_source(
        &mut self,
        _source_ordinal: usize,
        grid: &Grid<T, N>,
        index: Self::Index,
    ) -> Result<Self::Scalar, LoweringError> {
        Ok(grid.at(index))
    }

    fn focus(&mut self, index: Self::Index) -> Self::Index {
        index
    }

    fn offset(
        &mut self,
        index: Self::Index,
        delta: [isize; N],
    ) -> Result<Self::Index, LoweringError> {
        let mut out = [0; N];
        for axis in 0..N {
            let coord = index[axis] as isize + delta[axis];
            if coord < 0 {
                return Err(LoweringError::UnsupportedOperation("negative index"));
            }
            out[axis] = coord as usize;
        }
        Ok(out)
    }

    fn some_index(&mut self, index: Self::Index) -> Result<Self::MaybeIndex, LoweringError> {
        Ok(Some(index))
    }

    fn none_index(&mut self) -> Result<Self::MaybeIndex, LoweringError> {
        Ok(None)
    }

    fn make_view<'g, G>(
        &mut self,
        grid: &'g G,
        shape: Shape<N>,
        index: Self::Index,
        arg_offset: usize,
    ) -> Self::View<'g, G>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g,
    {
        CpuView {
            grid,
            shape,
            index,
            arg_offset,
        }
    }

    fn extract<'g, G>(&mut self, view: &Self::View<'g, G>) -> Result<Self::Scalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g,
    {
        let mut cursor = view.arg_offset;
        view.grid.eval_with(self, view.index, &mut cursor)
    }

    fn get<'g, G>(
        &mut self,
        view: &Self::View<'g, G>,
        offset: [isize; N],
    ) -> Result<Self::MaybeScalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized + 'g,
    {
        let mut index = [0; N];
        for axis in 0..N {
            let coord = view.index[axis] as isize + offset[axis];
            if coord < 0 || coord >= view.shape.0[axis] as isize {
                return Ok(None);
            }
            index[axis] = coord as usize;
        }
        let mut cursor = view.arg_offset;
        view.grid.eval_with(self, index, &mut cursor).map(Some)
    }

    fn unwrap_or(
        &mut self,
        value: Self::MaybeScalar,
        fallback: Self::Scalar,
    ) -> Result<Self::Scalar, LoweringError> {
        Ok(value.unwrap_or(fallback))
    }

    fn sample_or<G>(
        &mut self,
        grid: &G,
        mapped: Self::MaybeIndex,
        fill: T,
        arg_offset: usize,
    ) -> Result<Self::Scalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = T> + ?Sized,
    {
        let Some(index) = mapped else { return Ok(fill) };
        if !in_bounds(grid.shape(), index) {
            return Ok(fill);
        }
        let mut cursor = arg_offset;
        grid.eval_with(self, index, &mut cursor)
    }
}

impl<T, const N: usize> Arithmetic<N, T> for CpuEval
where
    T: Copy
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    fn literal(&mut self, value: T) -> Result<Self::Scalar, LoweringError> {
        Ok(value)
    }
    fn gt(&mut self, lhs: T, rhs: T) -> Result<Self::Bool, LoweringError> {
        Ok(lhs > rhs)
    }
    fn select(
        &mut self,
        condition: Self::Bool,
        if_true: Self::Scalar,
        if_false: Self::Scalar,
    ) -> Result<Self::Scalar, LoweringError> {
        Ok(if condition { if_true } else { if_false })
    }
    fn add(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError> {
        Ok(lhs + rhs)
    }
    fn sub(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError> {
        Ok(lhs - rhs)
    }
    fn mul(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError> {
        Ok(lhs * rhs)
    }
    fn div(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError> {
        Ok(lhs / rhs)
    }
    fn neg(&mut self, value: T) -> Result<T, LoweringError> {
        Ok(-value)
    }
}

struct MlirEval<'b, 'c, T: MlirElement, const N: usize> {
    block: &'b Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    arguments: &'b [Value<'c, 'c>],
    _marker: std::marker::PhantomData<T>,
}

#[derive(Clone)]
pub struct MlirIndex<'c, const N: usize> {
    coords: [Value<'c, 'c>; N],
}

#[derive(Clone)]
pub struct MlirMaybeIndex<'c, const N: usize> {
    valid: Value<'c, 'c>,
    index: MlirIndex<'c, N>,
}

#[derive(Clone)]
pub struct MlirMaybeScalar<'c> {
    valid: Value<'c, 'c>,
    value: Value<'c, 'c>,
}

pub struct MlirView<'g, 'c, G, T: MlirElement, const N: usize>
where
    G: LowerableGrid<N, Elem = T> + ?Sized,
{
    grid: &'g G,
    shape: Shape<N>,
    index: MlirIndex<'c, N>,
    arg_offset: usize,
}

impl<'c, G, T: MlirElement, const N: usize> Clone for MlirView<'_, 'c, G, T, N>
where
    G: LowerableGrid<N, Elem = T> + ?Sized,
{
    fn clone(&self) -> Self {
        Self {
            grid: self.grid,
            shape: self.shape,
            index: self.index.clone(),
            arg_offset: self.arg_offset,
        }
    }
}

impl<'c, const N: usize> MlirIndex<'c, N> {
    fn new(values: Vec<Value<'c, 'c>>) -> Self {
        Self {
            coords: values.try_into().ok().expect("exactly N indices"),
        }
    }

    fn as_slice(&self) -> &[Value<'c, 'c>] {
        &self.coords
    }
}

impl<'b, 'c, const N: usize> KernelBackend<N, f32> for MlirEval<'b, 'c, f32, N> {
    type Scalar = Value<'c, 'c>;
    type Bool = Value<'c, 'c>;
    type MaybeScalar = MlirMaybeScalar<'c>;
    type Index = MlirIndex<'c, N>;
    type MaybeIndex = MlirMaybeIndex<'c, N>;
    type View<'g, G>
        = MlirView<'g, 'c, G, f32, N>
    where
        G: LowerableGrid<N, Elem = f32> + ?Sized + 'g;

    fn load_source(
        &mut self,
        source_ordinal: usize,
        _grid: &Grid<f32, N>,
        index: Self::Index,
    ) -> Result<Self::Scalar, LoweringError> {
        let memref = *self.arguments.get(source_ordinal).ok_or_else(|| {
            LoweringError::BackendConstruction("missing source memref argument".into())
        })?;
        let op = self
            .block
            .append_operation(memref::load(memref, index.as_slice(), self.location));
        op.result(0)
            .map(|value| value_to_context(value.into()))
            .map_err(|_| LoweringError::BackendConstruction("memref.load returned no value".into()))
    }

    fn focus(&mut self, index: Self::Index) -> Self::Index {
        index
    }

    fn offset(
        &mut self,
        index: Self::Index,
        delta: [isize; N],
    ) -> Result<Self::Index, LoweringError> {
        let mut values = Vec::with_capacity(N);
        for axis in 0..N {
            let delta =
                lower_index_constant(self.block, self.context, self.location, delta[axis] as i64)?;
            let op =
                self.block
                    .append_operation(arith::addi(index.coords[axis], delta, self.location));
            values.push(value_to_context(op.result(0).unwrap().into()));
        }
        Ok(MlirIndex::new(values))
    }

    fn some_index(&mut self, index: Self::Index) -> Result<Self::MaybeIndex, LoweringError> {
        Ok(MlirMaybeIndex {
            valid: lower_bool_constant(self.block, self.context, self.location, true)?,
            index,
        })
    }

    fn none_index(&mut self) -> Result<Self::MaybeIndex, LoweringError> {
        Ok(MlirMaybeIndex {
            valid: lower_bool_constant(self.block, self.context, self.location, false)?,
            index: MlirIndex::new(vec![
                lower_index_constant(
                    self.block,
                    self.context,
                    self.location,
                    0
                )?;
                N
            ]),
        })
    }

    fn make_view<'g, G>(
        &mut self,
        grid: &'g G,
        shape: Shape<N>,
        index: Self::Index,
        arg_offset: usize,
    ) -> Self::View<'g, G>
    where
        G: LowerableGrid<N, Elem = f32> + ?Sized + 'g,
    {
        MlirView {
            grid,
            shape,
            index,
            arg_offset,
        }
    }

    fn extract<'g, G>(&mut self, view: &Self::View<'g, G>) -> Result<Self::Scalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = f32> + ?Sized + 'g,
    {
        let mut cursor = view.arg_offset;
        view.grid.eval_with(self, view.index.clone(), &mut cursor)
    }

    fn get<'g, G>(
        &mut self,
        view: &Self::View<'g, G>,
        offset: [isize; N],
    ) -> Result<Self::MaybeScalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = f32> + ?Sized + 'g,
    {
        let shifted = self.offset(view.index.clone(), offset)?;
        let valid = mlir_in_bounds(
            self.block,
            self.context,
            self.location,
            &shifted,
            view.shape,
        )?;
        let safe = clamp_mlir_index(
            self.block,
            self.context,
            self.location,
            &shifted,
            view.shape,
        )?;
        let mut cursor = view.arg_offset;
        let value = view.grid.eval_with(self, safe, &mut cursor)?;
        Ok(MlirMaybeScalar { valid, value })
    }

    fn unwrap_or(
        &mut self,
        value: Self::MaybeScalar,
        fallback: Self::Scalar,
    ) -> Result<Self::Scalar, LoweringError> {
        let op = self.block.append_operation(arith::select(
            value.valid,
            value.value,
            fallback,
            self.location,
        ));
        op.result(0)
            .map(|value| value_to_context(value.into()))
            .map_err(|_| {
                LoweringError::BackendConstruction("arith.select returned no value".into())
            })
    }

    fn sample_or<G>(
        &mut self,
        grid: &G,
        mapped: Self::MaybeIndex,
        fill: f32,
        arg_offset: usize,
    ) -> Result<Self::Scalar, LoweringError>
    where
        G: LowerableGrid<N, Elem = f32> + ?Sized,
    {
        let in_bounds = mlir_in_bounds(
            self.block,
            self.context,
            self.location,
            &mapped.index,
            grid.shape(),
        )?;
        let valid = and_bool(self.block, self.location, mapped.valid, in_bounds)?;
        let safe = clamp_mlir_index(
            self.block,
            self.context,
            self.location,
            &mapped.index,
            grid.shape(),
        )?;
        let mut cursor = arg_offset;
        let value = grid.eval_with(self, safe, &mut cursor)?;
        let fill = lower_scalar_constant::<f32>(self.block, self.context, self.location, fill)?;
        let op = self
            .block
            .append_operation(arith::select(valid, value, fill, self.location));
        op.result(0)
            .map(|value| value_to_context(value.into()))
            .map_err(|_| {
                LoweringError::BackendConstruction("arith.select returned no value".into())
            })
    }
}

impl<'b, 'c, const N: usize> Arithmetic<N, f32> for MlirEval<'b, 'c, f32, N> {
    fn literal(&mut self, value: f32) -> Result<Self::Scalar, LoweringError> {
        lower_scalar_constant::<f32>(self.block, self.context, self.location, value)
    }
    fn gt(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Bool, LoweringError> {
        let op = self.block.append_operation(arith::cmpf(
            self.context,
            arith::CmpfPredicate::Ogt,
            lhs,
            rhs,
            self.location,
        ));
        op.result(0)
            .map(|value| value_to_context(value.into()))
            .map_err(|_| LoweringError::BackendConstruction("arith.cmpf returned no value".into()))
    }
    fn select(
        &mut self,
        condition: Self::Bool,
        if_true: Self::Scalar,
        if_false: Self::Scalar,
    ) -> Result<Self::Scalar, LoweringError> {
        let op =
            self.block
                .append_operation(arith::select(condition, if_true, if_false, self.location));
        op.result(0)
            .map(|value| value_to_context(value.into()))
            .map_err(|_| {
                LoweringError::BackendConstruction("arith.select returned no value".into())
            })
    }
    fn add(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError> {
        lower_float_bin(
            self.block,
            self.location,
            lhs,
            rhs,
            arith::addf,
            "arith.addf",
        )
    }
    fn sub(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError> {
        lower_float_bin(
            self.block,
            self.location,
            lhs,
            rhs,
            arith::subf,
            "arith.subf",
        )
    }
    fn mul(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError> {
        lower_float_bin(
            self.block,
            self.location,
            lhs,
            rhs,
            arith::mulf,
            "arith.mulf",
        )
    }
    fn div(&mut self, lhs: Self::Scalar, rhs: Self::Scalar) -> Result<Self::Scalar, LoweringError> {
        lower_float_bin(
            self.block,
            self.location,
            lhs,
            rhs,
            arith::divf,
            "arith.divf",
        )
    }
    fn neg(&mut self, value: Self::Scalar) -> Result<Self::Scalar, LoweringError> {
        let op = self
            .block
            .append_operation(arith::negf(value, self.location));
        op.result(0)
            .map(|value| value_to_context(value.into()))
            .map_err(|_| LoweringError::BackendConstruction("arith.negf returned no value".into()))
    }
}

fn memref_type<'c, T: MlirElement, const N: usize>(
    context: &'c Context,
    shape: Shape<N>,
) -> MemRefType<'c> {
    let dims = shape.0.map(|extent| extent as i64);
    MemRefType::new(T::mlir_type(context), &dims, None, None)
}

fn lower_scalar_constant<'c, T: MlirElement>(
    block: &Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    value: T,
) -> Result<Value<'c, 'c>, LoweringError> {
    let op = block.append_operation(arith::constant(context, T::attr(context, value)?, location));
    op.result(0)
        .map(|value| value_to_context(value.into()))
        .map_err(|_| LoweringError::BackendConstruction("arith.constant returned no value".into()))
}

fn lower_bool_constant<'c>(
    block: &Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    value: bool,
) -> Result<Value<'c, 'c>, LoweringError> {
    let lit = if value { "true" } else { "false" };
    let op = block.append_operation(arith::constant(
        context,
        Attribute::parse(context, lit)
            .ok_or_else(|| LoweringError::BackendConstruction("invalid bool literal".into()))?,
        location,
    ));
    op.result(0)
        .map(|value| value_to_context(value.into()))
        .map_err(|_| LoweringError::BackendConstruction("arith.constant returned no value".into()))
}

fn lower_index_constant<'c>(
    block: &Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    value: i64,
) -> Result<Value<'c, 'c>, LoweringError> {
    let op = block.append_operation(index_constant(context, value, location));
    op.result(0)
        .map(|value| value_to_context(value.into()))
        .map_err(|_| LoweringError::BackendConstruction("arith.constant returned no value".into()))
}

fn lower_float_bin<'c>(
    block: &Block<'c>,
    location: Location<'c>,
    lhs: Value<'c, 'c>,
    rhs: Value<'c, 'c>,
    op: fn(Value<'c, '_>, Value<'c, '_>, Location<'c>) -> melior::ir::Operation<'c>,
    name: &'static str,
) -> Result<Value<'c, 'c>, LoweringError> {
    let op = block.append_operation(op(lhs, rhs, location));
    op.result(0)
        .map(|value| value_to_context(value.into()))
        .map_err(|_| LoweringError::BackendConstruction(format!("{name} returned no value")))
}

fn and_bool<'c>(
    block: &Block<'c>,
    location: Location<'c>,
    lhs: Value<'c, 'c>,
    rhs: Value<'c, 'c>,
) -> Result<Value<'c, 'c>, LoweringError> {
    let op = block.append_operation(arith::andi(lhs, rhs, location));
    op.result(0)
        .map(|value| value_to_context(value.into()))
        .map_err(|_| LoweringError::BackendConstruction("arith.andi returned no value".into()))
}

fn mlir_in_bounds<'c, const N: usize>(
    block: &Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    index: &MlirIndex<'c, N>,
    shape: Shape<N>,
) -> Result<Value<'c, 'c>, LoweringError> {
    let zero = lower_index_constant(block, context, location, 0)?;
    let mut valid = lower_bool_constant(block, context, location, true)?;
    for axis in 0..N {
        let non_negative = block.append_operation(arith::cmpi(
            context,
            arith::CmpiPredicate::Sge,
            index.coords[axis],
            zero,
            location,
        ));
        valid = and_bool(
            block,
            location,
            valid,
            value_to_context(non_negative.result(0).unwrap().into()),
        )?;
        let upper = lower_index_constant(block, context, location, shape.0[axis] as i64)?;
        let below_upper = block.append_operation(arith::cmpi(
            context,
            arith::CmpiPredicate::Slt,
            index.coords[axis],
            upper,
            location,
        ));
        valid = and_bool(
            block,
            location,
            valid,
            value_to_context(below_upper.result(0).unwrap().into()),
        )?;
    }
    Ok(valid)
}

fn clamp_mlir_index<'c, const N: usize>(
    block: &Block<'c>,
    context: &'c Context,
    location: Location<'c>,
    index: &MlirIndex<'c, N>,
    shape: Shape<N>,
) -> Result<MlirIndex<'c, N>, LoweringError> {
    let zero = lower_index_constant(block, context, location, 0)?;
    let mut values = Vec::with_capacity(N);
    for axis in 0..N {
        let below_zero = block.append_operation(arith::cmpi(
            context,
            arith::CmpiPredicate::Slt,
            index.coords[axis],
            zero,
            location,
        ));
        let low = block.append_operation(arith::select(
            below_zero.result(0).unwrap().into(),
            zero,
            index.coords[axis],
            location,
        ));
        let max = lower_index_constant(
            block,
            context,
            location,
            shape.0[axis].saturating_sub(1) as i64,
        )?;
        let too_high = block.append_operation(arith::cmpi(
            context,
            arith::CmpiPredicate::Sgt,
            value_to_context(low.result(0).unwrap().into()),
            max,
            location,
        ));
        let high = block.append_operation(arith::select(
            too_high.result(0).unwrap().into(),
            max,
            low.result(0).unwrap().into(),
            location,
        ));
        values.push(value_to_context(high.result(0).unwrap().into()));
    }
    Ok(MlirIndex::new(values))
}

fn index_constant<'c>(
    context: &'c Context,
    value: i64,
    location: Location<'c>,
) -> melior::ir::Operation<'c> {
    arith::constant(
        context,
        Attribute::parse(context, &format!("{value} : index")).expect("valid index literal"),
        location,
    )
}

#[repr(C)]
struct StridedMemRef<T, const N: usize> {
    base_ptr: *mut T,
    data: *mut T,
    offset: i64,
    sizes: [i64; N],
    strides: [i64; N],
}

fn row_major_strides<const N: usize>(shape: Shape<N>) -> [i64; N] {
    let mut strides = [0; N];
    let mut stride = 1_i64;

    for axis in (0..N).rev() {
        strides[axis] = stride;
        stride *= shape.0[axis] as i64;
    }

    strides
}

fn memref_descriptor<const N: usize>(grid: &Grid<f32, N>) -> StridedMemRef<f32, N> {
    StridedMemRef {
        base_ptr: grid.data.as_ptr() as *mut f32,
        data: grid.data.as_ptr() as *mut f32,
        offset: 0,
        sizes: grid.shape.0.map(|extent| extent as i64),
        strides: row_major_strides(grid.shape),
    }
}

fn memref_descriptor_mut<const N: usize>(grid: &mut Grid<f32, N>) -> StridedMemRef<f32, N> {
    StridedMemRef {
        base_ptr: grid.data.as_mut_ptr(),
        data: grid.data.as_mut_ptr(),
        offset: 0,
        sizes: grid.shape.0.map(|extent| extent as i64),
        strides: row_major_strides(grid.shape),
    }
}

fn value_to_context<'c>(value: Value<'c, '_>) -> Value<'c, 'c> {
    unsafe { std::mem::transmute::<Value<'c, '_>, Value<'c, 'c>>(value) }
}

fn in_bounds<const N: usize>(shape: Shape<N>, index: [usize; N]) -> bool {
    (0..N).all(|axis| index[axis] < shape.0[axis])
}

unsafe fn invoke_memref_ciface<const N: usize>(
    function: *mut (),
    inputs: &mut [StridedMemRef<f32, N>],
    output: &mut StridedMemRef<f32, N>,
) -> Result<(), LoweringError> {
    type MemRefArg<const N: usize> = *mut StridedMemRef<f32, N>;

    match inputs.len() {
        1 => {
            let function: unsafe extern "C" fn(MemRefArg<N>, MemRefArg<N>) =
                unsafe { std::mem::transmute(function) };
            unsafe { function(&mut inputs[0], output) };
            Ok(())
        }
        2 => {
            let function: unsafe extern "C" fn(MemRefArg<N>, MemRefArg<N>, MemRefArg<N>) =
                unsafe { std::mem::transmute(function) };
            unsafe { function(&mut inputs[0], &mut inputs[1], output) };
            Ok(())
        }
        3 => {
            let function: unsafe extern "C" fn(
                MemRefArg<N>,
                MemRefArg<N>,
                MemRefArg<N>,
                MemRefArg<N>,
            ) = unsafe { std::mem::transmute(function) };
            unsafe { function(&mut inputs[0], &mut inputs[1], &mut inputs[2], output) };
            Ok(())
        }
        4 => {
            let function: unsafe extern "C" fn(
                MemRefArg<N>,
                MemRefArg<N>,
                MemRefArg<N>,
                MemRefArg<N>,
                MemRefArg<N>,
            ) = unsafe { std::mem::transmute(function) };
            unsafe {
                function(
                    &mut inputs[0],
                    &mut inputs[1],
                    &mut inputs[2],
                    &mut inputs[3],
                    output,
                )
            };
            Ok(())
        }
        arity => Err(LoweringError::UnsupportedOperation(match arity {
            0 => "melior backend requires at least one input grid",
            _ => "melior backend supports up to four input grids",
        })),
    }
}
