use std::ops::{Add, Div, Mul, Neg, Sub};

use melior::{
    Context, ExecutionEngine, pass, utility,
    dialect::DialectRegistry,
    ir::{Module, operation::OperationLike},
};

use crate::{Grid, Image, Shape};

#[derive(Debug, Clone, PartialEq)]
pub enum LoweringError {
    UnsupportedOperation(&'static str),
    BackendConstruction(String),
}

pub struct CpuBackend;
pub struct MeliorBackend;

#[doc(hidden)]
pub trait EvalContext<T: Copy, const N: usize> {
    fn literal(&mut self, value: T) -> Result<T, LoweringError> {
        Ok(value)
    }

    fn load(&mut self, grid: &Grid<T, N>, index: [usize; N]) -> Result<T, LoweringError>;
    fn gt(&mut self, lhs: T, rhs: T) -> Result<bool, LoweringError>;
    fn select(&mut self, condition: bool, if_true: T, if_false: T) -> Result<T, LoweringError>;
    fn add(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError>;
    fn sub(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError>;
    fn mul(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError>;
    fn div(&mut self, lhs: T, rhs: T) -> Result<T, LoweringError>;
    fn neg(&mut self, value: T) -> Result<T, LoweringError>;
}

#[derive(Default)]
pub struct PureEval;

impl<T, const N: usize> EvalContext<T, N> for PureEval
where
    T: Copy
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    fn load(&mut self, grid: &Grid<T, N>, index: [usize; N]) -> Result<T, LoweringError> {
        Ok(grid.at(index))
    }

    fn gt(&mut self, lhs: T, rhs: T) -> Result<bool, LoweringError> {
        Ok(lhs > rhs)
    }

    fn select(&mut self, condition: bool, if_true: T, if_false: T) -> Result<T, LoweringError> {
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

pub trait Backend<const N: usize, T: Copy> {
    type Output;

    fn materialize<I>(image: &I) -> Result<Self::Output, LoweringError>
    where
        I: Image<N, Pixel = T> + ?Sized;
}

impl<const N: usize, T> Backend<N, T> for CpuBackend
where
    T: Copy
        + PartialOrd
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + Neg<Output = T>,
{
    type Output = Grid<T, N>;

    fn materialize<I>(image: &I) -> Result<Self::Output, LoweringError>
    where
        I: Image<N, Pixel = T> + ?Sized,
    {
        Ok(Grid::tabulate(image.shape().0, |index| image.sample(index)))
    }
}

impl<const N: usize> Backend<N, f32> for MeliorBackend {
    type Output = Grid<f32, N>;

    fn materialize<I>(image: &I) -> Result<Self::Output, LoweringError>
    where
        I: Image<N, Pixel = f32> + ?Sized,
    {
        let input = image.source_grid().ok_or(LoweringError::UnsupportedOperation(
            "melior backend currently supports exactly one source grid",
        ))?;

        let shape = image.shape();
        let module_text = build_module_text(image, shape);

        let context = Context::new();
        let registry = DialectRegistry::new();
        utility::register_all_dialects(&registry);
        context.append_dialect_registry(&registry);
        context.load_all_available_dialects();
        utility::register_all_llvm_translations(&context);

        let mut module = Module::parse(&context, &module_text).ok_or_else(|| {
            LoweringError::BackendConstruction("failed to parse generated MLIR module".into())
        })?;

        if !module.as_operation().verify() {
            return Err(LoweringError::BackendConstruction(
                "generated MLIR did not verify".into(),
            ));
        }

        utility::register_all_passes();
        let pass_manager = pass::PassManager::new(&context);
        pass_manager.add_pass(pass::conversion::create_scf_to_control_flow());
        pass_manager.add_pass(pass::conversion::create_to_llvm());

        pass_manager.run(&mut module).map_err(|_| {
            LoweringError::BackendConstruction("failed to lower MLIR to LLVM".into())
        })?;

        let engine = ExecutionEngine::new(&module, 2, &[], false);

        let mut output = Grid::new(shape.0, vec![0.0_f32; shape.size()]);
        let mut input_desc = memref_descriptor(input);
        let mut output_desc = memref_descriptor_mut(&mut output);
        let mut input_ptr = &mut input_desc as *mut StridedMemRef<f32, N>;
        let mut output_ptr = &mut output_desc as *mut StridedMemRef<f32, N>;

        unsafe {
            engine
                .invoke_packed(
                    "materialize",
                    &mut [
                        &mut input_ptr as *mut *mut StridedMemRef<f32, N> as *mut (),
                        &mut output_ptr as *mut *mut StridedMemRef<f32, N> as *mut (),
                    ],
                )
                .map_err(|_| {
                    LoweringError::BackendConstruction("failed to execute generated MLIR".into())
                })?;
        }

        Ok(output)
    }
}

fn build_module_text<I, const N: usize>(image: &I, shape: Shape<N>) -> String
where
    I: Image<N, Pixel = f32> + ?Sized,
{
    let memref_ty = memref_type(shape);
    let mut body = String::new();
    body.push_str("    %c0 = arith.constant 0 : index\n");
    body.push_str("    %c1 = arith.constant 1 : index\n");

    for (axis, extent) in shape.0.iter().enumerate() {
        body.push_str(&format!(
            "    %c{axis}_max = arith.constant {extent} : index\n"
        ));
    }

    let mut next_id = 0;
    emit_loops(image, shape, 0, &mut Vec::with_capacity(N), &mut next_id, &mut body, &memref_ty)
        .expect("mlir string generation is infallible for lowerable images");

    format!(
        "module {{
  func.func @materialize(%input: {memref_ty}, %output: {memref_ty}) attributes {{ llvm.emit_c_interface }} {{
{body}    return
  }}
}}
"
    )
}

fn emit_loops<I, const N: usize>(
    image: &I,
    shape: Shape<N>,
    depth: usize,
    indices: &mut Vec<String>,
    next_id: &mut usize,
    body: &mut String,
    memref_ty: &str,
) -> Result<(), LoweringError>
where
    I: Image<N, Pixel = f32> + ?Sized,
{
    if depth == N {
        let value = image.lower_mlir(indices, next_id, body)?;
        let indices_text = indices.join(", ");
        body.push_str(&format!(
            "          memref.store {value}, %output[{indices_text}] : {memref_ty}\n"
        ));
        return Ok(());
    }

    let indent = "    ".repeat(depth + 1);
    let index_name = format!("%i{depth}");
    body.push_str(&format!(
        "{indent}scf.for {index_name} = %c0 to %c{depth}_max step %c1 {{\n"
    ));
    indices.push(index_name);
    emit_loops(image, shape, depth + 1, indices, next_id, body, memref_ty)?;
    indices.pop();
    body.push_str(&format!("{indent}}}\n"));

    Ok(())
}

fn memref_type<const N: usize>(shape: Shape<N>) -> String {
    let extents = shape
        .0
        .iter()
        .map(|extent| extent.to_string())
        .collect::<Vec<_>>()
        .join("x");
    format!("memref<{extents}xf32>")
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
