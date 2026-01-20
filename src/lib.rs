use std::marker::PhantomData;

pub trait Phase: std::fmt::Debug + 'static {
    type Inst: std::fmt::Debug;
}

#[derive(Debug)]
pub enum Algorithm {}

impl Phase for Algorithm {
    type Inst = AlgorithmInst;
}

#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Value<P: Phase>(u32, PhantomData<P>);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp { Add, Mul }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CmpOp { Lt, Le, Eq }

#[derive(Debug)]
pub enum AlgorithmInst {
    Const(i64),
    Bin { op: BinOp, lhs: Value<Algorithm>, rhs: Value<Algorithm> },
    Cmp { op: CmpOp, lhs: Value<Algorithm>, rhs: Value<Algorithm> },
    Select { cond: Value<Algorithm>, t: Value<Algorithm>, f: Value<Algorithm> },
    Load { buffer: String, index: Value<Algorithm> },
}

pub struct Function<P: Phase> {
    pub insts: Vec<P::Inst>,
}

impl<P: Phase> Function<P> {
    pub fn new() -> Self {
        Self { insts: Vec::new() }
    }

    #[inline]
    fn push(&mut self, inst: P::Inst) -> Value<P> {
        let id = self.insts.len() as u32;
        self.insts.push(inst);
        Value(id, PhantomData)
    }
}

pub trait AlgorithmBuilder {
    fn c(&mut self, v: i64) -> Value<Algorithm>;
    fn load(&mut self, buf: &str, idx: Value<Algorithm>) -> Value<Algorithm>;
    fn add(&mut self, lhs: Value<Algorithm>, rhs: Value<Algorithm>) -> Value<Algorithm>;
    fn mul(&mut self, lhs: Value<Algorithm>, rhs: Value<Algorithm>) -> Value<Algorithm>;
    fn cmp(&mut self, op: CmpOp, lhs: Value<Algorithm>, rhs: Value<Algorithm>) -> Value<Algorithm>;
    fn select(&mut self, cond: Value<Algorithm>, t: Value<Algorithm>, f: Value<Algorithm>) -> Value<Algorithm>;
}

impl AlgorithmBuilder for Function<Algorithm> {
    fn c(&mut self, v: i64) -> Value<Algorithm> {
        self.push(AlgorithmInst::Const(v))
    }

    fn load(&mut self, buf: &str, index: Value<Algorithm>) -> Value<Algorithm> {
        self.push(AlgorithmInst::Load { buffer: buf.to_string(), index })
    }

    fn add(&mut self, lhs: Value<Algorithm>, rhs: Value<Algorithm>) -> Value<Algorithm> {
        self.push(AlgorithmInst::Bin { op: BinOp::Add, lhs, rhs })
    }

    fn mul(&mut self, lhs: Value<Algorithm>, rhs: Value<Algorithm>) -> Value<Algorithm> {
        self.push(AlgorithmInst::Bin { op: BinOp::Mul, lhs, rhs })
    }

    fn cmp(&mut self, op: CmpOp, lhs: Value<Algorithm>, rhs: Value<Algorithm>) -> Value<Algorithm> {
        self.push(AlgorithmInst::Cmp { op, lhs, rhs })
    }

    fn select(&mut self, cond: Value<Algorithm>, t: Value<Algorithm>, f: Value<Algorithm>) -> Value<Algorithm> {
        self.push(AlgorithmInst::Select { cond, t, f })
    }
}

#[test]
fn tests() {
    let mut f = Function::<Algorithm>::new();

    let x = f.c(7);
    let two = f.c(2);
    let one = f.c(1);

    let a = f.load("a", x);
    let mul = f.mul(a, two);
    let res = f.add(mul, one);

    // Verification
    assert_eq!(f.insts.len(), 6);
    println!("Final Result Value: {:?}", res);
    
    for (i, inst) in f.insts.iter().enumerate() {
        println!("%{} = {:?}", i, inst);
    }
}