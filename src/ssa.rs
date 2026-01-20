use crate::domain::Domain;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Value(u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Mul,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
}

#[derive(Debug)]
pub enum Inst {
    Const(i64),
    Bin {
        op: BinOp,
        lhs: Value,
        rhs: Value,
    },
    Cmp {
        op: CmpOp,
        lhs: Value,
        rhs: Value,
    },
    Select {
        cond: Value,
        t: Value,
        f: Value,
    },
    Load {
        buffer: &'static str,
        index: Value,
    },
}

#[derive(Debug, Default)]
pub struct Function {
    insts: Vec<Inst>,
}

impl Function {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn emit(&mut self, inst: Inst) -> Value {
        let id = Value(self.insts.len() as u32);
        self.insts.push(inst);
        id
    }

    pub fn c(&mut self, v: i64) -> Value {
        self.emit(Inst::Const(v))
    }

    pub fn load(&mut self, buf: &'static str, index: Value) -> Value {
        self.emit(Inst::Load { buffer: buf, index })
    }

    pub fn add(&mut self, lhs: Value, rhs: Value) -> Value {
        self.emit(Inst::Bin {
            op: BinOp::Add,
            lhs,
            rhs,
        })
    }

    pub fn mul(&mut self, lhs: Value, rhs: Value) -> Value {
        self.emit(Inst::Bin {
            op: BinOp::Mul,
            lhs,
            rhs,
        })
    }

    pub fn cmp(&mut self, op: CmpOp, lhs: Value, rhs: Value) -> Value {
        self.emit(Inst::Cmp { op, lhs, rhs })
    }

    pub fn select(&mut self, cond: Value, t: Value, f: Value) -> Value {
        self.emit(Inst::Select { cond, t, f })
    }
}

pub struct Func {
    pub domain: Domain,
    pub function: Function,
}

impl Func {
    pub fn new(function: Function, domain: Domain) -> Self {
        Self {
            domain,
            function,
        }
    }
}

#[test]
fn test_builder() {
    let mut f = Function::new();

    let x = f.c(7);
    let two = f.c(2);
    let one = f.c(1);

    let a = f.load("a", x);
    let mul = f.mul(a, two);
    let res = f.add(mul, one);

    assert_eq!(f.insts.len(), 6);

    println!("Final result: {:?}", res);
    for (i, inst) in f.insts.iter().enumerate() {
        println!("%{} = {:?}", i, inst);
    }
}
