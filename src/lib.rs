#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Mul,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Lt,
    Le,
    Eq,
}

pub type Buffer = String;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Value(u32);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Block(u32);

#[derive(Clone, Debug)]
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
        buffer: Buffer,
        index: Value,
    },
}

pub struct Function {
    pub insts: Vec<Inst>,
}

impl Function {
    pub fn new() -> Self {
        Self { insts: Vec::new() }
    }

    fn emit(&mut self, inst: Inst) -> Value {
        let v = Value(self.insts.len() as u32);
        self.insts.push(inst);
        v
    }

    pub fn c(&mut self, v: i64) -> Value {
        self.emit(Inst::Const(v))
    }

    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.emit(Inst::Bin {
            op: BinOp::Add,
            lhs: a,
            rhs: b,
        })
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.emit(Inst::Bin {
            op: BinOp::Mul,
            lhs: a,
            rhs: b,
        })
    }

    pub fn cmp(&mut self, op: CmpOp, a: Value, b: Value) -> Value {
        self.emit(Inst::Cmp {
            op,
            lhs: a,
            rhs: b,
        })
    }

    pub fn select(&mut self, c: Value, t: Value, f: Value) -> Value {
        self.emit(Inst::Select { cond: c, t, f })
    }

    pub fn load(&mut self, buf: &str, idx: Value) -> Value {
        self.emit(Inst::Load {
            buffer: buf.to_string(),
            index: idx,
        })
    }
}


#[test]
fn example() {
    let mut f = Function::new();

    let x = f.c(7);                
    let two = f.c(2);
    let one = f.c(1);

    let a = f.load("a", x);
    let mul = f.mul(a, two);
    let add = f.add(mul, one);

    println!("result = {:?}", add);
}
