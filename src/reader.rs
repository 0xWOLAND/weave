use std::rc::Rc;

#[derive(Clone)]
pub struct Reader<'a, Env, A> {
    run: Rc<dyn Fn(&Env) -> A + 'a>,
}

impl<'a, Env: 'a, A> Reader<'a, Env, A> {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&Env) -> A + 'a,
    {
        Self { run: Rc::new(f) }
    }

    pub fn asks<F>(f: F) -> Self
    where
        F: Fn(&Env) -> A + 'a,
    {
        Self::new(f)
    }

    pub fn run(&self, env: &Env) -> A {
        (self.run)(env)
    }

    pub fn map<B, F>(self, f: F) -> Reader<'a, Env, B>
    where
        F: Fn(A) -> B + 'a,
        A: 'a,
    {
        let run = self.run;
        Reader::new(move |env| f(run(env)))
    }

    pub fn and_then<B, F>(self, f: F) -> Reader<'a, Env, B>
    where
        F: Fn(A) -> Reader<'a, Env, B> + 'a,
        A: 'a,
    {
        let run = self.run;
        Reader::new(move |env| f(run(env)).run(env))
    }

    pub fn local<Outer: 'a, F>(self, f: F) -> Reader<'a, Outer, A>
    where
        F: Fn(&Outer) -> Env + 'a,
        Env: 'a,
        A: 'a,
    {
        let run = self.run;
        Reader::new(move |outer| {
            let env = f(outer);
            run(&env)
        })
    }
}

impl<'a, Env> Reader<'a, Env, Env>
where
    Env: Clone + 'a,
{
    pub fn ask() -> Self {
        Self::new(Clone::clone)
    }
}
