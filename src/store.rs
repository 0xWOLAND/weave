use std::rc::Rc;

#[derive(Clone)]
pub struct Store<'a, State: Copy, A> {
    position: State,
    peek: Rc<dyn Fn(State) -> A + 'a>,
}

impl<'a, State: Copy + 'a, A> Store<'a, State, A> {
    pub fn new<F>(position: State, peek: F) -> Self
    where
        F: Fn(State) -> A + 'a,
    {
        Self {
            position,
            peek: Rc::new(peek),
        }
    }

    pub fn position(&self) -> State {
        self.position
    }

    pub fn peek(&self, position: State) -> A {
        (self.peek)(position)
    }

    pub fn extract(&self) -> A {
        self.peek(self.position)
    }

    pub fn seek(&self, position: State) -> Self {
        Self {
            position,
            peek: Rc::clone(&self.peek),
        }
    }

    pub fn seeks<F>(&self, f: F) -> Self
    where
        F: Fn(State) -> State,
    {
        self.seek(f(self.position))
    }

    pub fn map<B, F>(self, f: F) -> Store<'a, State, B>
    where
        F: Fn(A) -> B + 'a,
        A: 'a,
    {
        let peek = self.peek;
        let position = self.position;
        Store::new(position, move |state| f(peek(state)))
    }

    pub fn duplicate(&self) -> Store<'a, State, Store<'a, State, A>>
    where
        A: 'a,
    {
        let peek = Rc::clone(&self.peek);
        Store::new(self.position, move |position| Store {
            position,
            peek: Rc::clone(&peek),
        })
    }

    pub fn extend<B, F>(&self, f: F) -> Store<'a, State, B>
    where
        F: Fn(Store<'a, State, A>) -> B + 'a,
        A: 'a,
    {
        self.duplicate().map(f)
    }

    pub fn experiment<I>(&self, positions: I) -> Vec<A>
    where
        I: IntoIterator<Item = State>,
    {
        positions
            .into_iter()
            .map(|position| self.peek(position))
            .collect()
    }
}
