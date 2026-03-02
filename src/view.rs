use crate::{Grid, Store};

#[derive(Clone, Copy)]
pub struct View<G: Grid> {
    grid: G,
    x: usize,
    y: usize,
}

impl<G: Grid> View<G> {
    pub fn new(grid: G, x: usize, y: usize) -> Self {
        Self { grid, x, y }
    }

    pub fn x(&self) -> usize {
        self.x
    }

    pub fn y(&self) -> usize {
        self.y
    }

    pub fn width(&self) -> usize {
        self.grid.width()
    }

    pub fn height(&self) -> usize {
        self.grid.height()
    }

    pub fn extract(&self) -> G::Elem {
        self.grid.at(self.x, self.y)
    }

    pub fn get(&self, dx: isize, dy: isize) -> Option<G::Elem> {
        let nx = self.x as isize + dx;
        let ny = self.y as isize + dy;

        if nx >= 0
            && ny >= 0
            && nx < self.grid.width() as isize
            && ny < self.grid.height() as isize
        {
            Some(self.grid.at(nx as usize, ny as usize))
        } else {
            None
        }
    }
}

impl<'a, G: Grid> From<View<&'a G>> for Store<'a, (usize, usize), G::Elem> {
    fn from(view: View<&'a G>) -> Self {
        let grid = view.grid;
        Store::new((view.x, view.y), move |(x, y)| grid.at(x, y))
    }
}
