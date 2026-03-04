use weave::{Grid, View};

fn main() {
    let input = Grid::new([3, 3], (0..9).collect());

    let output = Grid::tabulate([3, 3], |index| {
        let view = View::new(&input, index);
        let center = view.extract();
        let left = view.get([-1, 0]).unwrap_or(0);
        let right = view.get([1, 0]).unwrap_or(0);
        let up = view.get([0, -1]).unwrap_or(0);
        let down = view.get([0, 1]).unwrap_or(0);
        center + left + right + up + down
    });

    println!("input:  {:?}", input.data);
    println!("output: {:?}", output.data);
    println!("sample at [1, 1]: {}", output.sample([1, 1]));
}
