use weave::{GridLike, Image};

fn main() -> image::ImageResult<()> {
    let raw = image::open("input.png")?.to_rgba8();
    let [w, h] = [raw.width() as usize, raw.height() as usize];
    let rgba = &raw;
    let src = Image::new(
        [w, h],
        (0..w)
            .flat_map(|x| (0..h).map(move |y| rgba.get_pixel(x as u32, y as u32).0))
            .collect(),
    );
    let fill = [18, 20, 28, 255];

    Image::from(&src.remap([2 * w, 2 * h], fill, |[x, y]| {
        let [qx, qy] = [x / w, y / h];
        let [px, py] = [x % w, y % h];
        let [x, y] = [px as isize - w as isize / 2, py as isize - h as isize / 2];
        let [x, y] = match [qx, qy] {
            [0, 0] => [x - (w as isize / 5), y + (h as isize / 8)],
            [1, 0] => [4 * x / 3, 4 * y / 3],
            [0, 1] => [-y, x],
            _ => [x - (7 * y) / 20, y],
        };
        let [x, y] = [x + w as isize / 2, y + h as isize / 2];
        ((0..w as isize).contains(&x) && (0..h as isize).contains(&y))
            .then_some([x as usize, y as usize])
    }))
    .save_rgba("output.png")
}
