use weave::{Grid, View};

fn blur_pixel(image: &Grid<f32, 2>, index: [usize; 2]) -> f32 {
    let view = View::new(image, index);
    let mut sum = 0.0_f32;

    for dx in -2..=2 {
        for dy in -2..=2 {
            sum += view.get([dx, dy]).unwrap_or(0.0);
        }
    }

    sum * 0.04
}

fn main() -> image::ImageResult<()> {
    let raw = image::open("input.png")?.to_luma8();
    let [w, h] = [raw.width() as usize, raw.height() as usize];
    let gray_image = &raw;

    let gray = Grid::new(
        [w, h],
        (0..w)
            .flat_map(|x| {
                (0..h)
                    .map(move |y| f32::from(gray_image.get_pixel(x as u32, y as u32).0[0]) / 255.0)
            })
            .collect(),
    );

    let blurred = Grid::tabulate([w, h], |index| blur_pixel(&gray, index));

    let output = Grid::tabulate([w, h], |[x, y]| {
        let value = (blurred.sample([x, y]).clamp(0.0, 1.0) * 255.0).round() as u8;
        [value, value, value, 255]
    });

    output.save_rgba("output.png")
}
