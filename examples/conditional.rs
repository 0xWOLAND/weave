use weave::{Grid, ImageExt, MeliorBackend};

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

    let thresholded = (&gray)
        .mul_const(1.0)
        .threshold(0.5_f32, 1.0_f32, 0.0_f32)
        .materialize_with::<MeliorBackend>()
        .expect("materialize arithmetic image with MLIR");

    let output = Grid::tabulate([w, h], |[x, y]| {
        let value = (thresholded.sample([x, y]).clamp(0.0, 1.0) * 255.0).round() as u8;
        [value, value, value, 255]
    });

    output.save_rgba("output-threshold.png")
}
