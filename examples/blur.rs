use weave::{
    Arithmetic, ExtendKernel, Grid, GridLike, LowerableGrid, LoweringError, MeliorBackend,
    Representable,
};

struct Blur;

impl ExtendKernel<f32, 2> for Blur {
    fn apply<B: Arithmetic<2, f32>, G: LowerableGrid<2, Elem = f32> + ?Sized>(
        &self,
        backend: &mut B,
        view: B::View<'_, G>,
    ) -> Result<B::Scalar, LoweringError> {
        let zero = backend.literal(0.0)?;
        let one_twenty_fifth = backend.literal(0.04)?;
        let mut sum = zero.clone();

        for dx in -2..=2 {
            for dy in -2..=2 {
                let sampled = backend.get(&view, [dx, dy])?;
                let sample = backend.unwrap_or(sampled, zero.clone())?;
                sum = backend.add(sum, sample)?;
            }
        }

        backend.mul(sum, one_twenty_fifth)
    }
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

    let blurred = gray
        .staged()
        .extend(Blur)
        .materialize_with::<MeliorBackend>()
        .expect("execute staged blur pipeline");

    let output = Grid::tabulate([w, h], |[x, y]| {
        let value = (blurred.at([x, y]).clamp(0.0, 1.0) * 255.0).round() as u8;
        [value, value, value, 255]
    });

    output.save_rgba("output.png")
}
