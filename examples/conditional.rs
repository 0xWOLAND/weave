use weave::{
    Arithmetic, Grid, GridLike, LowerableGrid, LoweringError, MapKernel, MeliorBackend,
    Representable,
};

struct Threshold {
    cutoff: f32,
    low: f32,
    high: f32,
}

impl MapKernel<f32, 2> for Threshold {
    fn apply<B: Arithmetic<2, f32>>(
        &self,
        backend: &mut B,
        value: B::Scalar,
    ) -> Result<B::Scalar, LoweringError> {
        let cutoff = backend.literal(self.cutoff)?;
        let low = backend.literal(self.low)?;
        let high = backend.literal(self.high)?;
        let is_bright = backend.gt(value, cutoff)?;
        backend.select(is_bright, high, low)
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

    let thresholded = gray
        .staged()
        .map(Threshold {
            cutoff: 0.5,
            low: 0.0,
            high: 1.0,
        })
        .materialize_with::<MeliorBackend>()
        .expect("execute staged threshold pipeline");

    let output = Grid::tabulate([w, h], |[x, y]| {
        let value = (thresholded.at([x, y]).clamp(0.0, 1.0) * 255.0).round() as u8;
        [value, value, value, 255]
    });

    output.save_rgba("output-threshold.png")
}
