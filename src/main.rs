use image::{RgbaImage, error::ParameterError, error::ParameterErrorKind};
use weave::{GridLike, Image, View};

type Pixel = [u8; 4];

fn mandelbrot(view: View<&Image<()>, 2>, max_iter: u32) -> u32 {
    let [x, y] = view.position();
    let [w, h] = view.shape().0;
    let cx = -2.2 + 3.2 * x as f32 / w as f32;
    let cy = -1.2 + 2.4 * y as f32 / h as f32;

    view.iterate((0.0f32, 0.0f32), |(zx, zy)| {
        (zx * zx - zy * zy + cx, 2.0 * zx * zy + cy)
    })
    .take(max_iter as usize + 1)
    .position(|(zx, zy)| zx * zx + zy * zy > 4.0)
    .map_or(max_iter, |iter| iter as u32)
}

fn color(iter: u32, max_iter: u32) -> Pixel {
    if iter == max_iter {
        return [0, 0, 0, 255];
    }

    let t = iter as f32 / max_iter as f32;
    [
        (9.0 * (1.0 - t) * t * t * t * 255.0) as u8,
        (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8,
        (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8,
        255,
    ]
}

fn main() -> image::ImageResult<()> {
    const SHAPE: [usize; 2] = [1024, 768];
    const MAX_ITER: u32 = 256;

    let image = Image::from(
        &Image::new(SHAPE, vec![(); SHAPE[0] * SHAPE[1]])
            .extend(|view| mandelbrot(view, MAX_ITER))
            .map(|iter| color(iter, MAX_ITER)),
    );
    let [w, h] = image.shape.0;
    let grid = &image;
    let pixels = (0..h)
        .flat_map(|y| (0..w).map(move |x| grid.at([x, y])))
        .flat_map(|rgba| rgba)
        .collect();

    RgbaImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| {
            image::ImageError::Parameter(ParameterError::from_kind(
                ParameterErrorKind::DimensionMismatch,
            ))
        })?
        .save("output.png")
}
