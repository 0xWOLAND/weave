use std::iter::successors;

use weave::{GridLike, Image, View};

type Complex = [f32; 2];
type Pixel = [u8; 4];

fn complex(view: View<&Image<()>, 2>) -> Complex {
    let [x, y] = view.position();
    let [w, h] = view.shape().0;
    [-2.2 + 3.2 * x as f32 / w as f32, -1.2 + 2.4 * y as f32 / h as f32]
}

fn orbit([cx, cy]: Complex) -> impl Iterator<Item = Complex> {
    successors(Some([0.0, 0.0]), move |&[zx, zy]| {
        Some([zx * zx - zy * zy + cx, 2.0 * zx * zy + cy])
    })
}

fn escape(max: u32) -> impl Fn(Complex) -> u32 {
    move |c| {
        orbit(c)
            .take(max as usize + 1)
            .position(|[zx, zy]| zx * zx + zy * zy > 4.0)
            .map_or(max, |i| i as u32)
    }
}

fn color(max: u32) -> impl Fn(u32) -> Pixel {
    move |i| {
        if i == max {
            return [0, 0, 0, 255];
        }
        let t = i as f32 / max as f32;
        [
            (9.0 * (1.0 - t) * t * t * t * 255.0) as u8,
            (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0) as u8,
            (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0) as u8,
            255,
        ]
    }
}

fn main() -> image::ImageResult<()> {
    const SHAPE: [usize; 2] = [1024, 768];
    const MAX: u32 = 256;

    Image::from(
        &Image::new(SHAPE, vec![(); SHAPE[0] * SHAPE[1]])
            .extend(complex)
            .map(escape(MAX))
            .map(color(MAX)),
    )
    .save_rgba("output.png")
}
