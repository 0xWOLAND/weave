use std::fs::File;
use image::{Delay, Frame, RgbaImage, codecs::gif::{GifEncoder, Repeat}};
use weave::{Field, Grid, Image, Representable, View};

fn convolve<G: Grid<Elem = f32>, const W: usize, const H: usize>(
    view: View<G>,
    kernel: &[[f32; W]; H],
) -> f32 {
    let center = view.extract();
    let cx = (W / 2) as isize;
    let cy = (H / 2) as isize;

    kernel
        .iter()
        .enumerate()
        .fold(0.0, |acc, (ky, row)| {
            acc + row.iter().enumerate().fold(0.0, |acc, (kx, &weight)| {
                let sample = view
                    .get(kx as isize - cx, ky as isize - cy)
                    .unwrap_or(center);
                acc + weight * sample
            })
        })
}

fn laplacian<G: Grid<Elem = f32>>(view: View<&G>) -> f32 {
    const KERNEL: [[f32; 3]; 3] = [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ];

    convolve(view, &KERNEL)
}

fn step((u, lap): (f32, f32)) -> f32 {
    u + 0.1 * lap
}

fn advance<G: Grid<Elem = f32> + Clone>(grid: G) -> Image<f32> {
    Image::from(&grid.clone().zip(grid.extend(laplacian)).map(step))
}

pub fn main() {
    const W: usize = 256;
    const H: usize = 256;

    let img = Field::<W, H, f32>::tabulate(|(x, y)| {
        1000.0 * ((x, y) == (W / 2, H / 2)) as u8 as f32
    });

    let mut state = Image::from(&img);
    let mut gif = GifEncoder::new(File::create("output.gif").unwrap());
    gif.set_repeat(Repeat::Infinite).unwrap();

    for _ in 0..(1<<8){
        state = advance(state);
        let pixels = state.data.iter().flat_map(|&v| {
            let u = (v.sqrt() * 255.0) as u8;
            [u, u, u, 255]
        }).collect();
        let frame = Frame::from_parts(
            RgbaImage::from_raw(W as u32, H as u32, pixels).unwrap(),
            0,
            0,
            Delay::from_numer_denom_ms(30, 1),
        );
        gif.encode_frame(frame).unwrap();
    }
}
