use std::fs::File;
use image::{Delay, Frame, RgbaImage, codecs::gif::{GifEncoder, Repeat}};
use weave::{Field, Grid, Image, Representable, View, materialize};

fn laplacian(view: View<impl Grid<Elem = f32>>) -> f32 {
    let u = view.extract();
    view.get(-1, 0).unwrap_or(u)
        + view.get(1, 0).unwrap_or(u)
        + view.get(0, -1).unwrap_or(u)
        + view.get(0, 1).unwrap_or(u)
        - 4.0 * u
}

fn step((u, lap): (f32, f32)) -> f32 {
    u + 0.1 * lap
}

fn advance<G: Grid<Elem = f32> + Clone>(grid: G) -> Image<f32> {
    materialize(grid.clone().zip(grid.extend(laplacian)).map(step))
}

pub fn main() {
    const W: usize = 256;
    const H: usize = 256;

    let img = Field::<W, H, f32>::tabulate(|(x, y)| {
        1000.0 * ((x, y) == (W / 2, H / 2)) as u8 as f32
    });

    let mut state = materialize(img);
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
