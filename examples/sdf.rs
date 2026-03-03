use std::fs::File;

use image::{
    Delay, Frame, Rgba, RgbaImage,
    codecs::gif::{GifEncoder, Repeat},
};
use weave::{GridLike, Image, View};

type Pixel = [u8; 4];
type Vec3 = [f32; 3];
type Basis = (Vec3, Vec3, Vec3);
type Ray = (Vec3, Vec3);

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul(a: Vec3, t: f32) -> Vec3 {
    [a[0] * t, a[1] * t, a[2] * t]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: Vec3) -> Vec3 {
    mul(a, dot(a, a).sqrt().recip())
}

fn mix(a: Pixel, b: Pixel, t: f32) -> Pixel {
    let s = t.clamp(0.0, 1.0);
    [
        (a[0] as f32 + (b[0] as f32 - a[0] as f32) * s) as u8,
        (a[1] as f32 + (b[1] as f32 - a[1] as f32) * s) as u8,
        (a[2] as f32 + (b[2] as f32 - a[2] as f32) * s) as u8,
        255,
    ]
}

fn sdf([x, y, z]: Vec3) -> f32 {
    let mut zeta = [x, y, z];
    let mut dr = 1.0;
    let mut r = 0.0;

    for _ in 0..8 {
        r = dot(zeta, zeta).sqrt();
        if r > 2.0 {
            break;
        }
        let theta = (zeta[2] / r.max(1e-6)).acos() * 8.0;
        let phi = zeta[1].atan2(zeta[0]) * 8.0;
        let zr = r.powi(8);
        dr = 8.0 * r.powi(7) * dr + 1.0;
        zeta = [
            zr * theta.sin() * phi.cos() + x,
            zr * theta.sin() * phi.sin() + y,
            zr * theta.cos() + z,
        ];
    }

    0.5 * r.max(1e-6).ln() * r / dr
}

fn normal(p: Vec3) -> Vec3 {
    let e = 0.001;
    norm([
        sdf(add(p, [e, 0.0, 0.0])) - sdf(add(p, [-e, 0.0, 0.0])),
        sdf(add(p, [0.0, e, 0.0])) - sdf(add(p, [0.0, -e, 0.0])),
        sdf(add(p, [0.0, 0.0, e])) - sdf(add(p, [0.0, 0.0, -e])),
    ])
}

fn uv(view: View<&Image<()>, 2>) -> [f32; 2] {
    let [x, y] = view.position();
    let [w, h] = view.shape().0;
    [
        (2.0 * x as f32 - w as f32) / h as f32,
        (2.0 * y as f32 - h as f32) / h as f32,
    ]
}

fn camera(angle: f32) -> (Vec3, Basis) {
    let ro = [3.0 * angle.cos(), 0.8, 3.0 * angle.sin()];
    let ww = norm(mul(ro, -1.0));
    let uu = norm(cross([0.0, 1.0, 0.0], ww));
    let vv = cross(ww, uu);
    (ro, (uu, vv, ww))
}

fn ray((ro, (uu, vv, ww)): (Vec3, Basis), uv: [f32; 2]) -> Ray {
    (
        ro,
        norm(add(add(mul(uu, uv[0]), mul(vv, uv[1])), mul(ww, 1.7))),
    )
}

fn march(view: View<&Image<()>, 2>, (ro, rd): Ray) -> Option<f32> {
    let p = |t| add(ro, mul(rd, t));
    view.iterate(0.0, |t| t + sdf(p(t)).max(0.01))
        .take(96)
        .find(|&t| sdf(p(t)) < 0.001)
}

fn shade((ro, rd): Ray, t: f32) -> Pixel {
    let hit = add(ro, mul(rd, t));
    let n = normal(hit);
    let l = norm([0.6, 0.8, -0.5]);
    let diffuse = dot(n, l).max(0.0);
    let rim = (1.0 - dot(n, mul(rd, -1.0)).max(0.0)).powi(2);
    let light = (0.2 + 0.7 * diffuse + 0.5 * rim) / (1.0 + 0.15 * t * t);
    mix([24, 40, 90, 255], [255, 210, 120, 255], light)
}

fn background([_, y]: [f32; 2]) -> Pixel {
    mix([8, 10, 16, 255], [24, 36, 60, 255], 0.5 * (y + 1.0))
}

fn render(view: View<&Image<()>, 2>, angle: f32) -> Pixel {
    let uv = uv(view);
    let ray = ray(camera(angle), uv);
    march(view, ray).map_or_else(|| background(uv), |t| shade(ray, t))
}

fn main() -> image::ImageResult<()> {
    const SHAPE: [usize; 2] = [1024, 768];
    const FRAMES: usize = 1;

    let seed = Image::new(SHAPE, vec![(); SHAPE[0] * SHAPE[1]]);
    let mut gif = GifEncoder::new(File::create("output.gif")?);
    gif.set_repeat(Repeat::Infinite)?;

    for i in 0..FRAMES {
        let angle = i as f32 * std::f32::consts::TAU / FRAMES as f32;
        let image = Image::from(&seed.extend(|view| render(view, angle)));
        let frame = Frame::from_parts(
            RgbaImage::from_fn(SHAPE[0] as u32, SHAPE[1] as u32, |x, y| {
                Rgba(image.at([x as usize, y as usize]))
            }),
            0,
            0,
            Delay::from_numer_denom_ms(50, 1),
        );
        gif.encode_frame(frame)?;
    }

    Ok(())
}
