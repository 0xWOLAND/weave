use image::ImageReader;
use weave::{Grid, Image, View, materialize};

fn blur(view: View<impl Grid<Elem = f32>>) -> f32 {
    let mut sum = 0.0;
    let mut count = 0.0;

    for dy in -1..=1 {
        for dx in -1..=1 {
            if let Some(v) = view.get(dx, dy) {
                sum += v;
                count += 1.0;
            }
        }
    }

    sum / count
}

fn crop(view: View<impl Grid<Elem = f32>>) -> Option<f32> {
    let x = view.x();
    let y = view.y();
    let w = view.width();
    let h = view.height();

    (x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4).then(|| view.extract())
}

pub fn main() {
    let img = ImageReader::open("image.png").unwrap().decode().unwrap().to_luma8();
    let width = img.width() as usize;
    let height = img.height() as usize;
    let img: Vec<f32> = img.into_raw().into_iter().map(|v| v as f32 / 255.0).collect();
    let img = Image::new(width, height, img);

    let result = img
        .extend(blur)
        .extend(crop)
        .map(|x| x.unwrap_or(0.0))
        .map(|x| x.sqrt());

    let output = materialize(result);
    let output_img = output.data.into_iter().map(|v| (v * 255.0) as u8).collect();
    let output_img = image::GrayImage::from_raw(width as u32, height as u32, output_img).unwrap();
    output_img.save("output.png").unwrap();
}
