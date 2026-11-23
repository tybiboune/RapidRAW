// Face restoration with CodeFormer (S-Lab, non-commercial), from the export facefusion publishes.
//
// Each face is turned and scaled onto the 512-pixel square the network learned on, by the same five
// points the FFHQ faces were aligned by; the network gives back the face rebuilt from what it knows
// faces look like, as much or as little as the fidelity says; and the result goes back onto the
// picture through the same turn, feathered at the edges of the square so the seam does not show.

use std::sync::Mutex;

use anyhow::Result;
use image::{Rgb, RgbImage};
use ndarray::{Array0, Array4};
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use crate::masks::face::{FaceAnalysis, alignment_points};
use crate::retouch::clone_heal::Bounds;

const SIZE: usize = 512;
// FFHQ's five points in a 512 crop, as CodeFormer's face helper aligns a face: the eye on the left of
// the picture, the other eye, the tip of the nose, then the corners of the mouth, left and right
const TEMPLATE: [[f32; 2]; 5] = [
    [192.98138, 239.94708],
    [318.90277, 240.19366],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
];
// how the restored square is let in at its edges, in pixels of the square: not at all this close to
// an edge, fully from twice as far in, which is about how facefusion feathers the box it pastes back
const EDGE: f32 = 38.0;
const FEATHER: f32 = 76.0;
// a face many times bigger than the square is averaged into it, at most this many samples a side
const MAX_SUPERSAMPLING: usize = 4;

/// A turn, a scale and a shift: p' = [a -b; b a] p + t.
#[derive(Clone, Copy, Debug)]
struct Similarity {
    a: f32,
    b: f32,
    tx: f32,
    ty: f32,
}

impl Similarity {
    // the one that takes `from` onto `to` most closely, by least squares
    fn fit(from: &[[f32; 2]; 5], to: &[[f32; 2]; 5]) -> Self {
        let mean = |points: &[[f32; 2]; 5]| {
            let sum = points
                .iter()
                .fold([0.0f32; 2], |s, p| [s[0] + p[0], s[1] + p[1]]);
            [sum[0] / 5.0, sum[1] / 5.0]
        };
        let (mf, mt) = (mean(from), mean(to));
        let (mut dot, mut cross, mut spread) = (0.0f32, 0.0f32, 0.0f32);
        for (f, t) in from.iter().zip(to) {
            let (fx, fy) = (f[0] - mf[0], f[1] - mf[1]);
            let (tx, ty) = (t[0] - mt[0], t[1] - mt[1]);
            dot += fx * tx + fy * ty;
            cross += fx * ty - fy * tx;
            spread += fx * fx + fy * fy;
        }
        let a = dot / spread.max(1e-6);
        let b = cross / spread.max(1e-6);
        Similarity {
            a,
            b,
            tx: mt[0] - (a * mf[0] - b * mf[1]),
            ty: mt[1] - (b * mf[0] + a * mf[1]),
        }
    }

    fn apply(self, [x, y]: [f32; 2]) -> [f32; 2] {
        [
            self.a * x - self.b * y + self.tx,
            self.b * x + self.a * y + self.ty,
        ]
    }

    fn inverse(self) -> Self {
        let d = (self.a * self.a + self.b * self.b).max(1e-12);
        let (a, b) = (self.a / d, -self.b / d);
        Similarity {
            a,
            b,
            tx: -(a * self.tx - b * self.ty),
            ty: -(b * self.tx + a * self.ty),
        }
    }

    // how many pixels of the picture one unit of the other side spans
    fn scale(self) -> f32 {
        self.a.hypot(self.b)
    }
}

// The picture at a point, between its pixels, its edges carried on outward.
fn sample(image: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (image.width() as f32, image.height() as f32);
    let x = x.clamp(0.0, w - 1.0);
    let y = y.clamp(0.0, h - 1.0);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as u32, y0 as u32);
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let at = |x: u32, y: u32| image.get_pixel(x, y).0.map(f32::from);
    let (p00, p10, p01, p11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
    std::array::from_fn(|c| {
        let top = p00[c] + (p10[c] - p00[c]) * fx;
        let bottom = p01[c] + (p11[c] - p01[c]) * fx;
        top + (bottom - top) * fy
    })
}

// The face cut out onto the square, -1 to 1, channels first. `to_picture` takes a point of the square to
// the picture; when a pixel of the square covers many of the picture's, several samples are averaged.
fn face_tensor(image: &RgbImage, to_picture: Similarity) -> Array4<f32> {
    let spread = to_picture.scale();
    let n = (spread.ceil() as usize).clamp(1, MAX_SUPERSAMPLING);
    let plane = SIZE * SIZE;
    let mut tensor = Array4::<f32>::zeros((1, 3, SIZE, SIZE));
    let data = tensor.as_slice_mut().unwrap();
    let (red, rest) = data.split_at_mut(plane);
    let (green, blue) = rest.split_at_mut(plane);
    red.par_chunks_mut(SIZE)
        .zip(green.par_chunks_mut(SIZE))
        .zip(blue.par_chunks_mut(SIZE))
        .enumerate()
        .for_each(|(y, ((r, g), b))| {
            let row = r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut());
            for (x, ((r, g), b)) in row.enumerate() {
                let mut sum = [0.0f32; 3];
                for sy in 0..n {
                    for sx in 0..n {
                        let u = x as f32 + (sx as f32 + 0.5) / n as f32 - 0.5;
                        let v = y as f32 + (sy as f32 + 0.5) / n as f32 - 0.5;
                        let [px, py] = to_picture.apply([u, v]);
                        let rgb = sample(image, px, py);
                        (0..3).for_each(|c| sum[c] += rgb[c]);
                    }
                }
                let k = (n * n) as f32 * 127.5;
                [*r, *g, *b] = sum.map(|v| v / k - 1.0);
            }
        });
    tensor
}

fn run_codeformer(
    session: &Mutex<Session>,
    tensor: Array4<f32>,
    fidelity: f32,
) -> Result<Vec<f32>> {
    let weight = Array0::from_elem((), fidelity.clamp(0.0, 1.0) as f64);
    let mut session = session.lock().unwrap();
    let outputs = session.run(ort::inputs![
        "input" => Tensor::from_array(tensor)?,
        "weight" => Tensor::from_array(weight)?,
    ])?;
    let restored: Vec<f32> = outputs["output"]
        .try_extract_array::<f32>()?
        .iter()
        .copied()
        .collect();
    if restored.len() != 3 * SIZE * SIZE {
        return Err(anyhow::anyhow!(
            "Unexpected face restoration output of {} values",
            restored.len()
        ));
    }
    Ok(restored)
}

// How much of the restored square counts at a point of it: nothing at its edges, all of it inside.
fn square_weight(u: f32, v: f32) -> f32 {
    let edge = u
        .min(v)
        .min(SIZE as f32 - 1.0 - u)
        .min(SIZE as f32 - 1.0 - v);
    let t = ((edge - EDGE) / (FEATHER - EDGE)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Every face of the analysis that reaches into `bounds`, restored by CodeFormer and laid back over the
/// picture there, `strength` of the way. `fidelity` is CodeFormer's own weight: near 0 it rebuilds the
/// face from what it knows faces look like, near 1 it keeps close to this one. The result is the patch,
/// the size of `bounds`.
pub fn restore_faces(
    image: &RgbImage,
    analysis: &FaceAnalysis,
    session: &Mutex<Session>,
    bounds: Bounds,
    fidelity: f32,
    strength: f32,
) -> Result<RgbImage> {
    let (min_x, max_x, min_y, max_y) = bounds;
    let (w, h) = ((max_x - min_x + 1) as u32, (max_y - min_y + 1) as u32);
    let mut patch = image::imageops::crop_imm(image, min_x as u32, min_y as u32, w, h).to_image();
    let strength = strength.clamp(0.0, 1.0);
    for face in &analysis.faces {
        let to_square = Similarity::fit(&alignment_points(face), &TEMPLATE);
        let to_picture = to_square.inverse();
        // where the square lands on the picture, against the patch
        let corners = [
            [0.0, 0.0],
            [SIZE as f32, 0.0],
            [0.0, SIZE as f32],
            [SIZE as f32; 2],
        ]
        .map(|corner| to_picture.apply(corner));
        let reach = |axis: usize, lo: usize, hi: usize| {
            let min = corners.iter().map(|c| c[axis]).fold(f32::MAX, f32::min);
            let max = corners.iter().map(|c| c[axis]).fold(f32::MIN, f32::max);
            (
                min.floor().max(lo as f32) as usize,
                max.ceil().min(hi as f32) as usize,
            )
        };
        let (x0, x1) = reach(0, min_x, max_x);
        let (y0, y1) = reach(1, min_y, max_y);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }

        let restored = run_codeformer(session, face_tensor(image, to_picture), fidelity)?;
        let plane = SIZE * SIZE;
        let at = |c: usize, x: usize, y: usize| restored[c * plane + y * SIZE + x];
        for y in y0..=y1 {
            for x in x0..=x1 {
                let [u, v] = to_square.apply([x as f32, y as f32]);
                if u < 0.0 || v < 0.0 || u > (SIZE - 1) as f32 || v > (SIZE - 1) as f32 {
                    continue;
                }
                let k = square_weight(u, v) * strength;
                if k <= 0.0 {
                    continue;
                }
                let (u0, v0) = (u.floor() as usize, v.floor() as usize);
                let (u1, v1) = ((u0 + 1).min(SIZE - 1), (v0 + 1).min(SIZE - 1));
                let (fu, fv) = (u - u0 as f32, v - v0 as f32);
                let pixel = patch.get_pixel_mut((x - min_x) as u32, (y - min_y) as u32);
                let Rgb(before) = *pixel;
                *pixel = Rgb(std::array::from_fn(|c| {
                    let top = at(c, u0, v0) + (at(c, u1, v0) - at(c, u0, v0)) * fu;
                    let bottom = at(c, u0, v1) + (at(c, u1, v1) - at(c, u0, v1)) * fu;
                    let value = ((top + (bottom - top) * fv).clamp(-1.0, 1.0) + 1.0) * 127.5;
                    (before[c] as f32 + (value - before[c] as f32) * k).round() as u8
                }));
            }
        }
    }
    Ok(patch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_similarity_fits_and_undoes_itself() {
        let truth = Similarity {
            a: 0.8,
            b: 0.3,
            tx: 12.0,
            ty: -7.0,
        };
        let from = TEMPLATE.map(|p| [p[0] * 0.5 + 40.0, p[1] * 0.7 - 3.0]);
        let to = from.map(|p| truth.apply(p));
        let fitted = Similarity::fit(&from, &to);
        for p in from {
            let [x, y] = fitted.apply(p);
            let [ex, ey] = truth.apply(p);
            assert!((x - ex).abs() < 1e-2 && (y - ey).abs() < 1e-2);
            let [bx, by] = fitted.inverse().apply([x, y]);
            assert!((bx - p[0]).abs() < 1e-2 && (by - p[1]).abs() < 1e-2);
        }
    }

    #[test]
    fn the_square_is_let_in_only_away_from_its_edges() {
        assert_eq!(square_weight(0.0, 256.0), 0.0);
        assert_eq!(square_weight(256.0, 256.0), 1.0);
        let mid = square_weight((EDGE + FEATHER) / 2.0, 256.0);
        assert!(mid > 0.0 && mid < 1.0);
    }
}
