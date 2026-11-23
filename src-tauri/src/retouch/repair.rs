// Three repairs for the touch-up tools, each kept to the area the owner brushed:
//
// - deblurring, by a network trained on camera shake (NAFNet) or on missed focus (Restormer);
// - sensor dust, the soft grey spots a speck on the sensor casts on every frame at small apertures;
// - power lines and cables across a sky.
//
// Dust and wires are found here by their shape rather than by a network: nothing open has been trained
// well enough on either. Dust is a round, soft, neutral darkening of a flat area, so it is divided back
// out, which keeps whatever was under it. A wire is a long thin dark line, so it is filled from the sky
// on either side of it.

use std::sync::Mutex;

use anyhow::Result;
use image::imageops::{self, FilterType};
use image::{ImageBuffer, Luma, Rgb, Rgb32FImage, RgbImage};
use ort::session::Session;
use rayon::prelude::*;

use crate::retouch::clone_heal::Bounds;

type Plane = Vec<f32>;

// sRGB-coded 0-255 to linear light and back, close enough for working out how much darker one place is
// than another
fn to_linear(v: u8) -> f32 {
    (v as f32 / 255.0).powf(2.2)
}

fn to_code(v: f32) -> u8 {
    (v.max(0.0).powf(1.0 / 2.2) * 255.0)
        .round()
        .clamp(0.0, 255.0) as u8
}

fn blur(plane: &[f32], w: usize, h: usize, sigma: f32) -> Plane {
    if sigma < 0.3 {
        return plane.to_vec();
    }
    let image = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(w as u32, h as u32, plane.to_vec())
        .expect("plane matches its size");
    imageops::blur(&image, sigma).into_raw()
}

// the bounds, grown by `pad` on every side and kept to the picture: (x0, y0, width, height)
fn padded(bounds: Bounds, pad: usize, image: &RgbImage) -> (usize, usize, usize, usize) {
    let (min_x, max_x, min_y, max_y) = bounds;
    let x0 = min_x.saturating_sub(pad);
    let y0 = min_y.saturating_sub(pad);
    let x1 = (max_x + pad).min(image.width() as usize - 1);
    let y1 = (max_y + pad).min(image.height() as usize - 1);
    (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
}

// the part of `region` (at x0, y0 on the picture) the bounds cover, which is what a patch is
fn cut_to_bounds(region: &RgbImage, (x0, y0): (usize, usize), bounds: Bounds) -> RgbImage {
    let (min_x, max_x, min_y, max_y) = bounds;
    imageops::crop_imm(
        region,
        (min_x - x0) as u32,
        (min_y - y0) as u32,
        (max_x - min_x + 1) as u32,
        (max_y - min_y + 1) as u32,
    )
    .to_image()
}

/// The brushed area deblurred by `session`, `strength` of the way. A blur many pixels wide is beyond what
/// the networks learned on, which was small pictures, so the area is first brought down by `scale` (1 for
/// a slight blur, less for a wide one), deblurred there, and only what the network changed is brought back
/// up and added to the full-size picture, so its own fine grain is kept.
pub fn deblur(
    image: &RgbImage,
    bounds: Bounds,
    session: &Mutex<Session>,
    scale: f32,
    strength: f32,
    tile: usize,
) -> Result<RgbImage> {
    let scale = scale.clamp(0.2, 1.0);
    let pad = (48.0 / scale).round() as usize;
    let (x0, y0, w, h) = padded(bounds, pad, image);
    let region = imageops::crop_imm(image, x0 as u32, y0 as u32, w as u32, h as u32).to_image();
    let full = Rgb32FImage::from_fn(w as u32, h as u32, |x, y| {
        Rgb(region.get_pixel(x, y).0.map(|v| v as f32 / 255.0))
    });
    let (sw, sh) = (
        ((w as f32 * scale).round() as u32).max(16),
        ((h as f32 * scale).round() as u32).max(16),
    );
    let small = if (sw, sh) == (w as u32, h as u32) {
        full.clone()
    } else {
        imageops::resize(&full, sw, sh, FilterType::Triangle)
    };
    let sharp = crate::ai::tiling::run_ai_deblur(&small, session, tile, &|_: f32| {})?;
    let mut change = small;
    change
        .par_iter_mut()
        .zip(sharp.as_raw().par_iter())
        .for_each(|(c, &s)| *c = s - *c);
    let change = if change.dimensions() == (w as u32, h as u32) {
        change
    } else {
        imageops::resize(&change, w as u32, h as u32, FilterType::Triangle)
    };
    let strength = strength.clamp(0.0, 1.0);
    let out = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let before = full.get_pixel(x, y).0;
        let delta = change.get_pixel(x, y).0;
        Rgb(std::array::from_fn(|c| {
            ((before[c] + delta[c] * strength).clamp(0.0, 1.0) * 255.0).round() as u8
        }))
    });
    Ok(cut_to_bounds(&out, (x0, y0), bounds))
}

// A grey min or max over a square of `radius`, done one way and then the other.
fn extreme(plane: &[f32], w: usize, h: usize, radius: usize, max: bool) -> Plane {
    let pick = |a: f32, b: f32| if max { a.max(b) } else { a.min(b) };
    let start = if max { f32::MIN } else { f32::MAX };
    let mut rows = vec![0.0; w * h];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let line = &plane[y * w..(y + 1) * w];
        for (x, o) in out.iter_mut().enumerate() {
            let (a, b) = (x.saturating_sub(radius), (x + radius).min(w - 1));
            *o = line[a..=b].iter().fold(start, |m, &v| pick(m, v));
        }
    });
    let mut out = vec![0.0; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
        let (a, b) = (y.saturating_sub(radius), (y + radius).min(h - 1));
        for (x, o) in line.iter_mut().enumerate() {
            *o = (a..=b).fold(start, |m, yy| pick(m, rows[yy * w + x]));
        }
    });
    out
}

const NEIGHBOURS: [(i32, i32); 8] = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (1, 1),
    (-1, 1),
    (1, -1),
];

// The pieces of `on` that touch, each as the list of its pixels.
fn components(on: &[bool], w: usize, h: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; on.len()];
    let mut found = Vec::new();
    for start in 0..on.len() {
        if !on[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let mut piece = Vec::new();
        while let Some(i) = stack.pop() {
            piece.push(i);
            let (x, y) = (i % w, i / w);
            for (dx, dy) in NEIGHBOURS {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let n = ny as usize * w + nx as usize;
                if on[n] && !seen[n] {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        found.push(piece);
    }
    found
}

// How long and thin a piece is: the spread along its main axis over the spread across it.
fn elongation(piece: &[usize], w: usize) -> (f32, f32) {
    let n = piece.len() as f32;
    let (mut mx, mut my) = (0.0f32, 0.0f32);
    for &i in piece {
        mx += (i % w) as f32;
        my += (i / w) as f32;
    }
    let (mx, my) = (mx / n, my / n);
    let (mut xx, mut yy, mut xy) = (0.0f32, 0.0f32, 0.0f32);
    for &i in piece {
        let (dx, dy) = ((i % w) as f32 - mx, (i / w) as f32 - my);
        xx += dx * dx;
        yy += dy * dy;
        xy += dx * dy;
    }
    let (xx, yy, xy) = (xx / n, yy / n, xy / n);
    let mean = (xx + yy) / 2.0;
    let root = (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt();
    let (along, across) = (mean + root, (mean - root).max(1e-3));
    // the length along the main axis, as a uniform line that long would spread
    ((along / across).sqrt(), (12.0 * along).sqrt())
}

/// The dust spots in the brushed area divided back out. A spot is found as a round, soft darkening of a
/// flat area, as dark in every colour, no wider than `size` pixels across; it is corrected by dividing
/// it by how much it darkens, so the sky or wall under it comes back as it was. `sensitivity` from 0 to 1
/// lets fainter spots count.
///
/// `size` is how wide a spot looks: a dust spot's soft edge reaches well past what the eye takes for its
/// edge, so the area looked at for each spot is 1.6 times as wide as that.
pub fn remove_dust(image: &RgbImage, bounds: Bounds, size: f32, sensitivity: f32) -> RgbImage {
    let radius = (size * 1.6 / 2.0).max(2.0);
    let pad = (radius * 3.0).ceil() as usize;
    let (x0, y0, w, h) = padded(bounds, pad, image);
    let region = imageops::crop_imm(image, x0 as u32, y0 as u32, w as u32, h as u32).to_image();

    // looked for on a copy small enough that a spot is a couple of dozen pixels across: smaller than that
    // and the shrinking blurs a spot's darkest point, which is then taken out only in part
    let down = (radius / 12.0).max(1.0);
    let (sw, sh) = (
        ((w as f32 / down).round() as usize).max(8),
        ((h as f32 / down).round() as usize).max(8),
    );
    let small = imageops::resize(&region, sw as u32, sh as u32, FilterType::Triangle);
    let channel = |c: usize| -> Plane { small.pixels().map(|p| to_linear(p.0[c])).collect() };
    let rgb = [channel(0), channel(1), channel(2)];
    let luma: Plane = (0..sw * sh)
        .map(|i| 0.2126 * rgb[0][i] + 0.7152 * rgb[1][i] + 0.0722 * rgb[2][i])
        .collect();

    // the area as it would be without its spots: a closing wider than a spot takes them out, a blur
    // smooths what the closing leaves
    let r = (radius / down).ceil() as usize + 1;
    let background_of = |plane: &[f32]| {
        let closed = extreme(&extreme(plane, sw, sh, r, true), sw, sh, r, false);
        blur(&closed, sw, sh, r as f32 / 2.0)
    };
    let background = background_of(&luma);
    // and each colour's own, which is what tells a grey spot from a coloured one
    let backgrounds: [Plane; 3] = std::array::from_fn(|c| background_of(&rgb[c]));
    let fine = blur(&luma, sw, sh, 0.8);
    // how much darker each place is than the area around it, as a share
    let dark: Plane = fine
        .iter()
        .zip(&background)
        .map(|(&v, &b)| (1.0 - v / b.max(1e-4)).max(0.0))
        .collect();
    // how busy the area is: dust only shows, and is only told apart, where it is flat
    let detail: Plane = luma
        .iter()
        .zip(&fine)
        .map(|(&v, &f)| (v - f).abs())
        .collect();
    let busy = blur(&detail, sw, sh, r as f32);
    let busy: Plane = busy
        .iter()
        .zip(&background)
        .map(|(&d, &b)| d / b.max(1e-4))
        .collect();

    let threshold = 0.004 + 0.02 * (1.0 - sensitivity.clamp(0.0, 1.0));
    let on: Vec<bool> = (0..sw * sh)
        .map(|i| dark[i] > threshold.max(busy[i] * 3.0))
        .collect();
    let mut spots = vec![0.0f32; sw * sh];
    for piece in components(&on, sw, sh) {
        let area = piece.len() as f32;
        let (ratio, _) = elongation(&piece, sw);
        let largest = std::f32::consts::PI * (r as f32).powi(2) * 1.5;
        if area < 3.0 || area > largest || ratio > 2.0 {
            continue;
        }
        // as dark in every colour: a bird or a leaf is not
        let share = |c: usize| {
            piece
                .iter()
                .map(|&i| 1.0 - rgb[c][i] / backgrounds[c][i].max(1e-4))
                .sum::<f32>()
                / area
        };
        let (red, green, blue) = (share(0), share(1), share(2));
        let mean = (red + green + blue) / 3.0;
        if mean <= 0.0
            || (red - blue).abs() > 0.5 * mean + 0.01
            || (green - mean).abs() > 0.4 * mean
        {
            continue;
        }
        for &i in &piece {
            spots[i] = 1.0;
        }
    }
    // the whole of each spot, its soft edge too
    let spots = blur(&extreme(&spots, sw, sh, 1, true), sw, sh, 1.0);
    // how much light each place lets through, 1 where there is no spot
    let pass: Plane = spots
        .iter()
        .zip(&dark)
        .map(|(&s, &d)| 1.0 - s.min(1.0) * d.min(0.6))
        .collect();
    let pass = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(sw as u32, sh as u32, pass).unwrap();
    let pass = imageops::resize(&pass, w as u32, h as u32, FilterType::Triangle);

    let out = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let t = pass.get_pixel(x, y).0[0].clamp(0.4, 1.0);
        Rgb(region.get_pixel(x, y).0.map(|v| to_code(to_linear(v) / t)))
    });
    cut_to_bounds(&out, (x0, y0), bounds)
}

// Fills the pixels `hole` marks from those around them, coarse to fine: every level of a pyramid is the
// average of the known pixels under it, and a hole takes the level above wherever it has nothing of its
// own. Smooth and without seams, which is what a sky needs.
fn push_pull(values: &mut [f32], known: &[f32], w: usize, h: usize) {
    let mut levels = vec![(values.to_vec(), known.to_vec(), w, h)];
    loop {
        let (v, k, lw, lh) = levels.last().unwrap();
        let (lw, lh) = (*lw, *lh);
        if lw <= 1 && lh <= 1 {
            break;
        }
        let (nw, nh) = (lw.div_ceil(2), lh.div_ceil(2));
        let mut nv = vec![0.0; nw * nh];
        let mut nk = vec![0.0; nw * nh];
        for y in 0..lh {
            for x in 0..lw {
                let (i, j) = (y * lw + x, (y / 2) * nw + x / 2);
                nv[j] += v[i] * k[i];
                nk[j] += k[i];
            }
        }
        for (value, weight) in nv.iter_mut().zip(nk.iter_mut()) {
            if *weight > 0.0 {
                *value /= *weight;
            }
            *weight = weight.min(1.0);
        }
        levels.push((nv, nk, nw, nh));
    }
    for level in (0..levels.len() - 1).rev() {
        let (upper, rest) = levels.split_at_mut(level + 1);
        let (v, k, lw, lh) = &mut upper[level];
        let (cv, _, cw, ch) = &rest[0];
        for y in 0..*lh {
            for x in 0..*lw {
                let i = y * *lw + x;
                // the coarser level, read between its pixels
                let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (*cw - 1) as f32);
                let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (*ch - 1) as f32);
                let (ax, ay) = (fx.floor() as usize, fy.floor() as usize);
                let (bx, by) = ((ax + 1).min(*cw - 1), (ay + 1).min(*ch - 1));
                let (tx, ty) = (fx - ax as f32, fy - ay as f32);
                let top = cv[ay * cw + ax] * (1.0 - tx) + cv[ay * cw + bx] * tx;
                let bottom = cv[by * cw + ax] * (1.0 - tx) + cv[by * cw + bx] * tx;
                let coarse = top * (1.0 - ty) + bottom * ty;
                v[i] = v[i] * k[i] + coarse * (1.0 - k[i]);
            }
        }
        k.iter_mut().for_each(|k| *k = 1.0);
    }
    values.copy_from_slice(&levels[0].0);
}

/// The wires in the brushed area filled from the sky around them. A wire is found as a long, thin, dark
/// line no wider than `width` pixels: a ridge in the brightness, bending sharply across and hardly at all
/// along, that runs on far enough to be a cable and not a twig. `sensitivity` from 0 to 1 lets fainter
/// lines count.
pub fn remove_wires(image: &RgbImage, bounds: Bounds, width: f32, sensitivity: f32) -> RgbImage {
    let width = width.clamp(1.0, 40.0);
    let pad = (width * 4.0).ceil() as usize + 4;
    let (x0, y0, w, h) = padded(bounds, pad, image);
    let region = imageops::crop_imm(image, x0 as u32, y0 as u32, w as u32, h as u32).to_image();
    let channels: [Plane; 3] =
        std::array::from_fn(|c| region.pixels().map(|p| to_linear(p.0[c])).collect());
    // the log of the brightness, so a line reads the same in a bright sky and a dull one
    let log: Plane = (0..w * h)
        .map(|i| {
            let y = 0.2126 * channels[0][i] + 0.7152 * channels[1][i] + 0.0722 * channels[2][i];
            y.max(1e-4).ln()
        })
        .collect();

    // how much each place looks like a dark line, at the widths a wire this wide shows at
    let mut ridge = vec![0.0f32; w * h];
    for sigma in [width * 0.2, width * 0.35, width * 0.5] {
        let sigma = sigma.max(0.7);
        let smooth = blur(&log, w, h, sigma);
        let at = |x: usize, y: usize| smooth[y.min(h - 1) * w + x.min(w - 1)];
        let norm = sigma * sigma;
        ridge.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let (xm, ym) = (x.saturating_sub(1), y.saturating_sub(1));
                let c = at(x, y);
                let xx = at(x + 1, y) + at(xm, y) - 2.0 * c;
                let yy = at(x, y + 1) + at(x, ym) - 2.0 * c;
                let xy = (at(x + 1, y + 1) + at(xm, ym) - at(x + 1, ym) - at(xm, y + 1)) / 4.0;
                let mean = (xx + yy) / 2.0;
                let root = (((xx - yy) / 2.0).powi(2) + xy * xy).sqrt();
                let (across, along) = (mean + root, mean - root);
                if across <= 0.0 {
                    continue;
                }
                // Frangi's measure: strong across, nearly nothing along
                let line = (-(along / across).powi(2) / 0.5).exp();
                let strength = 1.0 - (-(across * norm / 0.02).powi(2)).exp();
                *out = out.max(line * strength);
            }
        });
    }

    let high = 0.55 - 0.35 * sensitivity.clamp(0.0, 1.0);
    let low = high * 0.5;
    let candidates: Vec<bool> = ridge.iter().map(|&r| r > low).collect();
    let longest = (w.max(h) as f32 * 0.04).max(width * 6.0);
    let mut wires = vec![0.0f32; w * h];
    for piece in components(&candidates, w, h) {
        // kept if part of it is plainly a line, and it is long and thin
        if !piece.iter().any(|&i| ridge[i] > high) {
            continue;
        }
        let (ratio, length) = elongation(&piece, w);
        if length < longest || ratio < 6.0 {
            continue;
        }
        for &i in &piece {
            wires[i] = 1.0;
        }
    }
    // the whole width of each wire, and a little more for its soft edge
    let reach = (width / 2.0).ceil() as usize + 1;
    let hole = extreme(&wires, w, h, reach, true);
    let known: Plane = hole.iter().map(|&v| 1.0 - v).collect();

    let filled: Vec<Plane> = channels
        .iter()
        .map(|channel| {
            let mut values = channel.clone();
            push_pull(&mut values, &known, w, h);
            values
        })
        .collect();
    let blend = blur(&hole, w, h, 1.0);
    let out = RgbImage::from_fn(w as u32, h as u32, |x, y| {
        let i = y as usize * w + x as usize;
        let k = blend[i].clamp(0.0, 1.0);
        Rgb(std::array::from_fn(|c| {
            to_code(channels[c][i] * (1.0 - k) + filled[c][i] * k)
        }))
    });
    cut_to_bounds(&out, (x0, y0), bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sky(w: u32, h: u32) -> RgbImage {
        RgbImage::from_fn(w, h, |_, y| {
            let v = 150 + (y * 40 / h) as u8;
            Rgb([v - 30, v - 10, v])
        })
    }

    #[test]
    fn a_dust_spot_on_a_sky_is_taken_out() {
        let mut image = sky(200, 200);
        for (x, y, p) in image.enumerate_pixels_mut() {
            let d = ((x as f32 - 100.0).powi(2) + (y as f32 - 100.0).powi(2)).sqrt();
            let t = 1.0 - 0.12 * (-(d / 8.0).powi(2)).exp();
            *p = Rgb(p.0.map(|v| to_code(to_linear(v) * t)));
        }
        let before = image.get_pixel(100, 100).0[1] as i32;
        // beside the spot, on its own row: the sky is brighter further down, which is no part of the spot
        let around = image.get_pixel(140, 100).0[1] as i32;
        let out = remove_dust(&image, (0, 199, 0, 199), 30.0, 0.5);
        let after = out.get_pixel(100, 100).0[1] as i32;
        assert!(
            (after - around).abs() < (before - around).abs() / 2,
            "{before} {after} {around}"
        );
    }

    // The five soft spots of different sizes the practical test used, on a sky gradient: how much of each
    // is still there after the repair, for several settings.
    fn dusty_sky() -> (RgbImage, Vec<(u32, u32, f32)>) {
        let (w, h) = (2400u32, 1600u32);
        let spots: Vec<(u32, u32, f32)> = vec![
            (500, 300, 22.0),
            (1200, 500, 30.0),
            (1800, 350, 18.0),
            (900, 900, 26.0),
            (2000, 1000, 34.0),
        ];
        let mut dark = vec![0.0f32; (w * h) as usize];
        for &(cx, cy, r) in &spots {
            for y in cy.saturating_sub(r as u32 * 3)..(cy + r as u32 * 3).min(h) {
                for x in cx.saturating_sub(r as u32 * 3)..(cx + r as u32 * 3).min(w) {
                    let d = (((x as f32 - cx as f32).powi(2) + (y as f32 - cy as f32).powi(2))
                        .sqrt()
                        - r)
                        / 9.0;
                    // a disc with an edge softened over about 9 pixels, 27 % darker at its middle
                    let v = 0.5 * (1.0 - (d * 1.2).tanh());
                    dark[(y * w + x) as usize] = dark[(y * w + x) as usize].max(0.27 * v);
                }
            }
        }
        let image = RgbImage::from_fn(w, h, |x, y| {
            let t = y as f32 / h as f32;
            let k = 1.0 - dark[(y * w + x) as usize];
            let base = [115.0 + 64.0 * t, 158.0 + 46.0 * t, 230.0 - 25.0 * t];
            Rgb(base.map(|v| (v * k).round() as u8))
        });
        (image, spots)
    }

    #[test]
    #[ignore]
    fn dust_profile() {
        let mut image = sky(200, 200);
        for (x, y, p) in image.enumerate_pixels_mut() {
            let d = ((x as f32 - 100.0).powi(2) + (y as f32 - 100.0).powi(2)).sqrt();
            let t = 1.0 - 0.12 * (-(d / 8.0).powi(2)).exp();
            *p = Rgb(p.0.map(|v| to_code(to_linear(v) * t)));
        }
        let out = remove_dust(&image, (0, 199, 0, 199), 30.0, 0.5);
        let row = |img: &RgbImage| -> Vec<i32> {
            (60..=140)
                .step_by(5)
                .map(|x| img.get_pixel(x, 100).0[1] as i32)
                .collect()
        };
        println!("before {:?}", row(&image));
        println!("after  {:?}", row(&out));
    }

    #[test]
    #[ignore]
    fn dust_table() {
        let (image, spots) = dusty_sky();
        let darkening = |img: &RgbImage, (cx, cy, _): (u32, u32, f32)| {
            let l = |x: u32, y: u32| img.get_pixel(x, y).0[1] as f32;
            let centre = l(cx, cy);
            let ring = (l(cx - 110, cy) + l(cx + 110, cy)) / 2.0;
            ring - centre
        };
        let before: Vec<f32> = spots.iter().map(|&s| darkening(&image, s)).collect();
        for (size, sensitivity) in [(30.0, 0.5), (60.0, 0.5), (100.0, 0.5), (60.0, 1.0)] {
            let out = remove_dust(&image, (0, 2399, 0, 1599), size, sensitivity);
            let left: Vec<String> = spots
                .iter()
                .zip(&before)
                .map(|(&s, b)| format!("{:3.0}%", 100.0 * darkening(&out, s) / b))
                .collect();
            println!(
                "size {size:5} sensitivity {sensitivity}: left {}",
                left.join(" ")
            );
        }
    }

    #[test]
    fn a_wire_across_a_sky_is_filled() {
        let mut image = sky(240, 160);
        for x in 0..240 {
            let y = 60 + x / 8;
            for dy in 0..3 {
                image.put_pixel(x, y + dy, Rgb([40, 40, 45]));
            }
        }
        let out = remove_wires(&image, (0, 239, 0, 159), 6.0, 0.5);
        let on_wire = out.get_pixel(120, 76).0[2] as i32;
        let sky_there = sky(240, 160).get_pixel(120, 76).0[2] as i32;
        assert!((on_wire - sky_there).abs() < 25, "{on_wire} {sky_there}");
    }
}
