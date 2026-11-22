// Skin renewal: the "one-minute skin retouch" of a Photoshop action, with every number it asks a person to
// judge by eye worked out from the picture instead.
//
// By hand, the technique is:
//   1. a big edge-keeping blur (Median, about 120 px): the even base of the skin;
//   2. a made-up fine texture: noise on a 50 % grey overlay layer, blurred about 1.8 px;
//   3. the finest detail of the original, from a High Pass of the same radius, in Linear Light at about 70 %;
//   4. Blend If, to keep all of it off the highlights and the shadows;
//   5. a layer mask, painted with white by eye: where the skin is bad, lightly near edges.
//
// Here the face is already found, so the mask is known, and everything else is measured:
//   - the blur radius is the smallest that has evened the skin out, as long as its real structure (the nose,
//     the lip line, the beard's edge) is still there;
//   - the fine texture has the grain size and strength measured in the cleanest skin of this very face;
//   - the original's detail is soft-clipped, so a blemish or a hair is not carried across as "texture";
//   - the highlights and the shadows left alone are taken from the skin's own tones;
//   - the mask is a score of how uneven each place is next to its own evened-out self, kept off edges.
//
// Everything is worked out in a gamma-coded copy, which is what the hand technique does it in, and is kept as
// a change to add to the picture, so the amount can move without any of it being done again.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use image::{DynamicImage, GenericImageView, GrayImage, Rgb32FImage, imageops};
use rayon::prelude::*;

use crate::render::adjustment_utils::is_effect_enabled;

/// What the picture is changed by, over the face's box: each pixel's change at full strength, and where and
/// how much of it counts.
pub struct Layer {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    /// red, green, blue change, in the picture's own encoding
    delta: Vec<f32>,
    weight: Vec<f32>,
    pub report: String,
}

static LAYER: Mutex<Option<(u64, Arc<Layer>)>> = Mutex::new(None);
static BUILD: Mutex<()> = Mutex::new(());

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    is_effect_enabled(adjustments, "skin", "skinEnabled")
        && adjustments["skinRenewal"].as_f64().unwrap_or(0.0) > 0.0
        && adjustments["skinFaceMap"]
            .as_str()
            .is_some_and(|map| !map.is_empty())
}

pub fn amount(adjustments: &serde_json::Value) -> f32 {
    (adjustments["skinRenewal"].as_f64().unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0)
}

/// 0.5 is the texture as the skin has it; less is smoother, more is rougher.
fn texture_gain(adjustments: &serde_json::Value) -> f32 {
    (adjustments["skinRenewalTexture"].as_f64().unwrap_or(50.0) as f32 / 50.0).clamp(0.0, 2.0)
}

/// What the renewal depends on besides its amount, for the caches that must notice a change.
pub fn hash_inputs(adjustments: &serde_json::Value, hasher: &mut DefaultHasher) {
    let on = is_enabled(adjustments);
    on.hash(hasher);
    if on {
        adjustments["skinRenewal"].to_string().hash(hasher);
        adjustments["skinRenewalTexture"].to_string().hash(hasher);
        let map = adjustments["skinFaceMap"].as_str().unwrap_or("");
        map.len().hash(hasher);
        map.as_bytes()
            .iter()
            .step_by((map.len() / 512).max(1))
            .for_each(|b| b.hash(hasher));
    }
}

/// The picture with its skin renewed, or as it was when there is nothing to renew.
pub fn apply<'a>(
    base: Cow<'a, DynamicImage>,
    path: &str,
    is_raw: bool,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    if !is_enabled(adjustments) {
        return base;
    }
    let Some(face_map) = adjustments["skinFaceMap"].as_str() else {
        return base;
    };
    let Some(map) = crate::effects::maps::resolve_luma_map(face_map, adjustments, None) else {
        return base;
    };
    let (w, h) = base.dimensions();
    let mut key = DefaultHasher::new();
    (
        path,
        w,
        h,
        is_raw,
        crate::effects::maps::effect_map_key(face_map, adjustments),
        (texture_gain(adjustments) * 100.0) as i32,
    )
        .hash(&mut key);
    let key = key.finish();

    let layer = {
        let cached = LAYER.lock().unwrap_or_else(|e| e.into_inner());
        cached
            .as_ref()
            .filter(|(k, _)| *k == key)
            .map(|(_, l)| l.clone())
    };
    let layer = match layer {
        Some(layer) => layer,
        None => {
            let _building = BUILD.lock().unwrap_or_else(|e| e.into_inner());
            let again = LAYER
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .filter(|(k, _)| *k == key)
                .map(|(_, l)| l.clone());
            match again {
                Some(layer) => layer,
                None => {
                    let rgb = base.to_rgb32f();
                    let map = if map.dimensions() == (w, h) {
                        map
                    } else {
                        Arc::new(imageops::resize(
                            &*map,
                            w,
                            h,
                            imageops::FilterType::Triangle,
                        ))
                    };
                    let started = std::time::Instant::now();
                    let Some(layer) = build(&rgb, &map, is_raw, texture_gain(adjustments)) else {
                        return base;
                    };
                    log::info!(
                        "Skin renewal worked out in {:.1}s: {}",
                        started.elapsed().as_secs_f32(),
                        layer.report
                    );
                    let layer = Arc::new(layer);
                    *LAYER.lock().unwrap_or_else(|e| e.into_inner()) = Some((key, layer.clone()));
                    layer
                }
            }
        }
    };

    let amount = amount(adjustments);
    let mut out = base.to_rgb32f();
    let stride = w as usize * 3;
    out.par_chunks_mut(stride)
        .enumerate()
        .skip(layer.y0)
        .take(layer.h)
        .for_each(|(y, row)| {
            let ly = y - layer.y0;
            for lx in 0..layer.w {
                let wt = (layer.weight[ly * layer.w + lx] * amount).clamp(0.0, 1.0);
                if wt <= 0.0 {
                    continue;
                }
                let px = (layer.x0 + lx) * 3;
                let d = &layer.delta[(ly * layer.w + lx) * 3..][..3];
                for c in 0..3 {
                    row[px + c] += wt * d[c];
                }
            }
        });
    Cow::Owned(DynamicImage::ImageRgb32F(out))
}

// ---- the work -----------------------------------------------------------------------------------------------

/// One-channel float image, row by row.
#[derive(Clone)]
struct Plane {
    w: usize,
    h: usize,
    v: Vec<f32>,
}

impl Plane {
    fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Plane {
        Plane {
            w: self.w,
            h: self.h,
            v: self.v.par_iter().map(|&x| f(x)).collect(),
        }
    }

    fn zip(&self, other: &Plane, f: impl Fn(f32, f32) -> f32 + Sync) -> Plane {
        Plane {
            w: self.w,
            h: self.h,
            v: self
                .v
                .par_iter()
                .zip(other.v.par_iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        }
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A running mean over a window of `radius` either side, edges repeated.
fn box_h(src: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let inv = 1.0 / (2 * radius + 1) as f64;
    out.par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(o, line)| {
            let at = |i: isize| line[i.clamp(0, w as isize - 1) as usize] as f64;
            let mut sum = 0.0f64;
            for i in -(radius as isize)..=(radius as isize) {
                sum += at(i);
            }
            for (x, v) in o.iter_mut().enumerate() {
                *v = (sum * inv) as f32;
                sum += at(x as isize + radius as isize + 1) - at(x as isize - radius as isize);
            }
        });
    out
}

fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for (y, v) in col.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
}

fn box_blur(p: &Plane, radius: usize) -> Plane {
    if radius == 0 {
        return p.clone();
    }
    let a = box_h(&p.v, p.w, p.h, radius);
    let t = transpose(&a, p.w, p.h);
    let b = box_h(&t, p.h, p.w, radius);
    Plane {
        w: p.w,
        h: p.h,
        v: transpose(&b, p.h, p.w),
    }
}

fn kernel_blur(p: &Plane, sigma: f32) -> Plane {
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r)
        .map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / total).collect();
    let pass = |src: &[f32], w: usize, h: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w)
            .zip(src.par_chunks(w))
            .for_each(|(o, line)| {
                for (x, v) in o.iter_mut().enumerate() {
                    let mut s = 0.0;
                    for (j, kv) in k.iter().enumerate() {
                        let i = (x as isize + j as isize - r).clamp(0, w as isize - 1) as usize;
                        s += kv * line[i];
                    }
                    *v = s;
                }
            });
        out
    };
    let a = pass(&p.v, p.w, p.h);
    let t = transpose(&a, p.w, p.h);
    let b = pass(&t, p.h, p.w);
    Plane {
        w: p.w,
        h: p.h,
        v: transpose(&b, p.h, p.w),
    }
}

fn downsample(p: &Plane, step: usize) -> Plane {
    let (sw, sh) = ((p.w / step).max(1), (p.h / step).max(1));
    let mut v = vec![0.0f32; sw * sh];
    v.par_chunks_mut(sw).enumerate().for_each(|(sy, row)| {
        for (sx, o) in row.iter_mut().enumerate() {
            let (x0, y0) = (sx * p.w / sw, sy * p.h / sh);
            let (x1, y1) = (
                ((sx + 1) * p.w / sw).max(x0 + 1),
                ((sy + 1) * p.h / sh).max(y0 + 1),
            );
            let mut s = 0.0;
            for y in y0..y1 {
                for x in x0..x1 {
                    s += p.v[y * p.w + x];
                }
            }
            *o = s / ((x1 - x0) * (y1 - y0)) as f32;
        }
    });
    Plane { w: sw, h: sh, v }
}

fn upsample(p: &Plane, w: usize, h: usize) -> Plane {
    let mut v = vec![0.0f32; w * h];
    v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let fy = ((y as f32 + 0.5) * p.h as f32 / h as f32 - 0.5).clamp(0.0, (p.h - 1) as f32);
        let (y0, wy) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(p.h - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let fx = ((x as f32 + 0.5) * p.w as f32 / w as f32 - 0.5).clamp(0.0, (p.w - 1) as f32);
            let (x0, wx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(p.w - 1);
            let top = p.v[y0 * p.w + x0] * (1.0 - wx) + p.v[y0 * p.w + x1] * wx;
            let bot = p.v[y1 * p.w + x0] * (1.0 - wx) + p.v[y1 * p.w + x1] * wx;
            *o = top * (1.0 - wy) + bot * wy;
        }
    });
    Plane { w, h, v }
}

/// A Gaussian blur of any width: a kernel for small ones, and for wide ones the same blur on a shrunken copy.
fn gauss(p: &Plane, sigma: f32) -> Plane {
    if sigma < 0.3 {
        return p.clone();
    }
    if sigma <= 7.0 {
        return kernel_blur(p, sigma);
    }
    let step = (sigma / 4.0).floor().max(2.0) as usize;
    let small = downsample(p, step);
    let blurred = kernel_blur(&small, sigma / step as f32);
    upsample(&blurred, p.w, p.h)
}

/// A median over a square of `radius`, on a small picture.
fn median_small(p: &Plane, radius: usize) -> Plane {
    let mut v = vec![0.0f32; p.w * p.h];
    v.par_chunks_mut(p.w).enumerate().for_each(|(y, row)| {
        let mut window = Vec::with_capacity((2 * radius + 1).pow(2));
        for (x, o) in row.iter_mut().enumerate() {
            window.clear();
            for yy in y.saturating_sub(radius)..=(y + radius).min(p.h - 1) {
                for xx in x.saturating_sub(radius)..=(x + radius).min(p.w - 1) {
                    window.push(p.v[yy * p.w + xx]);
                }
            }
            let mid = window.len() / 2;
            *o = *window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        }
    });
    Plane { w: p.w, h: p.h, v }
}

/// Smooths `src` along the edges of `guide` (He, Sun and Tang's guided filter).
fn guided(guide: &Plane, src: &Plane, radius: usize, eps: f32) -> Plane {
    let mean_g = box_blur(guide, radius);
    let mean_s = box_blur(src, radius);
    let mean_gs = box_blur(&guide.zip(src, |g, s| g * s), radius);
    let mean_gg = box_blur(&guide.zip(guide, |g, g2| g * g2), radius);
    let a = Plane {
        w: guide.w,
        h: guide.h,
        v: (0..guide.v.len())
            .into_par_iter()
            .map(|i| {
                let var = mean_gg.v[i] - mean_g.v[i] * mean_g.v[i];
                (mean_gs.v[i] - mean_g.v[i] * mean_s.v[i]) / (var + eps)
            })
            .collect(),
    };
    let b = Plane {
        w: guide.w,
        h: guide.h,
        v: (0..guide.v.len())
            .into_par_iter()
            .map(|i| mean_s.v[i] - a.v[i] * mean_g.v[i])
            .collect(),
    };
    let (ma, mb) = (box_blur(&a, radius), box_blur(&b, radius));
    Plane {
        w: guide.w,
        h: guide.h,
        v: (0..guide.v.len())
            .into_par_iter()
            .map(|i| ma.v[i] * guide.v[i] + mb.v[i])
            .collect(),
    }
}

/// Photoshop's big median, made cheap: the median is taken on a copy one `step`th the size and brought back
/// up along the edges of the full-size luma.
fn big_median(p: &Plane, radius: f32, guide: &Plane) -> Plane {
    let step = 8usize.min((radius / 3.0).floor().max(1.0) as usize);
    let small = downsample(p, step);
    let med = median_small(&small, ((radius / step as f32).round() as usize).max(1));
    let up = upsample(&med, p.w, p.h);
    guided(guide, &up, step.max(4), 2e-4)
}

fn mean_where(p: &Plane, weight: &Plane) -> f32 {
    let (s, n) =
        p.v.par_iter()
            .zip(weight.v.par_iter())
            .map(|(&v, &w)| (v as f64 * w as f64, w as f64))
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (s / (n + 1e-9)) as f32
}

/// The `q` quantile of `p` where `weight` is on, from a sample of it.
fn quantile(p: &Plane, weight: &Plane, q: f32) -> f32 {
    let step = (p.v.len() / 150_000).max(1);
    let mut v: Vec<f32> =
        p.v.iter()
            .zip(&weight.v)
            .step_by(step)
            .filter(|(_, w)| **w > 0.5)
            .map(|(v, _)| *v)
            .collect();
    if v.is_empty() {
        return 0.0;
    }
    let i = ((v.len() - 1) as f32 * q) as usize;
    *v.select_nth_unstable_by(i, |a, b| a.total_cmp(b)).1
}

fn gradient(p: &Plane) -> Plane {
    let mut v = vec![0.0f32; p.w * p.h];
    v.par_chunks_mut(p.w).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 >= p.h {
            return;
        }
        for (x, o) in row.iter_mut().enumerate().take(p.w - 1).skip(1) {
            let gx = p.v[y * p.w + x + 1] - p.v[y * p.w + x - 1];
            let gy = p.v[(y + 1) * p.w + x] - p.v[(y - 1) * p.w + x];
            *o = gx.abs() + gy.abs();
        }
    });
    Plane { w: p.w, h: p.h, v }
}

/// Gaussian noise, the same every time: a made-up skin texture must not change from one render to the next.
fn white_noise(w: usize, h: usize) -> Plane {
    let mut v = vec![0.0f32; w * h];
    v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut s = 0x9E37_79B9_7F4A_7C15u64 ^ (y as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        for o in row.iter_mut() {
            let (a, b) = (next(), next());
            *o = ((-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()) as f32;
        }
    });
    Plane { w, h, v }
}

const GAMMA: f32 = 2.2;

/// The biggest face box, in pixels, that is renewed: about 16 million (4000 x 4000).
const MAX_PIXELS: usize = 16_000_000;

fn build(image: &Rgb32FImage, mask: &GrayImage, is_raw: bool, texture: f32) -> Option<Layer> {
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    // the face's box
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (usize::MAX, 0usize, usize::MAX, 0usize);
    let mut covered = 0usize;
    for (x, y, p) in mask.enumerate_pixels() {
        if p.0[0] > 127 {
            covered += 1;
            min_x = min_x.min(x as usize);
            max_x = max_x.max(x as usize);
            min_y = min_y.min(y as usize);
            max_y = max_y.max(y as usize);
        }
    }
    if covered < 4000 {
        return None;
    }
    let face = ((max_x - min_x) as f32)
        .max(0.6 * (max_y - min_y) as f32)
        .max(32.0);
    let margin = (0.12 * face) as usize;
    let (x0, y0) = (min_x.saturating_sub(margin), min_y.saturating_sub(margin));
    let (x1, y1) = ((max_x + margin).min(iw - 1), (max_y + margin).min(ih - 1));
    let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
    // The work keeps a few dozen planes of the face's box at once: a face filling a big picture would need
    // several gigabytes, so it is left alone, said in the log, until the work is made to fit.
    if w * h > MAX_PIXELS {
        log::warn!(
            "Skin renewal skipped: the face's box is {w}x{h}, more than the {MAX_PIXELS} pixels it is worked out on"
        );
        return None;
    }

    // the skin's own light: raw pictures are linear, and are scaled so that the skin sits mid-grey in the
    // gamma-coded copy, which is where blur, noise and detail behave as they do in the hand technique
    let at = |x: usize, y: usize| image.get_pixel((x0 + x) as u32, (y0 + y) as u32).0;
    let mask_plane = Plane {
        w,
        h,
        v: (0..w * h)
            .into_par_iter()
            .map(|i| mask.get_pixel((x0 + i % w) as u32, (y0 + i / w) as u32).0[0] as f32 / 255.0)
            .collect(),
    };
    let scale = if is_raw {
        let mut lumas: Vec<f32> = (0..w * h)
            .step_by(((w * h) / 100_000).max(1))
            .filter(|&i| mask_plane.v[i] > 0.5)
            .map(|i| {
                let p = at(i % w, i / w);
                0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
            })
            .collect();
        if lumas.len() < 50 {
            return None;
        }
        let mid = lumas.len() / 2;
        let median = *lumas.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        (0.5f32.powf(GAMMA) / median.max(1e-5)).clamp(1.0, 2000.0)
    } else {
        1.0
    };
    let to_gamma = |v: f32| -> f32 {
        if is_raw {
            (v * scale).clamp(0.0, 1.0).powf(1.0 / GAMMA)
        } else {
            v.clamp(0.0, 1.0)
        }
    };
    let from_gamma = |g: f32| -> f32 {
        if is_raw {
            g.clamp(0.0, 1.0).powf(GAMMA) / scale
        } else {
            g
        }
    };
    let chan: [Plane; 3] = std::array::from_fn(|c| Plane {
        w,
        h,
        v: (0..w * h)
            .into_par_iter()
            .map(|i| to_gamma(at(i % w, i / w)[c]))
            .collect(),
    });
    let luma = |c: &[Plane; 3]| -> Plane {
        Plane {
            w,
            h,
            v: (0..w * h)
                .into_par_iter()
                .map(|i| 0.2126 * c[0].v[i] + 0.7152 * c[1].v[i] + 0.0722 * c[2].v[i])
                .collect(),
        }
    };
    let y = luma(&chan);
    let m = mask_plane;

    // skin well inside the mask, and the edges that are really there in it
    let interior = gauss(&m, 0.01 * face).map(|v| if v > 0.95 { 1.0 } else { 0.0 });
    if interior.v.iter().filter(|&&v| v > 0.5).count() < 2000 {
        return None;
    }
    let grad0 = gradient(&gauss(&y, 0.006 * face));
    let g85 = quantile(&grad0, &interior, 0.85);
    let structure = grad0.zip(&interior, |g, i| if g > g85 { i } else { 0.0 });
    let structure_grad = mean_where(&grad0, &structure).max(1e-9);

    // the base: the smallest radius that has evened out the skin, with its structure still in it
    let mut tried: Vec<(f32, f32, f32)> = Vec::new(); // radius, unevenness, structure kept
    for share in [0.02f32, 0.03, 0.045, 0.07, 0.10] {
        let radius = share * face;
        let base_y = big_median(&y, radius, &y);
        let mid = base_y.zip(&gauss(&base_y, 0.15 * face), |a, b| (a - b).abs());
        let unevenness = mean_where(&mid, &interior);
        let kept =
            mean_where(&gradient(&gauss(&base_y, 0.006 * face)), &structure) / structure_grad;
        tried.push((share, unevenness, kept));
    }
    let usable: Vec<&(f32, f32, f32)> = tried.iter().filter(|t| t.2 >= 0.6).collect();
    let usable: Vec<&(f32, f32, f32)> = if usable.is_empty() {
        vec![&tried[0]]
    } else {
        usable
    };
    let target = usable.last().unwrap().1 + 0.25 * (tried[0].1 - usable.last().unwrap().1);
    let chosen = *usable
        .iter()
        .find(|t| t.1 <= target)
        .unwrap_or_else(|| usable.last().unwrap());
    let radius = chosen.0 * face;

    let base: [Plane; 3] = std::array::from_fn(|c| big_median(&chan[c], radius, &y));
    let base_y = luma(&base);

    // where the skin is uneven, next to its own evened-out self
    let sig_mid = (0.004 * face).max(1.5);
    let lum_dev = gauss(&y, sig_mid).zip(&base_y, |a, b| (a - b).abs());
    let chroma_dev = {
        let rg = gauss(&chan[0].zip(&chan[1], |r, g| r - g), sig_mid);
        let bg = gauss(&chan[2].zip(&chan[1], |b, g| b - g), sig_mid);
        let brg = base[0].zip(&base[1], |r, g| r - g);
        let bbg = base[2].zip(&base[1], |b, g| b - g);
        Plane {
            w,
            h,
            v: (0..w * h)
                .into_par_iter()
                .map(|i| ((rg.v[i] - brg.v[i]).powi(2) + (bg.v[i] - bbg.v[i]).powi(2)).sqrt())
                .collect(),
        }
    };
    let pore = gauss(
        &y.zip(&gauss(&y, 0.0035 * face), |a, b| (a - b).abs()),
        0.006 * face,
    );
    let rel = |p: &Plane| quantile(p, &interior, 0.5).max(1e-6);
    let (rl, rc, rp) = (rel(&lum_dev), rel(&chroma_dev), rel(&pore));
    let score = Plane {
        w,
        h,
        v: (0..w * h)
            .into_par_iter()
            .map(|i| 0.45 * lum_dev.v[i] / rl + 0.40 * chroma_dev.v[i] / rc + 0.15 * pore.v[i] / rp)
            .collect(),
    };
    let score = gauss(&score, 0.015 * face);
    let (lo, hi) = (
        quantile(&score, &interior, 0.25),
        quantile(&score, &interior, 0.92),
    );
    let need_raw = score.map(|s| smoothstep(lo, hi, s));
    let need = need_raw.map(|n| 0.4 + 0.6 * n);

    // edges and the mask's border let go of the effect
    // only the strongest edges count, those of the nose, the lips and the eyes: a blemish's own edge is not
    // an edge to keep, it is the very thing being taken out
    let g99 = quantile(&grad0, &interior, 0.99).max(1e-9);
    let guard = gauss(&grad0.map(|g| (g / g99).clamp(0.0, 1.0)), 0.004 * face)
        .map(|e| 1.0 - smoothstep(0.3, 0.9, e));
    let border = gauss(&m, 0.012 * face).map(|b| smoothstep(0.0, 1.0, b));

    // highlights and shadows left alone, from the skin's own tones. They are judged on the evened-out skin
    // and not on each pixel: a dark spot on mid-tone skin is a blemish, not a shadow, and a shadow is dark
    // all around.
    let (hi0, hi1, sh1, sh0) = (
        quantile(&base_y, &interior, 0.93),
        quantile(&base_y, &interior, 0.995),
        quantile(&base_y, &interior, 0.04),
        quantile(&base_y, &interior, 0.002),
    );
    let tone = base_y.map(|v| (1.0 - smoothstep(hi0, hi1, v)) * smoothstep(sh0, sh1, v));

    // the fine texture: its strength and grain size measured in the cleanest skin of this face
    let sig_hp = (0.0016 * face).max(0.7);
    let detail = y.zip(&gauss(&y, sig_hp), |a, b| a - b);
    let clean = Plane {
        w,
        h,
        v: (0..w * h)
            .into_par_iter()
            .map(|i| {
                if need_raw.v[i] < 0.2 && guard.v[i] > 0.8 && interior.v[i] > 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    };
    let clean = if clean.v.iter().sum::<f32>() < 2000.0 {
        interior.clone()
    } else {
        clean
    };
    let s_clean = mean_where(&detail.map(|d| d * d), &clean).sqrt().max(1e-5);
    let dd: Vec<f32> = detail.v.iter().zip(&clean.v).map(|(d, c)| d * c).collect();
    let power: f64 = dd
        .par_iter()
        .map(|&d| (d as f64) * (d as f64))
        .sum::<f64>()
        .max(1e-12);
    let half_lag = (1..=8usize)
        .find(|&lag| {
            let s: f64 = (0..h)
                .into_par_iter()
                .map(|yy| {
                    let row = &dd[yy * w..(yy + 1) * w];
                    (lag..w)
                        .map(|x| row[x] as f64 * row[x - lag] as f64)
                        .sum::<f64>()
                })
                .sum();
            s / power < 0.5
        })
        .unwrap_or(8);
    let sig_tex = (half_lag as f32 * 0.85).clamp(0.8, 4.0);
    let noise = gauss(&white_noise(w, h), sig_tex);
    let (nm, nv) = {
        let n = noise.v.len() as f64;
        let mean = noise.v.iter().map(|&v| v as f64).sum::<f64>() / n;
        let var = noise
            .v
            .iter()
            .map(|&v| (v as f64 - mean).powi(2))
            .sum::<f64>()
            / n;
        (mean as f32, (var.sqrt() as f32).max(1e-9))
    };
    let tex_amp = s_clean * texture / nv;

    // the renewed skin: the base, a fresh texture, and the original's finest detail, soft-clipped
    let clip = 2.5 * s_clean;
    let mut delta = vec![0.0f32; w * h * 3];
    let mut weight = vec![0.0f32; w * h];
    delta
        .par_chunks_mut(3)
        .zip(weight.par_iter_mut())
        .enumerate()
        .for_each(|(i, (d, wt))| {
            let fine =
                0.55 * (noise.v[i] - nm) * tex_amp + 0.55 * clip * (detail.v[i] / clip).tanh();
            for c in 0..3 {
                let renewed = from_gamma(base[c].v[i] + fine);
                // a clipped pixel is changed from where the gamma-coded copy had it
                d[c] = renewed - from_gamma(chan[c].v[i]);
            }
            *wt = need.v[i] * guard.v[i] * border.v[i] * tone.v[i];
        });

    let report = format!(
        "face {:.0} px, median radius {:.0} px ({:.1}% of the face, {} candidates), pore grain {:.1} px, texture {:.4}, \
         highlights from {:.2}, shadows to {:.2}",
        face,
        radius,
        chosen.0 * 100.0,
        tried.len(),
        sig_tex,
        s_clean,
        hi0,
        sh1
    );
    Some(Layer {
        x0,
        y0,
        w,
        h,
        delta,
        weight,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    // A flat skin tone with fine grain, one soft blotch, one dark spot and a hard edge down the middle.
    fn skin() -> (Rgb32FImage, GrayImage) {
        let (w, h) = (360u32, 420u32);
        let noise = white_noise(w as usize, h as usize);
        let mut img = Rgb32FImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let g = noise.v[(y * w + x) as usize] * 0.004;
            let blotch = (-(((x as f32 - 90.0).powi(2) + (y as f32 - 120.0).powi(2))
                / (2.0 * 28.0 * 28.0)))
                .exp();
            let spot = (-(((x as f32 - 250.0).powi(2) + (y as f32 - 150.0).powi(2))
                / (2.0 * 5.0 * 5.0)))
                .exp();
            let edge = if x > 300 { -0.05 } else { 0.0 };
            let (r, gr, b) = (
                0.18 + g + 0.04 * blotch - 0.03 * spot + edge,
                0.12 + g - 0.02 * blotch - 0.045 * spot + edge,
                0.09 + g - 0.03 * blotch - 0.045 * spot + edge,
            );
            p.0 = [r, gr, b];
        }
        let mask = GrayImage::from_fn(w, h, |x, y| {
            Luma([if (10..350).contains(&x) && (10..410).contains(&y) {
                255
            } else {
                0
            }])
        });
        (img, mask)
    }

    #[test]
    fn a_spot_and_a_blotch_are_evened_and_an_edge_is_kept() {
        let (img, mask) = skin();
        let layer = build(&img, &mask, true, 1.0).expect("a layer");
        let at = |x: u32, y: u32| (y as usize - layer.y0) * layer.w + (x as usize - layer.x0);
        let renewed = |x: u32, y: u32, c: usize| {
            let i = at(x, y);
            img.get_pixel(x, y).0[c] + layer.weight[i] * layer.delta[i * 3 + c]
        };
        // the spot is darker than the skin by 0.045 in green; it should be closer to the skin after
        let skin_green = 0.12;
        let before = (img.get_pixel(250, 150).0[1] - skin_green).abs();
        let after = (renewed(250, 150, 1) - skin_green).abs();
        assert!(
            after < before * 0.7,
            "spot {before} -> {after}, weight {} change {}: {}",
            layer.weight[at(250, 150)],
            layer.delta[at(250, 150) * 3 + 1],
            layer.report
        );
        // the hard edge at x = 300 stays: the two sides are still about 0.05 apart
        let step = renewed(296, 300, 1) - renewed(310, 300, 1);
        assert!(step > 0.03, "edge {step}: {}", layer.report);
        // far from anything, the weight is not nothing and not everything
        assert!(layer.weight[at(180, 300)] > 0.0);
    }

    /// Runs the renewal on a linear float picture (RRAW_PHOTO: raw f32 RGB rows, RRAW_SIZE: "width,height") with
    /// its skin mask (RRAW_MASK: a grey PNG) and writes the result, as it would look at the amount RRAW_AMOUNT,
    /// to RRAW_OUT, with the weight beside it in RRAW_OUT + ".weight.png".
    #[test]
    #[ignore]
    fn renewal_on_a_picture() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let size: Vec<u32> = var("RRAW_SIZE")
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect();
        let (w, h) = (size[0], size[1]);
        let bytes = std::fs::read(var("RRAW_PHOTO")).unwrap();
        let data: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let img = Rgb32FImage::from_raw(w, h, data).expect("size matches");
        let mask = image::open(var("RRAW_MASK")).unwrap().to_luma8();
        let amount: f32 = std::env::var("RRAW_AMOUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let started = std::time::Instant::now();
        let layer = build(&img, &mask, true, 1.0).expect("a layer");
        println!(
            "built in {:.1}s: {}",
            started.elapsed().as_secs_f32(),
            layer.report
        );
        let view = |v: f32| ((v * 3.2).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        let mut out = image::RgbImage::new(w, h);
        let mut weights = image::GrayImage::new(w, h);
        for (x, y, p) in out.enumerate_pixels_mut() {
            let px = img.get_pixel(x, y).0;
            let (lx, ly) = (x as usize, y as usize);
            let inside = lx >= layer.x0
                && lx < layer.x0 + layer.w
                && ly >= layer.y0
                && ly < layer.y0 + layer.h;
            let i = (ly.wrapping_sub(layer.y0)) * layer.w + lx.wrapping_sub(layer.x0);
            let wt = if inside {
                (layer.weight[i] * amount).clamp(0.0, 1.0)
            } else {
                0.0
            };
            for (c, (out_c, in_c)) in p.0.iter_mut().zip(px).enumerate() {
                let d = if inside { layer.delta[i * 3 + c] } else { 0.0 };
                *out_c = view(in_c + wt * d);
            }
            weights.put_pixel(x, y, Luma([(wt * 255.0) as u8]));
        }
        out.save(var("RRAW_OUT")).unwrap();
        weights
            .save(format!("{}.weight.png", var("RRAW_OUT")))
            .unwrap();
    }

    #[test]
    fn nothing_changes_when_the_amount_is_nothing() {
        let adj = serde_json::json!({ "skinEnabled": true, "skinRenewal": 0, "skinFaceMap": "x" });
        assert!(!is_enabled(&adj));
    }
}
