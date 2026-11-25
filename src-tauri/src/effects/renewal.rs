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
//   - the median is Photoshop's: the middle value of each channel over a round neighbourhood, at full size;
//   - the fine texture has the grain size and strength measured in the cleanest skin of this very face, and is
//     laid on in Overlay; the High Pass has the grain's radius and goes on in Linear Light at 72 %, as in the
//     tutorial;
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
    /// the skin's middle brightness, in the picture's own encoding: what a map shown over it is scaled to
    skin_level: f32,
    pub report: String,
}

static LAYER: Mutex<Option<(u64, Arc<Layer>)>> = Mutex::new(None);
static BUILD: Mutex<()> = Mutex::new(());

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    is_effect_enabled(adjustments, "skin", "skinEnabled")
        && super::skin_method::Method::of(adjustments) == super::skin_method::Method::Renewal
        && (adjustments["skinRenewal"].as_f64().unwrap_or(0.0) > 0.0
            || super::skin_method::shown_map(adjustments) == super::skin_method::MAP_RENEWAL)
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
pub fn hash_inputs(adjustments: &serde_json::Value, hasher: &mut DefaultHasher, with_map: bool) {
    let on = is_enabled(adjustments);
    on.hash(hasher);
    if on {
        adjustments["skinRenewal"].to_string().hash(hasher);
        adjustments["skinRenewalTexture"].to_string().hash(hasher);
        if with_map {
            super::skin_method::shown_map(adjustments).hash(hasher);
        }
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
    let show = super::skin_method::shown_map(adjustments) == super::skin_method::MAP_RENEWAL;
    let mut out = base.to_rgb32f();
    let stride = w as usize * 3;
    out.par_chunks_mut(stride)
        .enumerate()
        .skip(layer.y0)
        .take(layer.h)
        .for_each(|(y, row)| {
            let ly = y - layer.y0;
            for lx in 0..layer.w {
                let px = (layer.x0 + lx) * 3;
                if show {
                    // where the renewal goes and how far, whatever the amount: the weight it is laid on with
                    let here = [row[px], row[px + 1], row[px + 2]];
                    let shown = super::skin_method::tint(
                        here,
                        layer.weight[ly * layer.w + lx],
                        layer.skin_level,
                    );
                    row[px..px + 3].copy_from_slice(&shown);
                    continue;
                }
                let wt = (layer.weight[ly * layer.w + lx] * amount).clamp(0.0, 1.0);
                if wt <= 0.0 {
                    continue;
                }
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
pub(super) struct Plane {
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) v: Vec<f32>,
}

impl Plane {
    pub(super) fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Plane {
        Plane {
            w: self.w,
            h: self.h,
            v: self.v.par_iter().map(|&x| f(x)).collect(),
        }
    }

    pub(super) fn zip(&self, other: &Plane, f: impl Fn(f32, f32) -> f32 + Sync) -> Plane {
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

pub(super) fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a).max(1e-9)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The "OG texture" layer's opacity in the tutorial ("72 to 74 %").
const DETAIL_OPACITY: f32 = 0.72;

/// Photoshop's Overlay of `top` on `bottom`, both 0..1.
fn overlay(bottom: f32, top: f32) -> f32 {
    let top = top.clamp(0.0, 1.0);
    if bottom < 0.5 {
        2.0 * bottom * top
    } else {
        1.0 - 2.0 * (1.0 - bottom) * (1.0 - top)
    }
}

/// Photoshop's Linear Light of `top` on `bottom`: the top's distance from mid-grey, doubled, added.
fn linear_light(bottom: f32, top: f32) -> f32 {
    (bottom + 2.0 * top - 1.0).clamp(0.0, 1.0)
}

pub(super) fn transpose(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(h).enumerate().for_each(|(x, col)| {
        for (y, v) in col.iter_mut().enumerate() {
            *v = src[y * w + x];
        }
    });
    out
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
pub(super) fn gauss(p: &Plane, sigma: f32) -> Plane {
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

/// Levels of the median's histogram over 0..1. The median is read inside its level, so it is finer than that.
const MEDIAN_BINS: usize = 1024;

/// Photoshop's Filter > Noise > Median: for every pixel, the middle value of the pixels within `radius` of it
/// (a round neighbourhood), the edge pixels repeated beyond the picture. Run on one channel at a time, as
/// Photoshop does on each of red, green and blue.
///
/// Huang's sliding histogram: one step along a row takes the circle's left edge out and puts its new right edge
/// in, so the cost grows with the radius and not with its square.
fn median(p: &Plane, radius: f32) -> Plane {
    let (w, h) = (p.w, p.h);
    let radius = radius.max(1.0);
    let r = radius.floor() as isize;
    // the circle's half-width on each of its rows
    let half: Vec<isize> = (-r..=r)
        .map(|dy| (radius * radius - (dy * dy) as f32).max(0.0).sqrt().floor() as isize)
        .collect();
    let count: u32 = half.iter().map(|&hw| (2 * hw + 1) as u32).sum();
    // the median is the value with `mid` values under it
    let mid = count / 2;
    let levels: Vec<u16> = p
        .v
        .par_iter()
        .map(|&v| ((v.clamp(0.0, 1.0) * MEDIAN_BINS as f32) as usize).min(MEDIAN_BINS - 1) as u16)
        .collect();
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let lines: Vec<&[u16]> = (-r..=r)
            .map(|dy| {
                let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                &levels[yy * w..(yy + 1) * w]
            })
            .collect();
        let at = |line: &[u16], x: isize| line[x.clamp(0, w as isize - 1) as usize] as usize;
        let mut hist = vec![0u32; MEDIAN_BINS];
        for (line, &hw) in lines.iter().zip(&half) {
            for x in -hw..=hw {
                hist[at(line, x)] += 1;
            }
        }
        // m: the median's level; below: how many values lie under it
        let (mut m, mut below) = (0usize, 0u32);
        for (x, o) in row.iter_mut().enumerate() {
            while below > mid {
                m -= 1;
                below -= hist[m];
            }
            while below + hist[m] <= mid {
                below += hist[m];
                m += 1;
            }
            *o = (m as f32 + ((mid - below) as f32 + 0.5) / hist[m] as f32) / MEDIAN_BINS as f32;
            if x + 1 < w {
                let x = x as isize;
                for (line, &hw) in lines.iter().zip(&half) {
                    let gone = at(line, x - hw);
                    hist[gone] -= 1;
                    if gone < m {
                        below -= 1;
                    }
                    let new = at(line, x + 1 + hw);
                    hist[new] += 1;
                    if new < m {
                        below += 1;
                    }
                }
            }
        }
    });
    Plane { w, h, v: out }
}

pub(super) fn mean_where(p: &Plane, weight: &Plane) -> f32 {
    let (s, n) =
        p.v.par_iter()
            .zip(weight.v.par_iter())
            .map(|(&v, &w)| (v as f64 * w as f64, w as f64))
            .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (s / (n + 1e-9)) as f32
}

/// The `q` quantile of `p` where `weight` is on, from a sample of it.
pub(super) fn quantile(p: &Plane, weight: &Plane, q: f32) -> f32 {
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

pub(super) fn gradient(p: &Plane) -> Plane {
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
pub(super) fn white_noise(w: usize, h: usize) -> Plane {
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

/// The narrowest high pass the grain is measured through, in pixels: wider than the coarsest grain used (4 px).
pub(super) const GRAIN_PROBE_MIN: f32 = 4.0;
/// The farthest shift, in pixels, the grain's likeness to itself is followed to.
const GRAIN_MAX_LAG: usize = 16;

/// How alike `detail` is to itself shifted by 0, 1, 2… pixels, across and down averaged, counting only pairs
/// of pixels that are both in `mask`; 1 at no shift.
fn grain_curve(detail: &Plane, mask: &Plane) -> Vec<f32> {
    let (w, h) = (detail.w, detail.h);
    let at = |x: usize, y: usize| (detail.v[y * w + x] as f64, mask.v[y * w + x] > 0.5);
    let along = |lag: usize, down: bool| -> f64 {
        let (s, a, b) = (0..h)
            .into_par_iter()
            .map(|y| {
                let mut acc = (0.0f64, 0.0f64, 0.0f64);
                if down && y + lag >= h {
                    return acc;
                }
                for x in 0..w {
                    if !down && x + lag >= w {
                        break;
                    }
                    let (d0, in0) = at(x, y);
                    let (d1, in1) = if down { at(x, y + lag) } else { at(x + lag, y) };
                    if in0 && in1 {
                        acc.0 += d0 * d1;
                        acc.1 += d0 * d0;
                        acc.2 += d1 * d1;
                    }
                }
                acc
            })
            .reduce(|| (0.0, 0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1, p.2 + q.2));
        s / (a * b).sqrt().max(1e-18)
    };
    (0..=GRAIN_MAX_LAG)
        .map(|lag| {
            if lag == 0 {
                1.0
            } else {
                (0.5 * (along(lag, false) + along(lag, true))) as f32
            }
        })
        .collect()
}

/// The shift, in pixels and between whole ones, at which the detail is half as alike to itself as unshifted.
pub(super) fn grain_half_width(detail: &Plane, mask: &Plane) -> f32 {
    let curve = grain_curve(detail, mask);
    for lag in 1..curve.len() {
        if curve[lag] < 0.5 {
            let (a, b) = (curve[lag - 1], curve[lag]);
            return (lag - 1) as f32 + (a - 0.5) / (a - b).max(1e-6);
        }
    }
    GRAIN_MAX_LAG as f32
}

/// Made-up grains, white noise of unit strength blurred by each width, measured through the high pass `probe`:
/// (blur, half width, strength left after the high pass).
pub(super) fn grain_table(probe: f32) -> Vec<(f32, f32, f32)> {
    let side = 320usize;
    let edge = (3.0 * probe).ceil() as usize + GRAIN_MAX_LAG;
    let inside = Plane {
        w: side,
        h: side,
        v: (0..side * side)
            .map(|i| {
                let (x, y) = (i % side, i / side);
                let ok = x >= edge && y >= edge && x + edge < side && y + edge < side;
                if ok { 1.0 } else { 0.0 }
            })
            .collect(),
    };
    let white = white_noise(side, side);
    let mut table: Vec<(f32, f32, f32)> = (0..=18)
        .map(|i| {
            let sigma = 0.5 + 0.25 * i as f32;
            let g = gauss(&white, sigma);
            let sd = mean_where(&g.map(|v| v * v), &inside).sqrt().max(1e-9);
            let g = g.map(|v| v / sd);
            let d = g.zip(&gauss(&g, probe), |a, b| a - b);
            let rms = mean_where(&d.map(|v| v * v), &inside).sqrt();
            (sigma, grain_half_width(&d, &inside), rms)
        })
        .collect();
    // a wider blur never measures narrower, even where the probe flattens the curve
    for i in 1..table.len() {
        table[i].1 = table[i].1.max(table[i - 1].1 + 1e-4);
    }
    table
}

/// Reads `out` against `key` in the table, between its rows (both increasing with the row).
fn interpolate(
    table: &[(f32, f32, f32)],
    key: impl Fn(&(f32, f32, f32)) -> f32,
    at: f32,
    out: impl Fn(&(f32, f32, f32)) -> f32,
) -> f32 {
    if at <= key(&table[0]) {
        return out(&table[0]);
    }
    for pair in table.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if at <= key(b) {
            let t = (at - key(a)) / (key(b) - key(a)).max(1e-9);
            return out(a) + t * (out(b) - out(a));
        }
    }
    out(&table[table.len() - 1])
}

/// The blur whose made-up grain measures `half_width`.
pub(super) fn blur_for_width(table: &[(f32, f32, f32)], half_width: f32) -> f32 {
    interpolate(table, |r| r.1, half_width, |r| r.0)
}

/// What is left, through the high pass, of a grain of unit strength blurred by `blur`.
fn strength_for_blur(table: &[(f32, f32, f32)], blur: f32) -> f32 {
    interpolate(table, |r| r.0, blur, |r| r.2).max(1e-6)
}

const GAMMA: f32 = 2.2;

/// Over how many median radii the face's light is taken back: half a radius brings back the creases and
/// the contours of the cheeks and lips the median softens, while blemishes, narrower still, stay evened.
/// Chosen by eye on six portraits against wider spreads, which left the faces looking flatter.
const VOLUME_SPREAD: f32 = 0.5;

/// The choices that shape how much of the face's own light the renewal keeps.
#[derive(Clone, Copy, Debug)]
pub(super) struct Tuning {
    /// how far, in median radii, the light the median took is given back over; 0 gives none back
    pub spread: f32,
    /// highlights judged on each place's own light, so a shine narrower than the median is kept
    pub local_highlights: bool,
}

const TUNING: Tuning = Tuning {
    spread: VOLUME_SPREAD,
    local_highlights: false,
};

/// The biggest face box, in pixels, that is worked out at full size: about 16 million (4000 x 4000). A bigger
/// one is worked out on a reduced copy.
const MAX_PIXELS: usize = 16_000_000;

/// The picture reduced by `k`, each pixel the mean of the k x k it stands for.
fn shrink_rgb(image: &Rgb32FImage, k: usize) -> Rgb32FImage {
    let (sw, sh) = (image.width() as usize / k, image.height() as usize / k);
    let mut out = Rgb32FImage::new(sw as u32, sh as u32);
    let n = (k * k) as f32;
    out.par_chunks_mut(sw * 3).enumerate().for_each(|(y, row)| {
        for x in 0..sw {
            let mut sum = [0.0f32; 3];
            for dy in 0..k {
                for dx in 0..k {
                    let p = image.get_pixel((x * k + dx) as u32, (y * k + dy) as u32).0;
                    for c in 0..3 {
                        sum[c] += p[c];
                    }
                }
            }
            for c in 0..3 {
                row[x * 3 + c] = sum[c] / n;
            }
        }
    });
    out
}

fn shrink_mask(mask: &GrayImage, k: usize) -> GrayImage {
    let (sw, sh) = (mask.width() as usize / k, mask.height() as usize / k);
    let mut out = GrayImage::new(sw as u32, sh as u32);
    out.par_chunks_mut(sw).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let mut sum = 0u32;
            for dy in 0..k {
                for dx in 0..k {
                    sum += mask.get_pixel((x * k + dx) as u32, (y * k + dy) as u32).0[0] as u32;
                }
            }
            *v = (sum / (k * k) as u32) as u8;
        }
    });
    out
}

/// A layer worked out at 1/k of the size, brought to a picture of `iw` x `ih`: its change and its weight
/// are interpolated between the reduced pixels' centres.
fn enlarge(small: &Layer, k: usize, iw: usize, ih: usize) -> Layer {
    let x0 = small.x0 * k;
    let y0 = small.y0 * k;
    let w = (small.w * k).min(iw - x0);
    let h = (small.h * k).min(ih - y0);
    // where a full-size pixel falls in the reduced layer: its two neighbours on each axis and the share of
    // the second
    let place = |p: usize, origin: usize, len: usize| -> (usize, usize, f32) {
        let s = ((origin + p) as f32 + 0.5) / k as f32 - 0.5 - (origin / k) as f32;
        let s = s.clamp(0.0, (len - 1) as f32);
        let a = s.floor() as usize;
        (a, (a + 1).min(len - 1), s - a as f32)
    };
    let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| place(x, x0, small.w)).collect();
    let mut delta = vec![0.0f32; w * h * 3];
    let mut weight = vec![0.0f32; w * h];
    delta
        .par_chunks_mut(w * 3)
        .zip(weight.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (drow, wrow))| {
            let (ya, yb, fy) = place(y, y0, small.h);
            for (x, &(xa, xb, fx)) in cols.iter().enumerate() {
                let mix = |v: &[f32], stride: usize, c: usize| {
                    let at = |sx: usize, sy: usize| v[(sy * small.w + sx) * stride + c];
                    let top = at(xa, ya) + fx * (at(xb, ya) - at(xa, ya));
                    let bottom = at(xa, yb) + fx * (at(xb, yb) - at(xa, yb));
                    top + fy * (bottom - top)
                };
                for c in 0..3 {
                    drow[x * 3 + c] = mix(&small.delta, 3, c);
                }
                wrow[x] = mix(&small.weight, 1, 0);
            }
        });
    Layer {
        x0,
        y0,
        w,
        h,
        delta,
        weight,
        skin_level: small.skin_level,
        report: format!("{}, worked out at 1/{k} of the size", small.report),
    }
}

fn build(image: &Rgb32FImage, mask: &GrayImage, is_raw: bool, texture: f32) -> Option<Layer> {
    build_within(image, mask, is_raw, texture, MAX_PIXELS, TUNING)
}

fn build_within(
    image: &Rgb32FImage,
    mask: &GrayImage,
    is_raw: bool,
    texture: f32,
    max_pixels: usize,
    tuning: Tuning,
) -> Option<Layer> {
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
    // The work keeps a few dozen planes of the face's box at once: a face filling a big picture (an upscaled
    // one above all) would need several gigabytes. It is then worked out on a copy reduced by a whole factor,
    // and only the change is scaled back up: added to the full-size picture, it keeps all of the picture's
    // detail finer than the copy, so the result is the renewed copy plus the full size's own finest grain.
    if w * h > max_pixels {
        let k = ((w * h) as f64 / max_pixels as f64).sqrt().ceil() as usize;
        let small = build_within(
            &shrink_rgb(image, k),
            &shrink_mask(mask, k),
            is_raw,
            texture,
            max_pixels,
            tuning,
        )?;
        return Some(enlarge(&small, k, iw, ih));
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

    // the base: the smallest radius that has evened out the skin, with its structure still in it. The radii are
    // tried on a copy whose face is at most about 1200 px wide: what is measured is the same, at a fraction of
    // the cost; the chosen one is then run on the full-size picture.
    let step = ((face / 1200.0).ceil() as usize).max(1);
    let (ys, inner_s) = if step > 1 {
        (
            downsample(&y, step),
            downsample(&interior, step).map(|v| if v > 0.5 { 1.0 } else { 0.0 }),
        )
    } else {
        (y.clone(), interior.clone())
    };
    let face_s = face / step as f32;
    let grad_s = gradient(&gauss(&ys, 0.006 * face_s));
    let g85_s = quantile(&grad_s, &inner_s, 0.85);
    let structure_s = grad_s.zip(&inner_s, |g, i| if g > g85_s { i } else { 0.0 });
    let structure_grad_s = mean_where(&grad_s, &structure_s).max(1e-9);
    let mut tried: Vec<(f32, f32, f32)> = Vec::new(); // radius, unevenness, structure kept
    for share in [0.02f32, 0.03, 0.045, 0.07, 0.10] {
        let base_s = median(&ys, share * face_s);
        let mid = base_s.zip(&gauss(&base_s, 0.15 * face_s), |a, b| (a - b).abs());
        let unevenness = mean_where(&mid, &inner_s);
        let kept =
            mean_where(&gradient(&gauss(&base_s, 0.006 * face_s)), &structure_s) / structure_grad_s;
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

    let evened: [Plane; 3] = std::array::from_fn(|c| median(&chan[c], radius));
    // The median evens the skin but also clips the face's own light: the top of a cheekbone, the shade
    // beside the nose, the roundness of the forehead all come out flatter. That light is given back: what
    // the median took from the brightness, spread wider than its radius so a blemish (narrower) melts
    // away in it, and counted only over the face so the hair and the background do not leak in. It goes on
    // every channel alike, so it brings back brightness and not a blotch's colour.
    let evened_y = luma(&evened);
    let taken = y.zip(&evened_y, |a, b| a - b).zip(&m, |d, w| d * w);
    let light = if tuning.spread > 0.0 {
        let spread = tuning.spread * radius;
        let weight_sum = gauss(&m, spread);
        gauss(&taken, spread).zip(&weight_sum, |d, w| d / w.max(1e-3))
    } else {
        taken.map(|_| 0.0)
    };
    let base: [Plane; 3] = std::array::from_fn(|c| evened[c].zip(&light, |b, l| b + l));
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
    let tone = if tuning.local_highlights {
        // a shine (the tip of the nose, a cheekbone, the forehead) is narrower than the median and gone
        // from the evened skin: highlights are judged on each place's own light, shadows still on the
        // evened skin
        let local = gauss(&y, sig_mid);
        base_y.zip(&local, |b, l| {
            (1.0 - smoothstep(hi0, hi1, b.max(l))) * smoothstep(sh0, sh1, b)
        })
    } else {
        base_y.map(|v| (1.0 - smoothstep(hi0, hi1, v)) * smoothstep(sh0, sh1, v))
    };

    // the fine texture: its strength and grain size measured in the cleanest skin of this face, on a high pass
    // made only to measure them, wide enough to let the coarsest grain through
    let sig_probe = (0.005 * face).max(GRAIN_PROBE_MIN);
    let detail = y.zip(&gauss(&y, sig_probe), |a, b| a - b);
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
    // How wide the grain is: how far, across and down, the skin's detail goes on looking like itself. That
    // width is then matched against made-up grains, blurred noise put through the same high pass, and the
    // blur whose grain measures the same is the one used.
    let half_width = grain_half_width(&detail, &clean);
    let grains = grain_table(sig_probe);
    // the grain's blur, and, by the tutorial's rule, the High Pass radius too: "the exact number you chose for
    // the Gaussian blur"
    let sig_tex = blur_for_width(&grains, half_width).clamp(0.8, 4.0);
    let unit_rms = strength_for_blur(&grains, sig_tex);
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
    // the noise layer's deviation from its 50 % grey: through the same high pass, as rough as the face's
    // cleanest skin
    let tex_amp = s_clean * texture / (nv * unit_rms);

    // "OG texture": a High Pass of each channel at the grain's radius, in Linear Light at the tutorial's 72 %.
    // Its strongest detail is soft-clipped, so a hair or a blemish's rim is not laid back as texture.
    let high_pass: [Plane; 3] =
        std::array::from_fn(|c| chan[c].zip(&gauss(&chan[c], sig_tex), |a, b| a - b));
    let s_detail = mean_where(&luma(&high_pass).map(|d| d * d), &clean)
        .sqrt()
        .max(1e-5);
    let clip = 2.5 * s_detail;

    // the renewed skin, layer by layer as in Photoshop: the median base; the noise layer in Overlay; the High
    // Pass in Linear Light
    let mut delta = vec![0.0f32; w * h * 3];
    let mut weight = vec![0.0f32; w * h];
    delta
        .par_chunks_mut(3)
        .zip(weight.par_iter_mut())
        .enumerate()
        .for_each(|(i, (d, wt))| {
            let grain = (noise.v[i] - nm) * tex_amp;
            for c in 0..3 {
                let b = base[c].v[i].clamp(0.0, 1.0);
                let over = overlay(b, 0.5 + grain);
                let hp = clip * (high_pass[c].v[i] / clip).tanh();
                let renewed = over + DETAIL_OPACITY * (linear_light(over, 0.5 + hp) - over);
                // a clipped pixel is changed from where the gamma-coded copy had it
                d[c] = from_gamma(renewed) - from_gamma(chan[c].v[i]);
            }
            *wt = need.v[i] * guard.v[i] * border.v[i] * tone.v[i];
        });

    let report = format!(
        "face {:.0} px, median radius {:.0} px ({:.1}% of the face, {} candidates), grain half width {:.2} px, \
         pore grain and high pass {:.2} px, texture {:.4}, detail {:.4}, highlights from {:.2}, shadows to {:.2}",
        face,
        radius,
        chosen.0 * 100.0,
        tried.len(),
        half_width,
        sig_tex,
        s_clean,
        s_detail,
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
        skin_level: from_gamma(quantile(&y, &interior, 0.5)),
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

    /// The light that shapes a face, the shading of its creases and contours, comes through the renewal: the
    /// median would soften it, and it is given back. The face is a mid-tone skin modelled by soft shading a
    /// few percent of its width across, with a brighter forehead so that the shading is not a highlight.
    fn modelled_face() -> (Rgb32FImage, GrayImage) {
        let (w, h) = (400u32, 480u32);
        let noise = white_noise(w as usize, h as usize);
        let mut img = Rgb32FImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let g = noise.v[(y * w + x) as usize] * 0.004;
            let (fx, fy) = (x as f32, y as f32);
            let shade = 0.02 * (fx / 9.0).sin() * (fy / 11.0).cos();
            let forehead = if y < 110 { 0.12 } else { 0.0 };
            let v = 0.15 + g + shade + forehead;
            p.0 = [v * 1.3, v, v * 0.8];
        }
        let mask = GrayImage::from_fn(w, h, |x, y| {
            Luma([if (10..390).contains(&x) && (10..470).contains(&y) {
                255
            } else {
                0
            }])
        });
        (img, mask)
    }

    fn shading_kept(spread: f32) -> f32 {
        let (img, mask) = modelled_face();
        let tuning = Tuning { spread, ..TUNING };
        let layer = build_within(&img, &mask, true, 1.0, MAX_PIXELS, tuning).expect("a layer");
        let at = |x: u32, y: u32| (y as usize - layer.y0) * layer.w + (x as usize - layer.x0);
        // the shading's swing over the cheeks, through a band that leaves out the grain
        let band = |renewed: bool| {
            let plane = Plane {
                w: 200,
                h: 200,
                v: (0..200 * 200)
                    .map(|i| {
                        let (x, y) = (100 + (i % 200) as u32, 200 + (i / 200) as u32);
                        let j = at(x, y);
                        let change = if renewed {
                            0.7 * layer.weight[j] * layer.delta[j * 3 + 1]
                        } else {
                            0.0
                        };
                        img.get_pixel(x, y).0[1] + change
                    })
                    .collect(),
            };
            let shading = gauss(&plane, 2.0);
            let mean = shading.v.iter().sum::<f32>() / shading.v.len() as f32;
            (shading.v.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / shading.v.len() as f32)
                .sqrt()
        };
        band(true) / band(false)
    }

    #[test]
    fn the_face_keeps_its_shading() {
        let kept = shading_kept(VOLUME_SPREAD);
        let flattened = shading_kept(0.0);
        assert!(
            kept > flattened + 0.1,
            "with the light given back {kept}, without {flattened}"
        );
        // shading as wide as the median's radius is the hardest to keep: about 0.85 of it is, against 0.7
        assert!(kept > 0.8, "the shading kept {kept}");
    }

    /// A face too big to work out at full size, as in an upscaled picture: it is renewed on a reduced copy,
    /// and the change, brought back to full size, still evens the spot and keeps the edge.
    #[test]
    fn a_face_too_big_is_renewed_on_a_reduced_copy() {
        let (small_img, small_mask) = skin();
        let img = imageops::resize(&small_img, 720, 840, imageops::FilterType::Nearest);
        let mask = imageops::resize(&small_mask, 720, 840, imageops::FilterType::Nearest);
        let layer = build_within(&img, &mask, true, 1.0, 200_000, TUNING).expect("a layer");
        assert!(layer.report.contains("1/2 of the size"), "{}", layer.report);
        assert!(layer.x0 + layer.w <= 720 && layer.y0 + layer.h <= 840);
        assert!(
            layer.w * layer.h > 200_000,
            "the layer covers the full-size box"
        );
        let at = |x: u32, y: u32| (y as usize - layer.y0) * layer.w + (x as usize - layer.x0);
        let renewed = |x: u32, y: u32, c: usize| {
            let i = at(x, y);
            img.get_pixel(x, y).0[c] + layer.weight[i] * layer.delta[i * 3 + c]
        };
        let skin_green = 0.12;
        let before = (img.get_pixel(500, 300).0[1] - skin_green).abs();
        let after = (renewed(500, 300, 1) - skin_green).abs();
        assert!(
            after < before * 0.7,
            "spot {before} -> {after}: {}",
            layer.report
        );
        let step = renewed(592, 600, 1) - renewed(620, 600, 1);
        assert!(step > 0.03, "edge {step}: {}", layer.report);
    }

    /// Renders the renewal of an 8-bit picture (RRAW_IN) with its skin mask (RRAW_MASK, grey PNG), at the
    /// amount RRAW_AMOUNT (0..1), once for each light spread in RRAW_SPREADS ("0,1.25,..."; 0 is no light
    /// given back), to RRAW_OUT_<spread>.png. For judging the face's volumes on real portraits.
    #[test]
    #[ignore]
    fn renewal_spreads_on_a_picture() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let img = image::open(var("RRAW_IN")).expect("picture").to_rgb32f();
        let mask = image::open(var("RRAW_MASK")).expect("mask").to_luma8();
        let amount: f32 = var("RRAW_AMOUNT").parse().unwrap();
        let out = var("RRAW_OUT");
        for spread in var("RRAW_SPREADS").split(',') {
            let (s, local) = spread.split_once(':').unwrap_or((spread, "0"));
            let tuning = Tuning {
                spread: s.parse().unwrap(),
                local_highlights: local == "1",
            };
            let layer = build_within(&img, &mask, false, 1.0, MAX_PIXELS, tuning).expect("a layer");
            eprintln!("variant {spread}: {}", layer.report);
            let mut res = img.clone();
            for ly in 0..layer.h {
                for lx in 0..layer.w {
                    let i = ly * layer.w + lx;
                    let wt = (layer.weight[i] * amount).clamp(0.0, 1.0);
                    let p = res.get_pixel_mut((layer.x0 + lx) as u32, (layer.y0 + ly) as u32);
                    for c in 0..3 {
                        p.0[c] += wt * layer.delta[i * 3 + c];
                    }
                }
            }
            image::DynamicImage::ImageRgb32F(res)
                .to_rgb8()
                .save(format!("{out}_{}.png", spread.replace(':', "_")))
                .unwrap();
        }
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

    /// Renews a raw file (RRAW_RAW) over an oval mask (RRAW_OVAL: "cx,cy,rx,ry" in pixels) and writes the face's
    /// box before and after to RRAW_OUT + ".before.png" and RRAW_OUT + ".after.png".
    #[test]
    #[ignore]
    fn renewal_on_a_raw() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let path = var("RRAW_RAW");
        let bytes = std::fs::read(&path).unwrap();
        let settings = crate::app::settings::AppSettings::default();
        let img = crate::decode::image_loader::load_base_image_from_bytes(
            &bytes, &path, false, &settings, None,
        )
        .expect("the raw develops")
        .to_rgb32f();
        let oval: Vec<f32> = var("RRAW_OVAL")
            .split(',')
            .map(|v| v.parse().unwrap())
            .collect();
        let (cx, cy, rx, ry) = (oval[0], oval[1], oval[2], oval[3]);
        let mask = GrayImage::from_fn(img.width(), img.height(), |x, y| {
            let d = ((x as f32 - cx) / rx).powi(2) + ((y as f32 - cy) / ry).powi(2);
            Luma([if d < 1.0 { 255 } else { 0 }])
        });
        let started = std::time::Instant::now();
        let layer = build(&img, &mask, true, 1.0).expect("a layer");
        println!(
            "built in {:.1}s: {}",
            started.elapsed().as_secs_f32(),
            layer.report
        );
        let amount: f32 = std::env::var("RRAW_AMOUNT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        // shown with the skin at mid-grey, as an editor would show it
        let mut lumas: Vec<f32> = mask
            .enumerate_pixels()
            .filter(|(_, _, m)| m.0[0] > 127)
            .step_by(97)
            .map(|(x, y, _)| {
                let p = img.get_pixel(x, y).0;
                0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
            })
            .collect();
        let mid = lumas.len() / 2;
        let gain = 0.2 / *lumas.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
        let view = |v: f32| ((v * gain).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0 + 0.5) as u8;
        let (w, h) = (layer.w as u32, layer.h as u32);
        let mut before = image::RgbImage::new(w, h);
        let mut after = image::RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) as usize;
                let px = img.get_pixel(layer.x0 as u32 + x, layer.y0 as u32 + y).0;
                let wt = (layer.weight[i] * amount).clamp(0.0, 1.0);
                for (c, &v) in px.iter().enumerate() {
                    before.get_pixel_mut(x, y).0[c] = view(v);
                    after.get_pixel_mut(x, y).0[c] = view(v + wt * layer.delta[i * 3 + c]);
                }
            }
        }
        before
            .save(format!("{}.before.png", var("RRAW_OUT")))
            .unwrap();
        after
            .save(format!("{}.after.png", var("RRAW_OUT")))
            .unwrap();
    }

    #[test]
    fn the_grain_size_is_found_again() {
        // a noise field unlike the one the grains are matched against: the same generator, read down the columns
        let side = 420usize;
        let white = white_noise(side, side);
        let other = Plane {
            w: side,
            h: side,
            v: transpose(&white.v, side, side),
        };
        let mask = Plane {
            w: side,
            h: side,
            v: (0..side * side)
                .map(|i| {
                    let (x, y) = ((i % side) as f32 - 210.0, (i / side) as f32 - 210.0);
                    if x * x + y * y < 170.0 * 170.0 {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        };
        let table = grain_table(GRAIN_PROBE_MIN);
        for blur in [0.9f32, 1.4, 1.8, 2.6, 3.5] {
            let skin = gauss(&other, blur).map(|v| 0.4 + 0.01 * v);
            let detail = skin.zip(&gauss(&skin, GRAIN_PROBE_MIN), |a, b| a - b);
            let found = blur_for_width(&table, grain_half_width(&detail, &mask));
            println!("grain blurred {blur} px: measured {found:.2} px");
            assert!(
                (found - blur).abs() < 0.12 * blur,
                "a grain blurred {blur} px was measured as {found} px"
            );
        }
    }

    #[test]
    fn the_median_is_the_middle_of_the_round_neighbourhood() {
        let (w, h) = (61usize, 47usize);
        let noise = white_noise(w, h);
        let p = Plane {
            w,
            h,
            v: noise
                .v
                .iter()
                .enumerate()
                .map(|(i, n)| (0.5 + 0.15 * n + if i % w > 30 { 0.2 } else { 0.0 }).clamp(0.0, 1.0))
                .collect(),
        };
        for radius in [1.0f32, 3.0, 7.5, 12.0] {
            let fast = median(&p, radius);
            let r = radius.floor() as isize;
            for y in 0..h as isize {
                for x in 0..w as isize {
                    let mut window = Vec::new();
                    for dy in -r..=r {
                        let hw =
                            (radius * radius - (dy * dy) as f32).max(0.0).sqrt().floor() as isize;
                        for dx in -hw..=hw {
                            let (xx, yy) = (
                                (x + dx).clamp(0, w as isize - 1),
                                (y + dy).clamp(0, h as isize - 1),
                            );
                            window.push(p.v[yy as usize * w + xx as usize]);
                        }
                    }
                    let mid = window.len() / 2;
                    let exact = *window.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
                    let got = fast.v[y as usize * w + x as usize];
                    assert!(
                        (got - exact).abs() <= 1.0 / MEDIAN_BINS as f32,
                        "radius {radius} at {x},{y}: {got} against {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn nothing_changes_when_the_amount_is_nothing() {
        let adj = serde_json::json!({ "skinEnabled": true, "skinRenewal": 0, "skinFaceMap": "x" });
        assert!(!is_enabled(&adj));
    }
}
