// Skin dodge & burn, done automatically.
//
// By hand, a retoucher dodges and burns skin with two curves layers and a soft brush at low flow, for an hour:
// every patch a little darker than the skin around it is lightened, every patch a little lighter is darkened,
// until the skin's light and shade run evenly. Nothing is blurred, so every pore stays exactly where it was; and
// nothing as large as the face's own shapes is touched, so the cheekbone, the jaw and the shading of the light
// stay. What goes is everything in between: blotches, the shadows of small bumps, uneven patches, light
// circles, the mottling of tired skin. It is the most natural retouch there is and the slowest by hand.
//
// Here it is done by splitting the skin, over the skin only, into bands between those two sizes and, in each
// band, bringing every place towards how even this face's own cleanest skin is in that band:
//
//   - the brightness is worked on as its logarithm, so a correction is a ratio, as a curve that lightens or
//     darkens is; and the colour as the logarithm of red and blue over green, so the colour is evened as well
//     (the "T" of the B.S.T. method) and brightness and colour stay apart;
//   - the finest size is above the pore grain, which is measured in this very picture (its sharpness, its
//     resolution, the person's skin), so the pores, the stubble and the noise never go into any band;
//   - the coarsest size is a share of the face, and the coarsest bands are taken out less and less, so the
//     face keeps its shape;
//   - how even the skin is to be made in each band is not a fixed amount: it is the roughness of the quietest
//     quarter of this face's skin in that band. A band that is rough everywhere - the shading of a hard light, a
//     face's own relief - is the face, and is left; a band that is quiet in most of the skin and rough in places
//     carries the faults, and those places are brought down to the rest. So the same strength suits a young face
//     and an old one, soft light and hard;
//   - spots (the "B" of B.S.T.) are what stands far out of a band at the size of a spot and is round, which a
//     crease or an edge is not; they are taken out whole. When spots cover the face, they are freckles: the
//     person's, not faults, and they are kept;
//   - hair, a beard or stubble is where the skin has far more fine detail than its own pores: left alone, so a
//     beard's shadow is not lightened away. So are the strongest edges (nostrils, lid lines, the lip line), the
//     deepest shadows, and the highlights where they are brighter than around (the face's sheen).
//
// One Strength slider: the dodge & burn goes with it; the spots are out by 40 %; the colour follows at 0.6 of
// it. All of it is worked out once, on a copy where the face is about a thousand pixels wide, and kept as a
// change to add, so the slider never redoes it.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

use image::{DynamicImage, GenericImageView, GrayImage, Rgb32FImage, imageops};
use rayon::prelude::*;

use super::renewal::{
    GRAIN_PROBE_MIN, Plane, blur_for_width, gauss, gradient, grain_half_width, grain_table, quantile,
    smoothstep,
};
use super::skin_method::{
    MAP_DODGE_BURN, MAP_DODGE_BURN_KEPT, MAP_DODGE_BURN_SPOTS, Method, shown_map, tint,
};
use crate::render::adjustment_utils::is_effect_enabled;

const GAMMA: f32 = 2.2;
/// The face's width, in pixels, on the copy everything is worked out on.
const ANALYSIS_FACE: f32 = 1000.0;
/// At full strength each band is brought to this share of the roughness of the face's quietest quarter.
const TARGET: f32 = 0.5;
/// How much of the dodge & burn the colour gets.
const CHROMA: f32 = 0.6;
/// The strength at which the spots are out entirely.
const SPOTS_FULL_AT: f32 = 0.4;

/// One face's change, on the copy it was worked out on (see `FIELDS`), and, to be shown on request, how much
/// changes, the spots and what is left alone.
struct Face {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
    aw: usize,
    ah: usize,
    fields: Vec<[f32; FIELDS]>,
    maps: Vec<[f32; 3]>,
    /// the skin's middle brightness in the picture's own encoding, for showing a map over it
    skin_level: f32,
    report: String,
}

impl Face {
    /// The fields between the copy's pixels, at the full-size pixel (lx, ly) of the face's box.
    fn at<const N: usize>(&self, data: &[[f32; N]], lx: usize, ly: usize) -> [f32; N] {
        let fx = ((lx as f32 + 0.5) * self.aw as f32 / self.w as f32 - 0.5)
            .clamp(0.0, (self.aw - 1) as f32);
        let fy = ((ly as f32 + 0.5) * self.ah as f32 / self.h as f32 - 0.5)
            .clamp(0.0, (self.ah - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.aw - 1), (y0 + 1).min(self.ah - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let (a, b) = (&data[y0 * self.aw + x0], &data[y0 * self.aw + x1]);
        let (c, d) = (&data[y1 * self.aw + x0], &data[y1 * self.aw + x1]);
        std::array::from_fn(|i| {
            let top = a[i] + (b[i] - a[i]) * tx;
            let bottom = c[i] + (d[i] - c[i]) * tx;
            top + (bottom - top) * ty
        })
    }
}

struct Layer {
    faces: Vec<Face>,
}

// What is kept for each pixel of the copy. For each of log brightness, log red over green and log blue over
// green: the change at full strength; the spot-sized bands as they are; and the spot-sized part of the change.
// The spots take out what the dodge & burn has left of the spot-sized bands at the strength asked for, so that
// together they never take out more than is there, whatever the strength.
const CHANGE: usize = 0;
const SMALL: usize = 3;
const SMALL_CHANGE: usize = 6;
const SPOT: usize = 9;
const WEIGHT: usize = 10;
const FIELDS: usize = 11;

static LAYER: Mutex<Option<(u64, Arc<Layer>)>> = Mutex::new(None);
static BUILD: Mutex<()> = Mutex::new(());

fn shows_own_map(adjustments: &serde_json::Value) -> bool {
    matches!(
        shown_map(adjustments),
        MAP_DODGE_BURN | MAP_DODGE_BURN_SPOTS | MAP_DODGE_BURN_KEPT
    )
}

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    is_effect_enabled(adjustments, "skin", "skinEnabled")
        && Method::of(adjustments) == Method::DodgeBurn
        && (amount(adjustments) > 0.0 || shows_own_map(adjustments))
        && adjustments["skinFaceMap"]
            .as_str()
            .is_some_and(|map| !map.is_empty())
}

/// The Strength slider, shared by every way of retouching.
fn amount(adjustments: &serde_json::Value) -> f32 {
    (adjustments["skinRenewal"].as_f64().unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0)
}

/// How far a spot has to stand out, as a factor of the usual: the Skin tool's spot and overall detection
/// sliders, read the way it reads them (50 the usual, each 25 either way halving or doubling it).
fn spot_threshold(adjustments: &serde_json::Value) -> f32 {
    let level = |key: &str| adjustments[key].as_f64().unwrap_or(50.0).clamp(0.0, 100.0) as f32;
    2.0f32.powf((50.0 - level("skinSpotsSensitivity")) / 25.0)
        * 2.0f32.powf((50.0 - level("skinSensitivity")) / 25.0)
}

/// What the dodge & burn depends on, for the caches that must notice a change.
pub fn hash_inputs(adjustments: &serde_json::Value, hasher: &mut DefaultHasher, with_map: bool) {
    let on = is_enabled(adjustments);
    on.hash(hasher);
    if on {
        "dodgeBurn".hash(hasher);
        adjustments["skinRenewal"].to_string().hash(hasher);
        ((spot_threshold(adjustments) * 1000.0) as i32).hash(hasher);
        if with_map {
            shown_map(adjustments).hash(hasher);
        }
        let map = adjustments["skinFaceMap"].as_str().unwrap_or("");
        map.len().hash(hasher);
        map.as_bytes()
            .iter()
            .step_by((map.len() / 512).max(1))
            .for_each(|b| b.hash(hasher));
    }
}

/// The picture with its skin dodged and burnt, or as it was when there is nothing to do.
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
    let threshold = spot_threshold(adjustments);
    let mut key = DefaultHasher::new();
    (
        path,
        w,
        h,
        is_raw,
        crate::effects::maps::effect_map_key(face_map, adjustments),
        (threshold * 1000.0) as i32,
    )
        .hash(&mut key);
    let key = key.finish();

    let cached = |key: u64| {
        LAYER
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .filter(|(k, _)| *k == key)
            .map(|(_, l)| l.clone())
    };
    let layer = match cached(key) {
        Some(layer) => layer,
        None => {
            let _building = BUILD.lock().unwrap_or_else(|e| e.into_inner());
            match cached(key) {
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
                    let layer = build(&rgb, &map, is_raw, threshold);
                    if layer.faces.is_empty() {
                        return base;
                    }
                    for face in &layer.faces {
                        log::info!(
                            "Skin dodge & burn worked out in {:.1}s: {}",
                            started.elapsed().as_secs_f32(),
                            face.report
                        );
                    }
                    let layer = Arc::new(layer);
                    *LAYER.lock().unwrap_or_else(|e| e.into_inner()) = Some((key, layer.clone()));
                    layer
                }
            }
        }
    };

    let mut out = base.to_rgb32f();
    paint(&mut out, &layer, is_raw, amount(adjustments), shown_map(adjustments));
    Cow::Owned(DynamicImage::ImageRgb32F(out))
}

/// Lays the change on at `strength`, or, when one of the dodge & burn's maps is asked for, shows it instead.
fn paint(out: &mut Rgb32FImage, layer: &Layer, is_raw: bool, strength: f32, show: u32) {
    let stride = out.width() as usize * 3;
    let spots = (strength / SPOTS_FULL_AT).min(1.0);
    let lin = |v: f32| {
        if is_raw {
            v.max(0.0)
        } else {
            v.clamp(0.0, 1.0).powf(GAMMA)
        }
    };
    let enc = |v: f32| {
        if is_raw {
            v.max(0.0)
        } else {
            v.clamp(0.0, 1.0).powf(1.0 / GAMMA)
        }
    };
    let map_index = match show {
        MAP_DODGE_BURN => Some(0),
        MAP_DODGE_BURN_SPOTS => Some(1),
        MAP_DODGE_BURN_KEPT => Some(2),
        _ => None,
    };
    for face in &layer.faces {
        out.par_chunks_mut(stride)
            .enumerate()
            .skip(face.y0)
            .take(face.h)
            .for_each(|(y, row)| {
                let ly = y - face.y0;
                for lx in 0..face.w {
                    let px = (face.x0 + lx) * 3;
                    let c = face.at(&face.fields, lx, ly);
                    let weight = c[WEIGHT];
                    if weight <= 1e-4 {
                        continue;
                    }
                    if let Some(i) = map_index {
                        let value = face.at(&face.maps, lx, ly)[i];
                        let here = [row[px], row[px + 1], row[px + 2]];
                        row[px..px + 3].copy_from_slice(&tint(here, value, face.skin_level));
                        continue;
                    }
                    let change = |j: usize, share: f32| {
                        let dodged = share * strength;
                        let left = c[SMALL + j] + dodged * c[SMALL_CHANGE + j];
                        weight * (dodged * c[CHANGE + j] - spots * c[SPOT] * left)
                    };
                    let (dl, du, dv) = (change(0, 1.0), change(1, CHROMA), change(2, CHROMA));
                    let (r, g, b) = (lin(row[px]), lin(row[px + 1]), lin(row[px + 2]));
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    if luma <= 1e-9 {
                        continue;
                    }
                    // red and blue moved against green by the colour change, then all three scaled so the
                    // brightness comes out as asked
                    let (r2, b2) = (r * du.exp(), b * dv.exp());
                    let luma2 = 0.2126 * r2 + 0.7152 * g + 0.0722 * b2;
                    let k = luma * dl.exp() / luma2.max(1e-12);
                    row[px] = enc(r2 * k);
                    row[px + 1] = enc(g * k);
                    row[px + 2] = enc(b2 * k);
                }
            });
    }
}

// ---- finding the faces ---------------------------------------------------------------------------------------

/// One face's skin on the map: its box, in full-size pixels, and which cells of a coarse grid are its own.
struct FaceArea {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    label: u32,
}

/// The faces on the skin map, each a separate blob of skin, with a coarse grid saying which blob (dilated a
/// little, so a soft edge belongs to its own face) every place is: two faces side by side are each worked out
/// at their own size and neither is done twice.
fn face_areas(mask: &GrayImage) -> (Vec<FaceArea>, Vec<u32>, usize, usize) {
    const SIDE: f32 = 192.0;
    let (mw, mh) = (mask.width() as usize, mask.height() as usize);
    let scale = (SIDE / mw.max(mh) as f32).min(1.0);
    let (gw, gh) = (
        ((mw as f32 * scale).round() as usize).max(1),
        ((mh as f32 * scale).round() as usize).max(1),
    );
    let on: Vec<bool> = (0..gw * gh)
        .map(|i| {
            let (gx, gy) = (i % gw, i / gw);
            let sx = (((gx as f32 + 0.5) * mw as f32 / gw as f32) as usize).min(mw - 1);
            let sy = (((gy as f32 + 0.5) * mh as f32 / gh as f32) as usize).min(mh - 1);
            mask.get_pixel(sx as u32, sy as u32).0[0] > 127
        })
        .collect();
    let mut label = vec![0u32; gw * gh];
    // (cells, min x, min y, max x, max y) of each blob
    let mut blobs = Vec::new();
    for (start, &is_on) in on.iter().enumerate() {
        if !is_on || label[start] != 0 {
            continue;
        }
        let id = blobs.len() as u32 + 1;
        label[start] = id;
        let mut stack = vec![start];
        let mut blob = (0usize, usize::MAX, usize::MAX, 0usize, 0usize);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % gw, i / gw);
            blob.0 += 1;
            blob.1 = blob.1.min(x);
            blob.2 = blob.2.min(y);
            blob.3 = blob.3.max(x);
            blob.4 = blob.4.max(y);
            let mut visit = |nx: usize, ny: usize| {
                let n = ny * gw + nx;
                if on[n] && label[n] == 0 {
                    label[n] = id;
                    stack.push(n);
                }
            };
            if x > 0 {
                visit(x - 1, y);
            }
            if x + 1 < gw {
                visit(x + 1, y);
            }
            if y > 0 {
                visit(x, y - 1);
            }
            if y + 1 < gh {
                visit(x, y + 1);
            }
        }
        blobs.push(blob);
    }
    let largest = blobs.iter().map(|b| b.0).max().unwrap_or(0);
    let kept: Vec<u32> = blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| b.0 >= 12 && b.0 * 10 >= largest)
        .map(|(i, _)| i as u32 + 1)
        .collect();
    // each kept blob reaches two cells past its edge, where no other blob is
    let mut own = vec![0u32; gw * gh];
    for (i, o) in own.iter_mut().enumerate() {
        let (x, y) = ((i % gw) as isize, (i / gw) as isize);
        let mut found = 0u32;
        'look: for dy in -2isize..=2 {
            for dx in -2isize..=2 {
                let (nx, ny) = (x + dx, y + dy);
                if !(0..gw as isize).contains(&nx) || !(0..gh as isize).contains(&ny) {
                    continue;
                }
                let l = label[ny as usize * gw + nx as usize];
                if l != 0 && kept.contains(&l) {
                    if found != 0 && found != l {
                        found = 0;
                        break 'look;
                    }
                    found = l;
                }
            }
        }
        *o = found;
    }
    let areas = kept
        .iter()
        .map(|&l| {
            let b = blobs[l as usize - 1];
            FaceArea {
                x0: b.1 * mw / gw,
                y0: b.2 * mh / gh,
                x1: ((b.3 + 1) * mw / gw).min(mw),
                y1: ((b.4 + 1) * mh / gh).min(mh),
                label: l,
            }
        })
        .collect();
    (areas, own, gw, gh)
}

fn build(image: &Rgb32FImage, mask: &GrayImage, is_raw: bool, threshold: f32) -> Layer {
    let (areas, own, gw, gh) = face_areas(mask);
    let (mw, mh) = (mask.width() as usize, mask.height() as usize);
    let faces = areas
        .iter()
        .filter_map(|area| {
            // this face's skin only
            let skin = |x: usize, y: usize| -> f32 {
                let (gx, gy) = ((x * gw / mw).min(gw - 1), (y * gh / mh).min(gh - 1));
                if own[gy * gw + gx] == area.label {
                    mask.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0
                } else {
                    0.0
                }
            };
            build_face(image, &skin, is_raw, area, threshold)
        })
        .collect();
    Layer { faces }
}

// ---- the work ------------------------------------------------------------------------------------------------

/// A blur over the skin only: the average of what is skin near each place, so nothing outside the face - hair,
/// an eye, the background - is ever averaged into it, and its edge makes no band of its own.
fn skin_blur(x: &Plane, m: &Plane, sigma: f32) -> Plane {
    let num = gauss(&x.zip(m, |a, b| a * b), sigma);
    let den = gauss(m, sigma);
    num.zip(&den, |a, b| a / b.max(1e-6))
}

/// How round a bump of `b` is at each place, from the curvature read `h` pixels apart: the smaller curvature
/// over the larger, 1 for a round spot, 0 for a line or an edge, below 0 for a saddle.
fn roundness(b: &Plane, h: usize) -> Plane {
    let (w, hh) = (b.w, b.h);
    let mut v = vec![0.0f32; w * hh];
    v.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        if y < h || y + h >= hh {
            return;
        }
        let at = |x: usize, y: usize| b.v[y * w + x];
        for (x, o) in row.iter_mut().enumerate().take(w.saturating_sub(h)).skip(h) {
            let c = at(x, y);
            let hxx = at(x + h, y) - 2.0 * c + at(x - h, y);
            let hyy = at(x, y + h) - 2.0 * c + at(x, y - h);
            let hxy = 0.25 * (at(x + h, y + h) - at(x - h, y + h) - at(x + h, y - h) + at(x - h, y - h));
            let (tr, det) = (hxx + hyy, hxx * hyy - hxy * hxy);
            let disc = (0.25 * tr * tr - det).max(0.0).sqrt();
            let (l1, l2) = (0.5 * tr + disc, 0.5 * tr - disc);
            let (big, small) = if l1.abs() >= l2.abs() { (l1, l2) } else { (l2, l1) };
            *o = if big.abs() < 1e-12 { 0.0 } else { small / big };
        }
    });
    Plane { w, h: hh, v }
}

/// What working out one channel comes to: its change at full strength, the spot-sized bands and their part of
/// the change (which the spots then finish), and where spots stand out of it.
struct Channel {
    change: Plane,
    small: Plane,
    small_change: Plane,
    spots: Plane,
    /// for the log: each band's width, the quiet roughness it is brought to, and how much of it is taken on
    /// average
    bands: Vec<(f32, f32, f32)>,
}

/// What every channel is worked out against.
struct Context<'a> {
    m: &'a Plane,
    interior: &'a Plane,
    /// the skin a spot's yardstick is measured on: well inside, not hair, not an edge
    plain: &'a Plane,
    keep: &'a Plane,
    sigmas: &'a [f32],
    face: f32,
    threshold: f32,
}

fn work_channel(x: &Plane, cx: &Context, highlights: Option<&Plane>) -> Channel {
    let (w, h) = (x.w, x.h);
    let mut change = Plane {
        w,
        h,
        v: vec![0.0; w * h],
    };
    let mut small = change.clone();
    let mut small_change = change.clone();
    let mut spots = change.clone();
    let mut bands = Vec::new();
    let s_hi = *cx.sigmas.last().unwrap_or(&1.0);
    let mut previous = skin_blur(x, cx.m, cx.sigmas[0]);
    for pair in cx.sigmas.windows(2) {
        let (sig, next_sig) = (pair[0], pair[1]);
        let next = skin_blur(x, cx.m, next_sig);
        let band = previous.zip(&next, |a, b| a - b);
        // how rough this band is around each place, and how rough the quietest quarter of the skin is
        let rough = skin_blur(&band.map(|v| v * v), cx.m, 2.0 * next_sig).map(|v| v.max(0.0).sqrt() + 1e-6);
        let quiet = quantile(&rough, cx.interior, 0.25).max(1e-6);
        // the coarsest bands are the face's shape: taken out less and less
        let taper = 1.0 - smoothstep(0.035 * cx.face, s_hi, next_sig);
        let take: Vec<f32> = (0..w * h)
            .into_par_iter()
            .map(|i| {
                let mut a = (1.0 - TARGET * quiet / rough.v[i]).clamp(0.0, 1.0) * taper * cx.keep.v[i];
                if let Some(hl) = highlights
                    && band.v[i] > 0.0
                {
                    a *= 1.0 - hl.v[i];
                }
                a
            })
            .collect();
        let mut taken = 0.0f64;
        let mut count = 0usize;
        for (i, &a) in take.iter().enumerate() {
            change.v[i] -= a * band.v[i];
            if sig <= 0.03 * cx.face {
                small.v[i] += band.v[i];
                small_change.v[i] -= a * band.v[i];
            }
            if cx.interior.v[i] > 0.5 {
                taken += a as f64;
                count += 1;
            }
        }
        bands.push((sig, quiet, (taken / count.max(1) as f64) as f32));

        // a spot: far out of what this skin usually does at a spot's size, and round
        if (0.003 * cx.face..=0.02 * cx.face).contains(&sig) {
            // (with a floor: a channel with nothing in it but this spot must not read rounding as spots)
            let spread = (1.4826 * quantile(&band.map(f32::abs), cx.plain, 0.5)).max(2e-3);
            let round = roundness(&band, ((0.5 * sig).round() as usize).clamp(1, 8));
            let inside = gauss(cx.m, 2.0 * sig);
            let (lo, hi) = (4.0 * cx.threshold, 6.5 * cx.threshold);
            spots.v.par_iter_mut().enumerate().for_each(|(i, s)| {
                if inside.v[i] > 0.9 {
                    let z = band.v[i].abs() / spread;
                    *s = s.max(smoothstep(lo, hi, z) * smoothstep(0.2, 0.45, round.v[i]));
                }
            });
        }
        previous = next;
    }
    Channel {
        change,
        small,
        small_change,
        spots,
        bands,
    }
}

/// The pore grain's blur, in full-size pixels, measured on a window of the face's middle, the way the renewal
/// measures it: how far the skin's finest detail goes on looking like itself, matched against made-up grains.
fn grain_blur(
    image: &Rgb32FImage,
    skin: &(dyn Fn(usize, usize) -> f32 + Sync),
    area: &FaceArea,
    face: f32,
) -> Option<f32> {
    let half = ((0.22 * face) as usize).clamp(48, 320);
    let (cx, cy) = ((area.x0 + area.x1) / 2, (area.y0 + area.y1) / 2);
    let (x0, y0) = (cx.saturating_sub(half), cy.saturating_sub(half));
    let (x1, y1) = (
        (cx + half).min(image.width() as usize),
        (cy + half).min(image.height() as usize),
    );
    let (w, h) = (x1.checked_sub(x0)?, y1.checked_sub(y0)?);
    if w < 64 || h < 64 {
        return None;
    }
    let y = Plane {
        w,
        h,
        v: (0..w * h)
            .into_par_iter()
            .map(|i| {
                let p = image.get_pixel((x0 + i % w) as u32, (y0 + i / w) as u32).0;
                (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).max(1e-6).ln()
            })
            .collect(),
    };
    let m = Plane {
        w,
        h,
        v: (0..w * h)
            .into_par_iter()
            .map(|i| if skin(x0 + i % w, y0 + i / w) > 0.9 { 1.0 } else { 0.0 })
            .collect(),
    };
    let probe = (0.005 * face).max(GRAIN_PROBE_MIN);
    let inside = gauss(&m, probe).map(|v| if v > 0.97 { 1.0 } else { 0.0 });
    if inside.v.iter().filter(|&&v| v > 0.5).count() < 2000 {
        return None;
    }
    let detail = y.zip(&gauss(&y, probe), |a, b| a - b);
    let half_width = grain_half_width(&detail, &inside);
    Some(blur_for_width(&grain_table(probe), half_width).clamp(0.5, 6.0))
}

fn build_face(
    image: &Rgb32FImage,
    skin: &(dyn Fn(usize, usize) -> f32 + Sync),
    is_raw: bool,
    area: &FaceArea,
    threshold: f32,
) -> Option<Face> {
    let (iw, ih) = (image.width() as usize, image.height() as usize);
    let face_full = ((area.x1 - area.x0) as f32)
        .max(0.6 * (area.y1 - area.y0) as f32)
        .max(32.0);
    let margin = (0.12 * face_full) as usize;
    let (x0, y0) = (area.x0.saturating_sub(margin), area.y0.saturating_sub(margin));
    let (x1, y1) = ((area.x1 + margin).min(iw), (area.y1 + margin).min(ih));
    let (w, h) = (x1 - x0, y1 - y0);
    let step = ((face_full / ANALYSIS_FACE).ceil() as usize).max(1);
    let (aw, ah) = ((w / step).max(1), (h / step).max(1));
    let face = face_full / step as f32;
    if face < 48.0 {
        return None;
    }
    let lin = |v: f32| {
        if is_raw {
            v.max(0.0)
        } else {
            v.clamp(0.0, 1.0).powf(GAMMA)
        }
    };

    // the face's box, in linear light, averaged down to the copy
    let cells: Vec<[f32; 4]> = (0..aw * ah)
        .into_par_iter()
        .map(|i| {
            let (ax, ay) = (i % aw, i / aw);
            let (sx0, sy0) = (ax * w / aw, ay * h / ah);
            let (sx1, sy1) = (
                ((ax + 1) * w / aw).max(sx0 + 1),
                ((ay + 1) * h / ah).max(sy0 + 1),
            );
            let mut acc = [0.0f32; 4];
            for y in sy0..sy1 {
                for x in sx0..sx1 {
                    let p = image.get_pixel((x0 + x) as u32, (y0 + y) as u32).0;
                    acc[0] += lin(p[0]);
                    acc[1] += lin(p[1]);
                    acc[2] += lin(p[2]);
                    acc[3] += skin(x0 + x, y0 + y);
                }
            }
            let n = ((sx1 - sx0) * (sy1 - sy0)) as f32;
            acc.map(|a| a / n)
        })
        .collect();
    let plane = |c: usize| Plane {
        w: aw,
        h: ah,
        v: cells.iter().map(|p| p[c]).collect(),
    };
    let (r, g, b, m) = (plane(0), plane(1), plane(2), plane(3));
    let luma = Plane {
        w: aw,
        h: ah,
        v: (0..aw * ah)
            .map(|i| 0.2126 * r.v[i] + 0.7152 * g.v[i] + 0.0722 * b.v[i])
            .collect(),
    };
    let interior = gauss(&m, 0.01 * face).map(|v| if v > 0.95 { 1.0 } else { 0.0 });
    if interior.v.iter().filter(|&&v| v > 0.5).count() < 1500 {
        return None;
    }
    let mid = quantile(&luma, &interior, 0.5);
    if mid <= 0.0 {
        return None;
    }
    let eps = (0.01 * mid).max(1e-9);
    let l = luma.map(|v| (v.max(0.0) + eps).ln());
    let u = r.zip(&g, |a, b| ((a.max(0.0) + eps) / (b.max(0.0) + eps)).ln());
    let v = b.zip(&g, |a, b| ((a.max(0.0) + eps) / (b.max(0.0) + eps)).ln());

    // The finest band starts above the pore grain: a share of the face, or more when this picture's grain is
    // coarser (a soft lens, a small sensor, an upscaled file), so the grain never gets into any band.
    let grain = grain_blur(image, skin, area, face_full);
    let s_lo = (0.0035 * face)
        .max(1.6 * grain.unwrap_or(0.0) / step as f32)
        .max(1.0);
    let s_hi = 0.08 * face;
    let mut sigmas = vec![s_lo];
    while *sigmas.last().unwrap_or(&s_hi) < s_hi {
        let next = sigmas[sigmas.len() - 1] * std::f32::consts::SQRT_2;
        sigmas.push(next);
    }
    if sigmas.len() < 3 {
        return None;
    }

    // hair, a beard, stubble: far more fine detail than this skin's own
    let fine = l.zip(&skin_blur(&l, &m, s_lo), |a, b| a - b);
    let busy = skin_blur(&fine.map(|v| v * v), &m, 0.008 * face).map(|v| v.max(0.0).sqrt());
    let usual = quantile(&busy, &interior, 0.5).max(1e-6);
    let hair = busy.map(|e| smoothstep(1.8, 3.2, e / usual));
    // the strongest edges: nostrils, lid lines, the lip line
    let grad = gradient(&skin_blur(&l, &m, 0.006 * face));
    let g99 = quantile(&grad, &interior, 0.99).max(1e-9);
    let edges = gauss(&grad.map(|g| (g / g99).clamp(0.0, 1.0)), 0.004 * face);
    // the deepest shadows, and the highlights, judged on the face's own light and shade, not pixel by pixel
    let lie = skin_blur(&l, &m, s_hi);
    let (h0, h1, d0, d1) = (
        quantile(&lie, &interior, 0.93),
        quantile(&lie, &interior, 0.995),
        quantile(&lie, &interior, 0.002),
        quantile(&lie, &interior, 0.04),
    );
    let highlights = lie.map(|v| smoothstep(h0, h1, v));
    let keep = Plane {
        w: aw,
        h: ah,
        v: (0..aw * ah)
            .map(|i| {
                (1.0 - smoothstep(0.35, 0.9, edges.v[i]))
                    * (1.0 - hair.v[i])
                    * smoothstep(d0, d1, lie.v[i])
            })
            .collect(),
    };
    let plain = interior.zip(&keep, |i, k| if i > 0.5 && k > 0.5 { 1.0 } else { 0.0 });

    let context = Context {
        m: &m,
        interior: &interior,
        plain: &plain,
        keep: &keep,
        sigmas: &sigmas,
        face,
        threshold,
    };
    let (cl, (cu, cv)) = rayon::join(
        || work_channel(&l, &context, Some(&highlights)),
        || {
            rayon::join(
                || work_channel(&u, &context, None),
                || work_channel(&v, &context, None),
            )
        },
    );

    // the spots: found in brightness, or in colour (a red spot can be no darker), spread over the whole of each
    let found = Plane {
        w: aw,
        h: ah,
        v: (0..aw * ah)
            .map(|i| cl.spots.v[i].max(0.7 * cu.spots.v[i].max(cv.spots.v[i])))
            .collect(),
    };
    let mut spots = gauss(&found, 0.004 * face).zip(&keep, |s, k| (2.0 * s).clamp(0.0, 1.0) * k);
    let cover = {
        let (on, all) = spots
            .v
            .iter()
            .zip(&interior.v)
            .filter(|(_, i)| **i > 0.5)
            .fold((0usize, 0usize), |(on, all), (s, _)| (on + usize::from(*s > 0.5), all + 1));
        on as f32 / all.max(1) as f32
    };
    // spots all over the face are freckles: the person's, and kept
    let freckles = smoothstep(0.04, 0.10, cover);
    spots = spots.map(|s| s * (1.0 - freckles));

    let border = gauss(&m, 0.012 * face).map(|b| smoothstep(0.0, 1.0, b));
    let fields: Vec<[f32; FIELDS]> = (0..aw * ah)
        .map(|i| {
            [
                cl.change.v[i],
                cu.change.v[i],
                cv.change.v[i],
                cl.small.v[i],
                cu.small.v[i],
                cv.small.v[i],
                cl.small_change.v[i],
                cu.small_change.v[i],
                cv.small_change.v[i],
                spots.v[i],
                border.v[i],
            ]
        })
        .collect();
    let maps: Vec<[f32; 3]> = (0..aw * ah)
        .map(|i| {
            let amount = (cl.change.v[i].abs() + 0.5 * (cu.change.v[i].abs() + cv.change.v[i].abs())) * 10.0;
            [
                amount.clamp(0.0, 1.0),
                spots.v[i],
                (1.0 - keep.v[i]) * m.v[i].min(1.0),
            ]
        })
        .collect();

    let skin_level = if is_raw { mid } else { mid.powf(1.0 / GAMMA) };
    let bands: Vec<String> = cl
        .bands
        .iter()
        .map(|(s, q, a)| format!("{:.1}px:{:.4}/{:.2}", s * step as f32, q, a))
        .collect();
    let report = format!(
        "face {:.0} px (worked on at 1/{}), pore grain {}, bands {} (width:quiet roughness/share taken), spots \
         cover {:.1}% of the skin{}",
        face_full,
        step,
        grain.map_or("not measured".to_string(), |g| format!("{g:.2} px")),
        bands.join(" "),
        cover * 100.0,
        if freckles > 0.5 {
            ", kept as freckles"
        } else {
            ""
        },
    );
    Some(Face {
        x0,
        y0,
        w,
        h,
        aw,
        ah,
        fields,
        maps,
        skin_level,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::renewal::white_noise;
    use image::Luma;

    // A flat skin tone with fine grain (the pores), one soft dark blotch, one small red spot, a soft lighter
    // patch and a hard edge down one side.
    fn skin() -> (Rgb32FImage, GrayImage) {
        let (w, h) = (420u32, 480u32);
        let noise = white_noise(w as usize, h as usize);
        let pores = gauss(
            &Plane {
                w: w as usize,
                h: h as usize,
                v: noise.v.clone(),
            },
            0.8,
        );
        let mut img = Rgb32FImage::new(w, h);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let i = (y * w + x) as usize;
            let grain = 1.0 + 0.06 * pores.v[i];
            let bump = |cx: f32, cy: f32, s: f32| {
                (-(((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)) / (2.0 * s * s))).exp()
            };
            let blotch = 1.0 - 0.18 * bump(120.0, 150.0, 14.0);
            let patch = 1.0 + 0.12 * bump(230.0, 330.0, 16.0);
            // the light falling off down the face: its shape, which stays
            let light = 1.0 + 0.3 * (1.0 - y as f32 / h as f32);
            let spot = bump(280.0, 140.0, 4.0);
            let edge = if x > 360 { 0.55 } else { 1.0 };
            let level = 0.2 * grain * blotch * patch * edge * light;
            p.0 = [
                level * (1.25 + 0.5 * spot),
                level * (1.0 - 0.25 * spot),
                level * (0.85 - 0.2 * spot),
            ];
        }
        let mask = GrayImage::from_fn(w, h, |x, y| {
            Luma([if (12..408).contains(&x) && (12..468).contains(&y) {
                255
            } else {
                0
            }])
        });
        (img, mask)
    }

    fn render(img: &Rgb32FImage, layer: &Layer, strength: f32) -> Rgb32FImage {
        let mut out = img.clone();
        paint(&mut out, layer, true, strength, 0);
        out
    }

    fn luma(img: &Rgb32FImage, x: u32, y: u32) -> f32 {
        let p = img.get_pixel(x, y).0;
        0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
    }

    // the average brightness of a small square, which the grain averages out of
    fn area(img: &Rgb32FImage, cx: u32, cy: u32, r: u32) -> f32 {
        let mut s = 0.0;
        for y in cy - r..=cy + r {
            for x in cx - r..=cx + r {
                s += luma(img, x, y);
            }
        }
        s / ((2 * r + 1) * (2 * r + 1)) as f32
    }

    // how rough the finest detail is around a place: the pores
    fn pore_roughness(img: &Rgb32FImage, cx: u32, cy: u32) -> f32 {
        let mut s = 0.0;
        let mut n = 0.0;
        for y in cy - 12..=cy + 12 {
            for x in cx - 12..=cx + 12 {
                let here = luma(img, x, y).ln();
                let around = (luma(img, x + 1, y) * luma(img, x - 1, y) * luma(img, x, y + 1) * luma(img, x, y - 1))
                    .ln()
                    / 4.0;
                s += (here - around).powi(2);
                n += 1.0;
            }
        }
        (s / n).sqrt()
    }

    #[test]
    fn blotches_even_out_and_the_pores_and_the_edge_stay() {
        let (img, mask) = skin();
        let layer = build(&img, &mask, true, 1.0);
        assert_eq!(layer.faces.len(), 1);
        let report = &layer.faces[0].report;
        let out = render(&img, &layer, 1.0);
        // the dark blotch comes up most of the way to the skin beside it, at the same height in the light
        let plain = area(&img, 200, 150, 6);
        let (before, after) = (area(&img, 120, 150, 4), area(&out, 120, 150, 4));
        assert!(
            (plain - after).abs() < 0.5 * (plain - before).abs(),
            "blotch {before} -> {after}, skin {plain}: {report}"
        );
        // and the lighter patch comes down
        let plain = area(&img, 130, 330, 6);
        let (before, after) = (area(&img, 230, 330, 4), area(&out, 230, 330, 4));
        assert!(
            (after - plain).abs() < 0.6 * (before - plain).abs(),
            "patch {before} -> {after}, skin {plain}: {report}"
        );
        // the light still falls off down the face as it did
        let fall = |img: &Rgb32FImage| area(img, 200, 60, 6) / area(img, 200, 420, 6);
        let (f0, f1) = (fall(&img), fall(&out));
        assert!((f1 / f0 - 1.0).abs() < 0.08, "fall-off {f0} -> {f1}: {report}");
        // the pores are where they were, as rough as they were
        let (p0, p1) = (pore_roughness(&img, 200, 220), pore_roughness(&out, 200, 220));
        assert!((p1 / p0 - 1.0).abs() < 0.15, "pores {p0} -> {p1}: {report}");
        // the hard edge is still there
        let step = area(&out, 340, 240, 4) / area(&out, 385, 240, 4);
        assert!(step > 1.5, "edge ratio {step}: {report}");
    }

    #[test]
    fn a_red_spot_is_taken_out_by_the_strength_the_spots_are_full_at() {
        let (img, mask) = skin();
        let layer = build(&img, &mask, true, 1.0);
        let report = &layer.faces[0].report;
        let red = |img: &Rgb32FImage, x: u32, y: u32| {
            let p = img.get_pixel(x, y).0;
            p[0] / p[1]
        };
        let skin_red = 1.25;
        let out = render(&img, &layer, SPOTS_FULL_AT);
        let (before, after) = (red(&img, 280, 140), red(&out, 280, 140));
        assert!(
            (after - skin_red).abs() < 0.5 * (before - skin_red).abs(),
            "spot redness {before} -> {after}: {report}"
        );
    }

    #[test]
    fn nothing_moves_at_no_strength() {
        let (img, mask) = skin();
        let layer = build(&img, &mask, true, 1.0);
        let out = render(&img, &layer, 0.0);
        for (a, b) in img.pixels().zip(out.pixels()) {
            for c in 0..3 {
                assert!((a.0[c] - b.0[c]).abs() <= 1e-6 * a.0[c].abs().max(1.0));
            }
        }
    }

    #[test]
    fn two_faces_are_found_apart() {
        let mask = GrayImage::from_fn(600, 300, |x, y| {
            let a = (x as f32 - 150.0).powi(2) / 90.0f32.powi(2) + (y as f32 - 150.0).powi(2) / 120.0f32.powi(2);
            let b = (x as f32 - 450.0).powi(2) / 60.0f32.powi(2) + (y as f32 - 150.0).powi(2) / 80.0f32.powi(2);
            Luma([if a < 1.0 || b < 1.0 { 255 } else { 0 }])
        });
        let (areas, _, _, _) = face_areas(&mask);
        assert_eq!(areas.len(), 2);
        let widths: Vec<usize> = areas.iter().map(|a| a.x1 - a.x0).collect();
        assert!(widths.iter().any(|&w| (160..=200).contains(&w)), "{widths:?}");
        assert!(widths.iter().any(|&w| (100..=140).contains(&w)), "{widths:?}");
    }

    #[test]
    fn it_is_off_for_the_other_method_and_without_strength() {
        let on = serde_json::json!({
            "skinEnabled": true, "skinRenewal": 60, "skinRenewMethod": "dodgeBurn", "skinFaceMap": "x"
        });
        assert!(is_enabled(&on));
        let renewal = serde_json::json!({ "skinEnabled": true, "skinRenewal": 60, "skinFaceMap": "x" });
        assert!(!is_enabled(&renewal));
        let none = serde_json::json!({
            "skinEnabled": true, "skinRenewal": 0, "skinRenewMethod": "dodgeBurn", "skinFaceMap": "x"
        });
        assert!(!is_enabled(&none));
        let map = serde_json::json!({
            "skinEnabled": true, "skinRenewal": 0, "skinRenewMethod": "dodgeBurn", "skinFaceMap": "x",
            "skinShowMap": MAP_DODGE_BURN_SPOTS
        });
        assert!(is_enabled(&map));
    }
}
