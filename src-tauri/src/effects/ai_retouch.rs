// The skin retouch learned from retouched portraits: Alibaba's ABPN (CVPR 2022), as its pipeline runs it.
//
// The models run once, when the method is chosen (ai::commands::generate_ai_skin_retouch, masks::face::
// compose_ai_retouch), and leave two maps with the edit: the spots filled with skin, as a change, and a blend layer
// that evens the skin. Here they are only laid on, so the Strength slider moves at once and nothing needs a model:
//
//   - the spots first, all of them by 40 % of the slider, as the other methods take them;
//   - then the blend layer as a soft light, (1 - 2m) x^2 + 2m x, with the layer scaled around 0.5 by the slider:
//     the pipeline's own `degree`, so 100 % is the retouch exactly as ABPN makes it.
//
// Both act on the picture as the models saw it: a raw file is put through the same display curve the face models
// get (gamma and contrast), changed there, and brought back to linear; a pixel the curve clips is left alone. The
// face's skin map, widened a little, keeps the change on the face.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use image::{DynamicImage, GenericImageView, GrayImage, RgbaImage};
use rayon::prelude::*;

use super::renewal::{Plane, gauss, smoothstep};
use super::skin_method::{MAP_AI_RETOUCH, Method, shown_map, tint};
use crate::render::adjustment_utils::is_effect_enabled;

/// The display curve the face models are given a raw picture through
/// (render::image_processing::apply_cpu_default_raw_processing).
const RAW_GAMMA: f32 = 2.38;
const RAW_CONTRAST: f32 = 1.28;
/// The strength at which the spots are out entirely.
const SPOTS_FULL_AT: f32 = 0.4;

const KEYS: [&str; 3] = ["skinAiBlendMap", "skinAiSpotsMap", "skinFaceMap"];

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    is_effect_enabled(adjustments, "skin", "skinEnabled")
        && Method::of(adjustments) == Method::Ai
        && (amount(adjustments) > 0.0 || shown_map(adjustments) == MAP_AI_RETOUCH)
        && KEYS.iter().all(|key| {
            adjustments[*key]
                .as_str()
                .is_some_and(|map| !map.is_empty())
        })
}

/// The Strength slider, shared by every way of retouching.
fn amount(adjustments: &serde_json::Value) -> f32 {
    (adjustments["skinRenewal"].as_f64().unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0)
}

/// What the learned retouch depends on, for the caches that must notice a change.
pub fn hash_inputs(adjustments: &serde_json::Value, hasher: &mut DefaultHasher, with_map: bool) {
    let on = is_enabled(adjustments);
    on.hash(hasher);
    if on {
        "ai".hash(hasher);
        adjustments["skinRenewal"].to_string().hash(hasher);
        if with_map {
            shown_map(adjustments).hash(hasher);
        }
        for key in KEYS {
            let map = adjustments[key].as_str().unwrap_or("");
            map.len().hash(hasher);
            map.as_bytes()
                .iter()
                .step_by((map.len() / 512).max(1))
                .for_each(|b| b.hash(hasher));
        }
    }
}

/// The face's skin, widened a little and softened, at the skin map's own size: where the retouch may go.
static WHERE: Mutex<Option<(u64, Arc<Plane>)>> = Mutex::new(None);

fn where_to(map: &GrayImage, key: u64) -> Arc<Plane> {
    if let Some((k, plane)) = WHERE.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
        && *k == key
    {
        return plane.clone();
    }
    let (w, h) = (map.width() as usize, map.height() as usize);
    let skin = Plane {
        w,
        h,
        v: map.pixels().map(|p| p.0[0] as f32 / 255.0).collect(),
    };
    let sigma = 0.006 * w.max(h) as f32;
    let plane = Arc::new(gauss(&skin, sigma).map(|v| smoothstep(0.02, 0.5, v)));
    *WHERE.lock().unwrap_or_else(|e| e.into_inner()) = Some((key, plane.clone()));
    plane
}

/// A map read between its pixels at the picture's pixel (x, y): channels 0..N of each.
fn sample<const N: usize>(
    data: &[u8],
    stride: usize,
    (mw, mh): (usize, usize),
    (w, h): (usize, usize),
    (x, y): (usize, usize),
) -> [f32; N] {
    let fx = ((x as f32 + 0.5) * mw as f32 / w as f32 - 0.5).clamp(0.0, (mw - 1) as f32);
    let fy = ((y as f32 + 0.5) * mh as f32 / h as f32 - 0.5).clamp(0.0, (mh - 1) as f32);
    let (x0, y0) = (fx as usize, fy as usize);
    let (x1, y1) = ((x0 + 1).min(mw - 1), (y0 + 1).min(mh - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: usize, y: usize, c: usize| data[(y * mw + x) * stride + c] as f32;
    std::array::from_fn(|c| {
        let top = at(x0, y0, c) + (at(x1, y0, c) - at(x0, y0, c)) * tx;
        let bottom = at(x0, y1, c) + (at(x1, y1, c) - at(x0, y1, c)) * tx;
        top + (bottom - top) * ty
    })
}

fn sample_plane(plane: &Plane, (w, h): (usize, usize), (x, y): (usize, usize)) -> f32 {
    let fx = ((x as f32 + 0.5) * plane.w as f32 / w as f32 - 0.5).clamp(0.0, (plane.w - 1) as f32);
    let fy = ((y as f32 + 0.5) * plane.h as f32 / h as f32 - 0.5).clamp(0.0, (plane.h - 1) as f32);
    let (x0, y0) = (fx as usize, fy as usize);
    let (x1, y1) = ((x0 + 1).min(plane.w - 1), (y0 + 1).min(plane.h - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let at = |x: usize, y: usize| plane.v[y * plane.w + x];
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    top + (bottom - top) * ty
}

/// A raw picture's linear value as the models saw it, or None where their curve clips it.
fn to_model(v: f32) -> Option<f32> {
    let e = (v.max(0.0).powf(1.0 / RAW_GAMMA) - 0.5) * RAW_CONTRAST + 0.5;
    (0.0..=1.0).contains(&e).then_some(e)
}

fn from_model(e: f32) -> f32 {
    ((e.clamp(0.0, 1.0) - 0.5) / RAW_CONTRAST + 0.5)
        .max(0.0)
        .powf(RAW_GAMMA)
}

/// One pixel retouched: `e` as the models saw it (0..1), the spots' change `d`, the blend layer `m` (0.5 where
/// nothing changes), at `strength`.
fn retouch(e: [f32; 3], d: [f32; 3], m: [f32; 3], strength: f32) -> [f32; 3] {
    let spots = (strength / SPOTS_FULL_AT).min(1.0);
    std::array::from_fn(|c| {
        let x = (e[c] + spots * d[c]).clamp(0.0, 1.0);
        let layer = 0.5 + (m[c] - 0.5) * strength;
        ((1.0 - 2.0 * layer) * x * x + 2.0 * layer * x).clamp(0.0, 1.0)
    })
}

/// The picture with its skin retouched the learned way, or as it was when there is nothing to do.
pub fn apply<'a>(
    base: Cow<'a, DynamicImage>,
    is_raw: bool,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    if !is_enabled(adjustments) {
        return base;
    }
    let resolve_rgba = |key: &str| -> Option<Arc<RgbaImage>> {
        crate::effects::maps::resolve_rgba_map(adjustments[key].as_str()?, adjustments, None)
    };
    let (Some(blend), Some(spots)) = (
        resolve_rgba("skinAiBlendMap"),
        resolve_rgba("skinAiSpotsMap"),
    ) else {
        return base;
    };
    let face_map = adjustments["skinFaceMap"].as_str().unwrap_or("");
    let Some(skin) = crate::effects::maps::resolve_luma_map(face_map, adjustments, None) else {
        return base;
    };
    let skin = where_to(
        &skin,
        crate::effects::maps::effect_map_key(face_map, adjustments),
    );
    if [blend.width(), blend.height(), spots.width(), spots.height()].contains(&0) {
        return base;
    }

    let strength = amount(adjustments);
    let show = shown_map(adjustments) == MAP_AI_RETOUCH;
    let (w, h) = base.dimensions();
    let size = (w as usize, h as usize);
    let blend_size = (blend.width() as usize, blend.height() as usize);
    let spots_size = (spots.width() as usize, spots.height() as usize);
    // a map over a dark raw picture is shown at the brightness of mid-tone skin as the models saw it
    let level = if is_raw { from_model(0.6) } else { 0.6 };
    let mut out = base.to_rgb32f();
    let stride = w as usize * 3;
    out.par_chunks_mut(stride).enumerate().for_each(|(y, row)| {
        for x in 0..w as usize {
            let weight = sample_plane(&skin, size, (x, y));
            if weight <= 1e-3 {
                continue;
            }
            let m: [f32; 3] =
                sample::<3>(blend.as_raw(), 4, blend_size, size, (x, y)).map(|v| v / 255.0);
            let d: [f32; 3] = sample::<3>(spots.as_raw(), 4, spots_size, size, (x, y))
                .map(|v| (v - 128.0) / 254.0);
            let px = x * 3;
            let here = [row[px], row[px + 1], row[px + 2]];
            if show {
                let value = m.iter().map(|v| (v - 0.5).abs()).sum::<f32>() * 6.0
                    + d.iter().map(|v| v.abs()).sum::<f32>() * 3.0;
                row[px..px + 3].copy_from_slice(&tint(here, value * weight, level));
                continue;
            }
            let seen: [Option<f32>; 3] = if is_raw {
                here.map(to_model)
            } else {
                here.map(|v| Some(v.clamp(0.0, 1.0)))
            };
            let [Some(r), Some(g), Some(b)] = seen else {
                continue;
            };
            let before = [r, g, b];
            let done = retouch(before, d, m, strength);
            for (c, (&after, &was)) in done.iter().zip(&before).enumerate() {
                let e = was + weight * (after - was);
                row[px + c] = if is_raw { from_model(e) } else { e };
            }
        }
    });
    Cow::Owned(DynamicImage::ImageRgb32F(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_changes_at_no_strength_and_a_neutral_layer() {
        let e = [0.3, 0.55, 0.8];
        assert_eq!(retouch(e, [0.1, -0.2, 0.05], [0.8, 0.2, 0.6], 0.0), e);
        let same = retouch(e, [0.0; 3], [0.5; 3], 1.0);
        for c in 0..3 {
            assert!((same[c] - e[c]).abs() < 1e-6);
        }
    }

    #[test]
    fn the_layer_lightens_above_half_and_darkens_below() {
        let e = [0.4; 3];
        let lighter = retouch(e, [0.0; 3], [0.7; 3], 1.0);
        let darker = retouch(e, [0.0; 3], [0.3; 3], 1.0);
        assert!(lighter[0] > 0.4 && darker[0] < 0.4);
        // and half the strength is half the layer
        let half = retouch(e, [0.0; 3], [0.7; 3], 0.5);
        let expected = retouch(e, [0.0; 3], [0.6; 3], 1.0);
        assert!((half[0] - expected[0]).abs() < 1e-6);
    }

    #[test]
    fn the_spots_are_filled_by_the_strength_they_are_full_at() {
        let e = [0.5; 3];
        let filled = retouch(e, [0.1; 3], [0.5; 3], SPOTS_FULL_AT);
        assert!((filled[0] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn the_raw_curve_comes_back() {
        for v in [0.01f32, 0.05, 0.18, 0.4] {
            let e = to_model(v).expect("inside the curve");
            assert!((from_model(e) - v).abs() < 1e-4 * v.max(1.0), "{v}");
        }
        assert!(to_model(5.0).is_none());
    }

    #[test]
    fn it_needs_the_method_and_its_maps() {
        let mut adj = serde_json::json!({
            "skinEnabled": true, "skinRenewal": 70, "skinRenewMethod": "ai",
            "skinAiBlendMap": "a", "skinAiSpotsMap": "b", "skinFaceMap": "c"
        });
        assert!(is_enabled(&adj));
        adj["skinAiSpotsMap"] = serde_json::Value::Null;
        assert!(!is_enabled(&adj));
    }
}
