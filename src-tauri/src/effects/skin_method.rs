// The one-button skin retouch, done one of several ways, chosen by the owner:
//   - Renewal: the "one-minute skin retouch" of a Photoshop action (renewal.rs). A median takes the skin down to
//     an even base and a made-up grain is laid back on. Strong; for skin with a lot wrong with it.
//   - Natural (dodge & burn): the light and shade of the skin evened out between the size of a pore and the size
//     of the face's own shapes, the pores themselves never touched (dodge_burn.rs). Subtle; the most natural.
//   - AI (learned): Alibaba's ABPN, trained on retouched portraits: its spots filled and its skin evened by a
//     learned blend layer (ai_retouch.rs). Its models run once and leave two maps with the edit.
//
// Each runs on the developed picture before the GPU pipeline, kept as a change to add, so its one Strength slider
// moves without anything being worked out again. This module picks the one that is asked for, and holds what
// they share: which map the preview is asked to show, and how a map is shown over the picture.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;

use glam::{Mat3, Vec3};
use image::{DynamicImage, Rgb32FImage};
use rayon::prelude::*;

use crate::render::color_space::{srgb_to_working, working_to_srgb};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Renewal,
    DodgeBurn,
    Ai,
}

impl Method {
    pub fn of(adjustments: &serde_json::Value) -> Self {
        match adjustments["skinRenewMethod"].as_str() {
            Some("dodgeBurn") => Method::DodgeBurn,
            Some("ai") => Method::Ai,
            _ => Method::Renewal,
        }
    }
}

// The maps the preview can be asked to show in place of the retouch, so that what a tool has found can be seen
// and its detection set before anything is applied. The preview asks for one with `skinShowMap`; it is never
// saved with the edit nor sent with an export.
//
// The Skin tool's own maps are drawn by its shader (skin.wgsl):
pub const MAP_SKIN: u32 = 1;
pub const MAP_SPOTS: u32 = 2;
// (redness and pigment are only ever read by the shader, by their number)
#[allow(dead_code)]
pub const MAP_REDNESS: u32 = 3;
#[allow(dead_code)]
pub const MAP_PIGMENT: u32 = 4;
pub const MAP_LINES: u32 = 5;
pub const MAP_UNDER_EYE: u32 = 6;
pub const MAP_SHINE: u32 = 7;
// and those of the one-button retouch here:
/// where the renewal goes and how far
pub const MAP_RENEWAL: u32 = 10;
/// how much the dodge & burn changes each place, at full strength
pub const MAP_DODGE_BURN: u32 = 11;
/// the spots the dodge & burn found and takes out whole
pub const MAP_DODGE_BURN_SPOTS: u32 = 12;
/// what the dodge & burn leaves alone inside the skin: hair and stubble, strong edges, deep shadows
pub const MAP_DODGE_BURN_KEPT: u32 = 13;
/// how much the learned retouch changes each place, at full strength
pub const MAP_AI_RETOUCH: u32 = 14;

/// The map the preview asks to see, or 0.
pub fn shown_map(adjustments: &serde_json::Value) -> u32 {
    adjustments["skinShowMap"].as_u64().unwrap_or(0).min(64) as u32
}

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    super::renewal::is_enabled(adjustments)
        || super::dodge_burn::is_enabled(adjustments)
        || super::ai_retouch::is_enabled(adjustments)
}

/// What the retouch depends on, for the caches that must notice a change. `with_map` counts the map shown in
/// the preview too: the upscaler's cache leaves it out, so showing a map does not start an upscale again (the
/// map is then not shown on an upscaled picture).
pub fn hash_inputs(adjustments: &serde_json::Value, hasher: &mut DefaultHasher, with_map: bool) {
    super::renewal::hash_inputs(adjustments, hasher, with_map);
    super::dodge_burn::hash_inputs(adjustments, hasher, with_map);
    super::ai_retouch::hash_inputs(adjustments, hasher, with_map);
}

/// The picture with its skin retouched the way asked for, or as it was.
///
/// The picture comes in the working space (Rec.2020 primaries): linear for a raw, encoded with the sRGB curve
/// otherwise. Each retouch finds what it works on - the skin's colour, its redness, its spots - with numbers
/// made for sRGB, so it is given the picture in sRGB primaries, in the same encoding, and only what it
/// changed is taken back to the working space: a colour sRGB cannot hold stays as it was wherever nothing
/// was done to it.
pub fn apply<'a>(
    base: Cow<'a, DynamicImage>,
    path: &str,
    is_raw: bool,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    let method = Method::of(adjustments);
    let enabled = match method {
        Method::Renewal => super::renewal::is_enabled(adjustments),
        Method::DodgeBurn => super::dodge_burn::is_enabled(adjustments),
        Method::Ai => super::ai_retouch::is_enabled(adjustments),
    };
    if !enabled {
        return base;
    }
    let working = base.to_rgb32f();
    let srgb = DynamicImage::ImageRgb32F(convert(&working, working_to_srgb(), is_raw));
    let retouched = match method {
        Method::Renewal => super::renewal::apply(Cow::Borrowed(&srgb), path, is_raw, adjustments),
        Method::DodgeBurn => {
            super::dodge_burn::apply(Cow::Borrowed(&srgb), path, is_raw, adjustments)
        }
        Method::Ai => super::ai_retouch::apply(Cow::Borrowed(&srgb), is_raw, adjustments),
    };
    let Cow::Owned(retouched) = retouched else {
        return base;
    };
    let retouched = retouched.into_rgb32f();
    let Some(srgb) = srgb.as_rgb32f() else {
        return base;
    };
    if retouched.dimensions() != working.dimensions() {
        return Cow::Owned(DynamicImage::ImageRgb32F(convert(
            &retouched,
            srgb_to_working(),
            is_raw,
        )));
    }
    Cow::Owned(DynamicImage::ImageRgb32F(add_change(
        working, srgb, &retouched, is_raw,
    )))
}

/// The picture taken through `matrix` (from one set of primaries to another), linear or in the sRGB curve.
fn convert(image: &Rgb32FImage, matrix: Mat3, linear: bool) -> Rgb32FImage {
    let mut out = image.clone();
    out.par_chunks_exact_mut(3)
        .for_each(|pixel| pixel.copy_from_slice(&through(matrix, pixel, linear)));
    out
}

/// The working picture with what the retouch changed, in sRGB primaries, added in the working space.
fn add_change(
    mut working: Rgb32FImage,
    before: &Rgb32FImage,
    after: &Rgb32FImage,
    linear: bool,
) -> Rgb32FImage {
    let back = srgb_to_working();
    working
        .par_chunks_exact_mut(3)
        .zip(before.par_chunks_exact(3))
        .zip(after.par_chunks_exact(3))
        .for_each(|((pixel, before), after)| {
            if before == after {
                return;
            }
            let (from, to) = (through(back, before, linear), through(back, after, linear));
            for c in 0..3 {
                pixel[c] += to[c] - from[c];
            }
        });
    working
}

fn through(matrix: Mat3, pixel: &[f32], linear: bool) -> [f32; 3] {
    let rgb = Vec3::new(pixel[0], pixel[1], pixel[2]);
    if linear {
        return (matrix * rgb).to_array();
    }
    let decoded = rgb.to_array().map(signed_decode);
    (matrix * Vec3::from_array(decoded))
        .to_array()
        .map(signed_encode)
}

// the sRGB curve, kept odd so a colour outside the gamut (a negative channel) comes back where it was
fn signed_decode(c: f32) -> f32 {
    c.signum() * crate::render::color_space::srgb_decode(c.abs())
}

fn signed_encode(c: f32) -> f32 {
    c.signum() * crate::render::color_space::srgb_encode(c.abs())
}

/// A pixel shown as part of a map: the picture in grey, dimmed, with the map's colour over it as strongly as
/// the map reads there. `level` is the skin's middle brightness in the picture's own encoding, which the
/// colour is scaled to, so a map reads the same on a dark picture as on a bright one.
pub fn tint(rgb: [f32; 3], value: f32, level: f32) -> [f32; 3] {
    let v = value.clamp(0.0, 1.0);
    let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    let grey = 0.45 * luma;
    let hot = [1.6 * level, 0.18 * level, 0.6 * level];
    [
        grey + (hot[0] - grey) * v,
        grey + (hot[1] - grey) * v,
        grey + (hot[2] - grey) * v,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_method_is_renewal_unless_asked() {
        assert_eq!(Method::of(&serde_json::json!({})), Method::Renewal);
        assert_eq!(
            Method::of(&serde_json::json!({ "skinRenewMethod": "dodgeBurn" })),
            Method::DodgeBurn
        );
    }

    #[test]
    fn a_map_tints_where_it_reads_and_greys_elsewhere() {
        let off = tint([0.4, 0.3, 0.2], 0.0, 0.3);
        assert!((off[0] - off[1]).abs() < 1e-6 && (off[1] - off[2]).abs() < 1e-6);
        let on = tint([0.4, 0.3, 0.2], 1.0, 0.3);
        assert!(on[0] > on[2] && on[2] > on[1]);
    }
}
