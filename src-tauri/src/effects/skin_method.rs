// The one-button skin retouch, done one of several ways, chosen by the owner:
//   - Renewal: the "one-minute skin retouch" of a Photoshop action (renewal.rs). A median takes the skin down to
//     an even base and a made-up grain is laid back on. Strong; for skin with a lot wrong with it.
//   - Natural (dodge & burn): the light and shade of the skin evened out between the size of a pore and the size
//     of the face's own shapes, the pores themselves never touched (dodge_burn.rs). Subtle; the most natural.
//
// Each runs on the developed picture before the GPU pipeline, kept as a change to add, so its one Strength slider
// moves without anything being worked out again. This module picks the one that is asked for, and holds what
// they share: which map the preview is asked to show, and how a map is shown over the picture.

use std::borrow::Cow;
use std::collections::hash_map::DefaultHasher;

use image::DynamicImage;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Renewal,
    DodgeBurn,
}

impl Method {
    pub fn of(adjustments: &serde_json::Value) -> Self {
        match adjustments["skinRenewMethod"].as_str() {
            Some("dodgeBurn") => Method::DodgeBurn,
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
pub const MAP_REDNESS: u32 = 3;
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

/// The map the preview asks to see, or 0.
pub fn shown_map(adjustments: &serde_json::Value) -> u32 {
    adjustments["skinShowMap"].as_u64().unwrap_or(0).min(64) as u32
}

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    super::renewal::is_enabled(adjustments) || super::dodge_burn::is_enabled(adjustments)
}

/// What the retouch depends on, for the caches that must notice a change. `with_map` counts the map shown in
/// the preview too: the upscaler's cache leaves it out, so showing a map does not start an upscale again (the
/// map is then not shown on an upscaled picture).
pub fn hash_inputs(adjustments: &serde_json::Value, hasher: &mut DefaultHasher, with_map: bool) {
    super::renewal::hash_inputs(adjustments, hasher, with_map);
    super::dodge_burn::hash_inputs(adjustments, hasher, with_map);
}

/// The picture with its skin retouched the way asked for, or as it was.
pub fn apply<'a>(
    base: Cow<'a, DynamicImage>,
    path: &str,
    is_raw: bool,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    match Method::of(adjustments) {
        Method::Renewal => super::renewal::apply(base, path, is_raw, adjustments),
        Method::DodgeBurn => super::dodge_burn::apply(base, path, is_raw, adjustments),
    }
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
