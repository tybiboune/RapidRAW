// Skin by kind of fault, over the face mask the masking system already builds.
//
// What this adds that a smoothing slider cannot: each kind of fault has its own map, so taking a
// blemish out does not cost the pores, and the skin's own grain goes back on top of whatever was done.
// The face's colour is measured on the face itself rather than assumed, which is what lets the redness
// slider send a flush to the colour of the cheek beside it instead of to a guess.
//
// All of it runs on the card, including the measuring: see shaders/skin.wgsl. Nothing here is on until
// a slider is moved - a face is not something to correct by default.

use std::sync::Arc;

use image::GrayImage;

use crate::render::adjustment_utils::is_effect_enabled;
use crate::render::geometry::GeometryParams;
use crate::render::gpu::compute::{Bind, ComputeEncoder, Grid, Group, Out, RGB_MODE, Slot};

const STATS_LONG_SIDE: f32 = 512.0;

#[repr(C)]
#[derive(Debug, Default, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// The order is the layout: a pair of numbers sits on an eight-byte boundary and the whole thing comes
// to a multiple of sixteen, which is what the shader's copy of this struct also works out to.
struct Params {
    grid: Grid,
    spots: f32,
    redness: f32,
    shine: f32,
    keep_texture: f32,
    low_size: [u32; 2],
    face_size: [u32; 2],
    redness_way: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct Skin {
    face_map: Arc<GrayImage>,
    map_key: u64,
    spots: f32,
    redness: f32,
    shine: f32,
    keep_texture: f32,
    redness_way: u32,
}

pub fn prepare(
    adjustments: &serde_json::Value,
    geometry: Option<&GeometryParams>,
    (w, h): (u32, u32),
) -> Option<Skin> {
    if !is_effect_enabled(adjustments, "skin", "skinEnabled") {
        return None;
    }
    // the face skin coverage, as the face-part masking already produces it
    let face_b64 = adjustments["skinFaceMap"]
        .as_str()
        .filter(|map| !map.is_empty())?;
    let get = |key: &str| (adjustments[key].as_f64().unwrap_or(0.0) as f32 / 100.0).clamp(0.0, 1.0);
    let spots = get("skinSpots");
    let redness = get("skinRedness");
    let shine = get("skinShine");
    if spots <= 0.0 && redness <= 0.0 && shine <= 0.0 {
        return None;
    }
    if w < 8 || h < 8 {
        return None;
    }
    let face_map = crate::effects::maps::resolve_luma_map(face_b64, adjustments, geometry)?;
    if face_map.width() < 2 || face_map.height() < 2 {
        return None;
    }
    Some(Skin {
        map_key: crate::effects::maps::effect_map_key(face_b64, adjustments),
        face_map,
        spots,
        redness,
        shine,
        keep_texture: get("skinKeepTexture"),
        redness_way: adjustments["skinRednessWay"].as_u64().unwrap_or(0).min(3) as u32,
    })
}

pub fn encode(enc: &mut ComputeEncoder, job: &Skin, src: &Slot, dst: &Slot) {
    let size = src.size;
    let f32x4 = wgpu::TextureFormat::Rgba32Float;
    let r32 = wgpu::TextureFormat::R32Float;
    let (fw, fh) = job.face_map.dimensions();

    // a sample of the picture is enough to measure a middle and a spread on, and keeps the one pass
    // that walks every skin pixel short
    let scale = (STATS_LONG_SIDE / size.0.max(size.1) as f32).min(1.0);
    let low_size = (
        ((size.0 as f32 * scale).round() as u32).clamp(4, size.0),
        ((size.1 as f32 * scale).round() as u32).clamp(4, size.1),
    );

    let params = Params {
        grid: Grid::new(size),
        spots: job.spots,
        redness: job.redness,
        shine: job.shine,
        keep_texture: job.keep_texture,
        low_size: [low_size.0, low_size.1],
        face_size: [fw, fh],
        redness_way: job.redness_way,
        ..Default::default()
    };

    let face = enc.cached_map("Skin Face Map", job.map_key, (fw, fh), r32, || {
        job.face_map.pixels().map(|p| p[0] as f32 / 255.0).collect()
    });

    // what this face's own skin colour comes to, left in scratch for the passes that read against it
    let low = enc.area_down(src, low_size, RGB_MODE);
    let stats = enc.buffer("Skin Stats", bytemuck::cast_slice(&[0u32; 4]));
    enc.run(
        Group::Skin,
        "sk_stats",
        &params,
        low_size,
        &[&low.view, &face.view],
        Out::None,
        Bind {
            scratch: Some(&stats),
            groups: Some((1, 1)),
            ..Default::default()
        },
    );

    // Brightness, the two colour axes and the skin weight, so the blurs that follow average exactly
    // the quantities the maps are made of. The blurring takes the picture it is given, so the split is
    // laid down once for each size wanted.
    let split = |enc: &mut ComputeEncoder, label: &'static str| {
        let tex = enc.texture(label, size, f32x4);
        enc.run(
            Group::Skin,
            "sk_split",
            &params,
            size,
            &[&src.view, &face.view],
            Out::F32(&tex.view),
            Bind::default(),
        );
        tex
    };

    // Three sizes: the grain, the blemish, and the lie of the face. A blemish is a blob of brightness
    // at its own size, so it shows up between the first two and not in the third.
    let long = size.0.max(size.1) as f32;
    let fine_radius = ((long * 0.004).round() as u32).clamp(2, 24);
    let wide_radius = ((long * 0.022).round() as u32).clamp(6, 90);
    let here = split(enc, "Skin Split");
    let fine_src = split(enc, "Skin Fine");
    let wide_src = split(enc, "Skin Wide");
    let fine = enc.blur(fine_src, fine_radius, 2, 0);
    let wide = enc.blur(wide_src, wide_radius, 2, 0);

    let maps = enc.texture("Skin Maps", size, f32x4);
    enc.run(
        Group::Skin,
        "sk_maps",
        &params,
        size,
        &[&here.view, &fine.view, &wide.view],
        Out::F32(&maps.view),
        Bind {
            scratch: Some(&stats),
            ..Default::default()
        },
    );

    enc.run(
        Group::Skin,
        "sk_apply",
        &params,
        size,
        &[&src.view, &maps.view, &fine.view, &wide.view],
        Out::F16(&dst.view),
        Bind {
            scratch: Some(&stats),
            ..Default::default()
        },
    );
}
