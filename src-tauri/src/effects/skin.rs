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
    region_size: [u32; 2],
    redness_way: u32,
    lines: f32,
    under_eye: f32,
    // how far apart, in pixels, the curvature of each of the three blurred copies is read
    step_a: u32,
    step_b: u32,
    step_c: u32,
}

// the widths lines and blemishes are looked for at, as a share of the face's width: a fine crease or a
// pore-sized spot, a crease or a mole, a fold or a pimple. They are standard deviations; each copy of the
// picture is blurred to about this much.
const DETAIL_WIDTHS: [f32; 3] = [0.0025, 0.005, 0.010];

// where a face folds and where it shades under the eyes: two soft maps made from the landmarks
struct Regions {
    lines: Arc<GrayImage>,
    under_eye: Arc<GrayImage>,
    key: u64,
}

pub struct Skin {
    face_map: Arc<GrayImage>,
    map_key: u64,
    regions: Option<Regions>,
    // how wide the face is, as a fraction of the picture, which is what the widths of lines go by
    face_fraction: f32,
    spots: f32,
    redness: f32,
    shine: f32,
    lines: f32,
    under_eye: f32,
    keep_texture: f32,
    redness_way: u32,
}

// The width of the face as a fraction of the picture's width, from the area of the largest blob on the
// map: a face is about as tall as it is wide and a third again, with the eyes and the mouth left out, so
// its width comes to the square root of its area over a little under one. Going by the area and not by the
// extent along the picture's width means a head turned on its side is measured the same as an upright one.
// Lines and blemishes are a fixed share of a face, so they have to be looked for at a size that follows the
// face and not the frame - a small face in a big picture has thinner lines in pixels than a large one.
fn face_fraction(map: &GrayImage) -> f32 {
    const SIDE: u32 = 128;
    const FALLBACK: f32 = 0.2;
    let (mw, mh) = map.dimensions();
    let scale = (SIDE as f32 / mw.max(mh) as f32).min(1.0);
    let (gw, gh) = (
        ((mw as f32 * scale).round() as u32).max(1),
        ((mh as f32 * scale).round() as u32).max(1),
    );
    let on = |x: u32, y: u32| {
        let (sx, sy) = (
            (((x as f32 + 0.5) / gw as f32 * mw as f32) as u32).min(mw - 1),
            (((y as f32 + 0.5) / gh as f32 * mh as f32) as u32).min(mh - 1),
        );
        map.get_pixel(sx, sy)[0] > 127
    };
    let mut seen = vec![false; (gw * gh) as usize];
    let mut best = 0u32;
    for start_y in 0..gh {
        for start_x in 0..gw {
            let start = (start_y * gw + start_x) as usize;
            if seen[start] || !on(start_x, start_y) {
                continue;
            }
            seen[start] = true;
            let mut stack = vec![(start_x, start_y)];
            let mut area = 0u32;
            while let Some((x, y)) = stack.pop() {
                area += 1;
                let around = [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ];
                for (nx, ny) in around {
                    if nx >= gw || ny >= gh {
                        continue;
                    }
                    let index = (ny * gw + nx) as usize;
                    if !seen[index] && on(nx, ny) {
                        seen[index] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            best = best.max(area);
        }
    }
    if best == 0 {
        return FALLBACK;
    }
    let area = best as f32 / (gw * gh) as f32 * (mw as f32 * mh as f32);
    ((area / 0.92).sqrt() / mw as f32).clamp(0.02, 1.0)
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
    let lines = get("skinLines");
    let under_eye = get("skinUnderEye");
    if spots <= 0.0 && redness <= 0.0 && shine <= 0.0 && lines <= 0.0 && under_eye <= 0.0 {
        return None;
    }
    if w < 8 || h < 8 {
        return None;
    }
    let face_map = crate::effects::maps::resolve_luma_map(face_b64, adjustments, geometry)?;
    if face_map.width() < 2 || face_map.height() < 2 {
        return None;
    }
    // An edit saved before these maps existed has a face map and not these: the lines are then looked
    // for everywhere on the skin alike, and there is no under-eye band to correct.
    let map_of = |key: &str| {
        let data = adjustments[key].as_str().filter(|map| !map.is_empty())?;
        let map = crate::effects::maps::resolve_luma_map(data, adjustments, geometry)?;
        let key = crate::effects::maps::effect_map_key(data, adjustments);
        Some((map, key))
    };
    let regions = match (map_of("skinLinesMap"), map_of("skinUnderEyeMap")) {
        (Some((lines, lines_key)), Some((under_eye, under_eye_key)))
            if lines.dimensions() == under_eye.dimensions() && lines.width() >= 2 =>
        {
            Some(Regions {
                lines,
                under_eye,
                key: lines_key ^ under_eye_key.rotate_left(17),
            })
        }
        _ => None,
    };
    Some(Skin {
        map_key: crate::effects::maps::effect_map_key(face_b64, adjustments),
        face_fraction: face_fraction(&face_map),
        face_map,
        regions,
        spots,
        redness,
        shine,
        lines,
        under_eye,
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

    let face = enc.cached_map("Skin Face Map", job.map_key, (fw, fh), r32, || {
        job.face_map.pixels().map(|p| p[0] as f32 / 255.0).collect()
    });

    // the two region maps travel as one texture, the folds in the first channel and the under-eye band
    // in the second; without them a single empty texel stands in, which reads as no region at all
    let rg32 = wgpu::TextureFormat::Rg32Float;
    let (region_size, regions) = match &job.regions {
        Some(regions) => {
            let size = regions.lines.dimensions();
            let texture = enc.cached_map("Skin Regions", regions.key, size, rg32, || {
                regions
                    .lines
                    .pixels()
                    .zip(regions.under_eye.pixels())
                    .flat_map(|(lines, under_eye)| {
                        [lines[0] as f32 / 255.0, under_eye[0] as f32 / 255.0]
                    })
                    .collect()
            });
            (size, texture)
        }
        None => (
            (1, 1),
            enc.cached_map("Skin Regions", 0, (1, 1), rg32, || vec![0.45, 0.0]),
        ),
    };

    // Everything a face does is measured in faces, not in pixels: a blemish, a crease and the shading of
    // a cheek are a fixed share of its width, so a small face in a big picture and a large one are
    // looked at the same way.
    let face_px = (job.face_fraction * size.0 as f32).max(24.0);
    let step = |share: f32| ((face_px * share).round() as u32).max(1);

    let params = Params {
        grid: Grid::new(size),
        spots: job.spots,
        redness: job.redness,
        shine: job.shine,
        keep_texture: job.keep_texture,
        low_size: [low_size.0, low_size.1],
        face_size: [fw, fh],
        region_size: [region_size.0, region_size.1],
        redness_way: job.redness_way,
        lines: job.lines,
        under_eye: job.under_eye,
        step_a: step(DETAIL_WIDTHS[0]),
        step_b: step(DETAIL_WIDTHS[1]),
        step_c: step(DETAIL_WIDTHS[2]),
    };

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
    let fine_radius = ((face_px * 0.012).round() as u32).clamp(2, 60);
    let wide_radius = ((face_px * 0.066).round() as u32).clamp(6, 150);
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

    // Lines, blemishes and the under-eye band cost several more passes, so each is only looked for when
    // asked for. They all read the picture blurred to the same three widths.
    //
    // A line is a ridge in the brightness: dark along a thin curve, so across it the brightness bends
    // sharply upward and along it hardly at all. A blemish is a blob: it bends the same way in every
    // direction. Both are read from the curvature of the blurred copies, which is what tells a crease
    // from a spot from a pore, and the blur is what keeps noise out of it.
    let wants_lines = job.lines > 0.0 || job.under_eye > 0.0;
    let wants_blemishes = job.spots > 0.0;
    let mut extras = None;
    let mut blemishes = None;
    if wants_lines || wants_blemishes {
        // a blur of this radius spreads about as far as a gaussian a little under it
        let radius = |share: f32| ((face_px * share * 1.2).round() as u32).clamp(1, 150);
        let copy_a = split(enc, "Skin Detail A");
        let copy_b = split(enc, "Skin Detail B");
        let copy_c = split(enc, "Skin Detail C");
        let ridge_a = enc.blur(copy_a, radius(DETAIL_WIDTHS[0]), 2, 0);
        let ridge_b = enc.blur(copy_b, radius(DETAIL_WIDTHS[1]), 2, 0);
        let ridge_c = enc.blur(copy_c, radius(DETAIL_WIDTHS[2]), 2, 0);

        // how rough this skin's finest detail is from place to place: the yardstick a crease or a
        // blemish is read against
        let band = enc.texture("Skin Band", size, f32x4);
        enc.run(
            Group::Skin,
            "sk_band",
            &params,
            size,
            &[&here.view, &ridge_a.view],
            Out::F32(&band.view),
            Bind::default(),
        );
        let spread = enc.blur(band, radius(0.09), 2, 0);

        if wants_lines {
            let lines = enc.texture("Skin Extras", size, f32x4);
            enc.run(
                Group::Skin,
                "sk_lines",
                &params,
                size,
                &[
                    &ridge_a.view,
                    &ridge_b.view,
                    &ridge_c.view,
                    &spread.view,
                    &regions.view,
                    &wide.view,
                ],
                Out::F32(&lines.view),
                Bind {
                    scratch: Some(&stats),
                    ..Default::default()
                },
            );
            // softened a little, so where a line starts and stops is not a hard edge on the skin
            extras = Some(enc.blur(lines, radius(DETAIL_WIDTHS[0]), 1, 0));
        }

        if wants_blemishes {
            // the blemishes found, twice: once as they are, once to be blurred out so the whole of each
            // blemish is covered and not only its middle
            let found = enc.texture("Skin Blemishes", size, f32x4);
            let found_wide = enc.texture("Skin Blemishes Wide", size, f32x4);
            for target in [&found, &found_wide] {
                enc.run(
                    Group::Skin,
                    "sk_blob",
                    &params,
                    size,
                    &[&ridge_a.view, &ridge_b.view, &ridge_c.view, &spread.view],
                    Out::F32(&target.view),
                    Bind {
                        scratch: Some(&stats),
                        ..Default::default()
                    },
                );
            }
            let found_wide = enc.blur(found_wide, radius(DETAIL_WIDTHS[1]), 2, 0);
            let healed = enc.texture("Skin Heal", size, f32x4);
            enc.run(
                Group::Skin,
                "sk_heal",
                &params,
                size,
                &[&found.view, &found_wide.view],
                Out::F32(&healed.view),
                Bind::default(),
            );
            blemishes = Some(healed);
        }
    }

    enc.run(
        Group::Skin,
        "sk_apply",
        &params,
        size,
        &[
            &src.view,
            &maps.view,
            &fine.view,
            &wide.view,
            // when lines and the under-eye band are not asked for the apply pass never reads this one,
            // nor the next when spots are not
            extras.as_ref().map_or(&maps.view, |extras| &extras.view),
            blemishes
                .as_ref()
                .map_or(&maps.view, |blemishes| &blemishes.view),
        ],
        Out::F16(&dst.view),
        Bind {
            scratch: Some(&stats),
            ..Default::default()
        },
    );
}
#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::{Arc, Mutex, OnceLock};

    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use half::f16;
    use image::{DynamicImage, ImageFormat, Rgb32FImage};
    use serde_json::json;

    use super::prepare;
    use crate::render::gpu::compute::{self, IMAGE_FORMAT, Slot};
    use crate::render::gpu::context::GpuContext;
    use crate::render::source::{RenderSource, SourceTransform};

    const N: u32 = 384;
    const SPOT: (f32, f32) = (150.0, 150.0);
    const SHINE: (f32, f32) = (260.0, 200.0);
    const FLUSH: (f32, f32) = (110.0, 290.0);

    fn gpu() -> GpuContext {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
            .expect("no wgpu adapter (install a Vulkan driver, e.g. mesa lavapipe)");
        eprintln!("adapter: {:?}", adapter.get_info().name);
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .expect("device");
        GpuContext {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            limits,
            display: Arc::new(Mutex::new(None)),
            viewer_display: Arc::new(Mutex::new(None)),
            pipelines: Arc::new(OnceLock::new()),
            compute: Arc::new(OnceLock::new()),
        }
    }

    fn noise(x: u32, y: u32) -> f32 {
        let mut h = x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263);
        h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        ((h ^ (h >> 16)) & 0xffff) as f32 / 65535.0 - 0.5
    }

    // flat skin tone with grain, one dark red blemish and one shiny patch
    fn portrait() -> Rgb32FImage {
        let mut img = Rgb32FImage::new(N, N);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let g = noise(x, y) * 0.02;
            let (mut r, mut gr, mut b) = (0.42 + g, 0.27 + g, 0.20 + g);
            let d = |c: (f32, f32)| ((x as f32 - c.0).powi(2) + (y as f32 - c.1).powi(2)).sqrt();
            let spot = (-(d(SPOT) / 12.0).powi(2)).exp();
            r *= 1.0 - 0.25 * spot;
            gr *= 1.0 - 0.55 * spot;
            b *= 1.0 - 0.55 * spot;
            let shine = (-(d(SHINE) / 6.0).powi(2)).exp();
            r += 0.30 * shine;
            gr += 0.30 * shine;
            b += 0.28 * shine;
            let flush = (-(d(FLUSH) / 30.0).powi(2)).exp();
            r *= 1.0 + 0.30 * flush;
            gr *= 1.0 - 0.12 * flush;
            b *= 1.0 - 0.12 * flush;
            px.0 = [r, gr, b];
        }
        img
    }

    fn face_map() -> String {
        let map = image::GrayImage::from_pixel(N, N, image::Luma([255]));
        let mut buf = Cursor::new(Vec::new());
        map.write_to(&mut buf, ImageFormat::Png).unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(buf.into_inner())
        )
    }

    fn read(context: &GpuContext, slot: &Slot) -> Vec<[f32; 3]> {
        let (w, h) = slot.size;
        let row = (w * 8).next_multiple_of(256);
        let buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = context.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            slot.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        context.queue.submit(Some(enc.finish()));
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            let line = &data[(y * row) as usize..(y * row + w * 8) as usize];
            for px in line.as_chunks::<8>().0.iter() {
                let c = |i: usize| f16::from_le_bytes([px[i * 2], px[i * 2 + 1]]).to_f32();
                out.push([c(0), c(1), c(2)]);
            }
        }
        out
    }

    fn render(
        context: &GpuContext,
        img: &Rgb32FImage,
        adjustments: serde_json::Value,
    ) -> Vec<[f32; 3]> {
        render_sized(context, img, adjustments)
    }

    fn render_sized(
        context: &GpuContext,
        img: &Rgb32FImage,
        adjustments: serde_json::Value,
    ) -> Vec<[f32; 3]> {
        assert_eq!(IMAGE_FORMAT, wgpu::TextureFormat::Rgba16Float);
        let dynamic = DynamicImage::ImageRgb32F(img.clone());
        let slot =
            compute::upload_image(context, img.dimensions(), &compute::rgba_halves(&dynamic));
        let source = RenderSource::new(
            &dynamic,
            1,
            &adjustments,
            SourceTransform {
                effects: true,
                ..Default::default()
            },
        );
        let out = source.render(context, &slot).expect("render");
        read(context, &out)
    }

    fn luma(p: [f32; 3]) -> f32 {
        0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
    }

    fn mean(px: &[[f32; 3]], c: (f32, f32), r: f32) -> [f32; 3] {
        let (mut s, mut n) = ([0.0; 3], 0.0);
        for y in 0..N {
            for x in 0..N {
                if ((x as f32 - c.0).powi(2) + (y as f32 - c.1).powi(2)).sqrt() <= r {
                    let p = px[(y * N + x) as usize];
                    (0..3).for_each(|i| s[i] += p[i]);
                    n += 1.0;
                }
            }
        }
        s.map(|v| v / n)
    }

    fn adj(extra: serde_json::Value) -> serde_json::Value {
        let mut v =
            json!({ "skinEnabled": true, "skinFaceMap": face_map(), "skinKeepTexture": 50 });
        for (k, val) in extra.as_object().unwrap() {
            v[k] = val.clone();
        }
        v
    }

    #[test]
    fn skin_runs_on_the_gpu_and_corrects_each_fault() {
        let context = gpu();
        let img = portrait();
        let input: Vec<[f32; 3]> = img.pixels().map(|p| p.0).collect();

        // nothing is applied until a slider moves
        assert!(prepare(&adj(json!({})), None, (N, N)).is_none());

        let spots = render(&context, &img, adj(json!({ "skinSpots": 100 })));
        let shine = render(&context, &img, adj(json!({ "skinShine": 100 })));
        let redness = render(
            &context,
            &img,
            adj(json!({ "skinRedness": 100, "skinRednessWay": 0 })),
        );
        for out in [&spots, &shine, &redness] {
            assert!(
                out.iter().all(|p| p.iter().all(|v| v.is_finite())),
                "non-finite output"
            );
        }

        let bg = mean(&input, (330.0, 330.0), 25.0);
        let in_spot = mean(&input, SPOT, 4.0);
        let out_spot = mean(&spots, SPOT, 4.0);
        eprintln!("bg {bg:?}\nspot in {in_spot:?} out {out_spot:?}");
        assert!(
            (luma(out_spot) - luma(bg)).abs() < (luma(in_spot) - luma(bg)).abs() * 0.6,
            "spot not reduced"
        );

        let in_shine = mean(&input, SHINE, 5.0);
        let out_shine = mean(&shine, SHINE, 5.0);
        eprintln!("shine in {in_shine:?} out {out_shine:?}");
        assert!(luma(out_shine) < luma(in_shine) - 0.03, "shine not reduced");

        // far from every fault, and with only spots on, the skin is left essentially alone
        let far_in = mean(&input, (320.0, 60.0), 25.0);
        let far_out = mean(&spots, (320.0, 60.0), 25.0);
        eprintln!("far in {far_in:?} out {far_out:?}");
        for i in 0..3 {
            assert!((far_in[i] - far_out[i]).abs() < 0.02, "clean skin changed");
        }

        // redness: the flush comes back towards the skin around it, brightness stays put
        let ratio = |p: [f32; 3]| p[0] / (p[0] + p[1] + p[2]);
        let r_in = mean(&input, FLUSH, 10.0);
        let r_out = mean(&redness, FLUSH, 10.0);
        eprintln!(
            "flush in {r_in:?} out {r_out:?} (bg ratio {:.3})",
            ratio(bg)
        );
        assert!(ratio(r_in) - ratio(r_out) > 0.01, "flush not reduced");
        assert!(ratio(r_out) >= ratio(bg) - 0.01, "flush overcorrected");
        assert!(
            (luma(r_in) - luma(r_out)).abs() < 0.03,
            "redness changed brightness"
        );

        for way in 0..4 {
            let out = render(
                &context,
                &img,
                adj(json!({ "skinRedness": 80, "skinRednessWay": way })),
            );
            assert!(
                out.iter().all(|p| p.iter().all(|v| v.is_finite())),
                "way {way}"
            );
        }
    }

    const CREASE_Y: u32 = 290;
    const BAG: (f32, f32) = (90.0, 90.0);
    const MOLE: (f32, f32) = (330.0, 120.0);

    // the flat portrait with a thin dark crease across it and a shadow under an eye
    fn portrait_with_folds() -> Rgb32FImage {
        let mut img = portrait();
        for (x, y, px) in img.enumerate_pixels_mut() {
            if (60..330).contains(&x) && (CREASE_Y..CREASE_Y + 3).contains(&y) {
                px.0 = px.0.map(|v| v * 0.82);
            }
            let (dx, dy) = ((x as f32 - BAG.0) / 7.0, (y as f32 - BAG.1) / 3.5);
            let bag = (-(dx * dx + dy * dy)).exp();
            px.0 = px.0.map(|v| v * (1.0 - 0.25 * bag));
            // a small dark red mole
            let mole =
                (-(((x as f32 - MOLE.0).powi(2) + (y as f32 - MOLE.1).powi(2)) / 18.0)).exp();
            px.0 = [
                px.0[0] * (1.0 - 0.4 * mole),
                px.0[1] * (1.0 - 0.65 * mole),
                px.0[2] * (1.0 - 0.65 * mole),
            ];
        }
        img
    }

    fn gray_map(value: impl Fn(u32, u32) -> u8) -> String {
        let map = image::GrayImage::from_fn(N, N, |x, y| image::Luma([value(x, y)]));
        let mut buf = Cursor::new(Vec::new());
        map.write_to(&mut buf, ImageFormat::Png).unwrap();
        format!(
            "data:image/png;base64,{}",
            STANDARD.encode(buf.into_inner())
        )
    }

    fn with_regions(extra: serde_json::Value) -> serde_json::Value {
        let mut v = adj(extra);
        v["skinLinesMap"] = json!(gray_map(|_, _| 255));
        v["skinUnderEyeMap"] = json!(gray_map(|x, y| {
            let d = ((x as f32 - BAG.0).powi(2) + (y as f32 - BAG.1).powi(2)).sqrt();
            if d < 30.0 { 255 } else { 0 }
        }));
        v
    }

    #[test]
    fn lines_and_under_eyes_are_filled_back_towards_the_skin_around_them() {
        let context = gpu();
        let img = portrait_with_folds();
        let input: Vec<[f32; 3]> = img.pixels().map(|p| p.0).collect();

        // like every other slider, neither does anything until it moves
        assert!(prepare(&with_regions(json!({})), None, (N, N)).is_none());
        assert!(prepare(&with_regions(json!({ "skinLines": 10 })), None, (N, N)).is_some());
        assert!(prepare(&with_regions(json!({ "skinUnderEye": 10 })), None, (N, N)).is_some());

        let lines = render(&context, &img, with_regions(json!({ "skinLines": 100 })));
        let under = render(&context, &img, with_regions(json!({ "skinUnderEye": 100 })));
        // an edit saved before the region maps existed still renders, with the lines weighted evenly
        let bare = render(
            &context,
            &img,
            adj(json!({ "skinLines": 100, "skinUnderEye": 100 })),
        );
        for out in [&lines, &under, &bare] {
            assert!(
                out.iter().all(|p| p.iter().all(|v| v.is_finite())),
                "non-finite output"
            );
        }

        // where, if anywhere, did Lines change the picture at all
        let mut changed = (0.0f32, 0usize, 0u32, 0u32);
        for (i, (a, b)) in input.iter().zip(&lines).enumerate() {
            let d = (0..3).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max);
            if d > changed.0 {
                changed = (d, i, i as u32 % N, i as u32 / N);
            }
        }
        eprintln!("largest change by Lines {changed:?}");
        let crease = (240.0, CREASE_Y as f32 + 1.0);
        let beside = mean(&input, (240.0, 320.0), 10.0);
        let crease_in = mean(&input, crease, 1.5);
        let crease_out = mean(&lines, crease, 1.5);
        eprintln!("crease in {crease_in:?} out {crease_out:?} beside {beside:?}");
        assert!(
            luma(beside) - luma(crease_out) < (luma(beside) - luma(crease_in)) * 0.9,
            "crease not filled"
        );

        let shadow_beside = mean(&input, (90.0, 135.0), 8.0);
        let bag_in = mean(&input, BAG, 2.0);
        let bag_out = mean(&under, BAG, 2.0);
        eprintln!("bag in {bag_in:?} out {bag_out:?} beside {shadow_beside:?}");
        assert!(
            luma(shadow_beside) - luma(bag_out) < (luma(shadow_beside) - luma(bag_in)) * 0.9,
            "under-eye shadow not lifted"
        );

        // far from the crease and the shadow the skin is left essentially alone
        let far_in = mean(&input, (320.0, 60.0), 25.0);
        for out in [&lines, &under] {
            let far_out = mean(out, (320.0, 60.0), 25.0);
            for i in 0..3 {
                assert!((far_in[i] - far_out[i]).abs() < 0.02, "clean skin changed");
            }
        }
    }

    // Not a test: renders the skin tools on a real photograph, through the real face detection, so the
    // result can be looked at. It writes before/after crops of the face as PNG files.
    //
    //   RRAW_PHOTO=<file> RRAW_OUT=<folder> ORT_DYLIB_PATH=<onnxruntime.dll> RRAW_CASES=lines:50,lines:100 \
    //     cargo test --lib real_photo_preview -- --ignored --nocapture
    #[test]
    #[ignore]
    fn real_photo_preview() {
        use std::path::PathBuf;

        use crate::ai::commands::{MaskTransform, compute_face_analysis};
        use crate::ai::models::AiModels;
        use crate::masks::face::{FacePart, compose_face_mask, compose_skin_regions};

        let Ok(photo) = std::env::var("RRAW_PHOTO") else {
            return;
        };
        let out_dir = PathBuf::from(std::env::var("RRAW_OUT").expect("RRAW_OUT"));
        std::fs::create_dir_all(&out_dir).unwrap();
        let app_data =
            PathBuf::from(std::env::var("APPDATA").unwrap()).join("io.github.CyberTimon.RapidRAW");
        let settings: crate::app::settings::AppSettings =
            serde_json::from_slice(&std::fs::read(app_data.join("settings.json")).unwrap())
                .unwrap();

        // the raw developer is called directly: the app's full image loader pulls in dialog code that a
        // test executable, which has no Windows manifest, cannot even start with
        let bytes = std::fs::read(&photo).unwrap();
        let is_raw = crate::decode::formats::is_raw_file(&photo);
        let base = if is_raw {
            crate::decode::raw_processing::develop_raw_image(
                &bytes,
                false,
                settings.raw_highlight_compression.unwrap_or(2.5),
                settings.linear_raw_mode.clone(),
                settings.raw_preprocessing_color_nr.unwrap_or(0.5),
                settings.raw_preprocessing_sharpening.unwrap_or(0.35),
                None,
            )
            .unwrap()
        } else {
            image::load_from_memory(&bytes).unwrap()
        };
        let converted =
            crate::masks::generation::build_converted_source_image(&base, is_raw, &json!({}))
                .into_owned();
        let (width, height) = (converted.width(), converted.height());
        eprintln!("photo {width}x{height}");

        let _ = ort::init().with_name("preview").commit();
        let models = AiModels::new(app_data.join("models"), false);
        let transform = MaskTransform {
            rotation: 0.0,
            flip_horizontal: false,
            flip_vertical: false,
            orientation_steps: 0,
        };
        let started = std::time::Instant::now();
        let analysis =
            compute_face_analysis(&Mutex::new(None), &models, "preview", &converted, transform)
                .expect("face analysis");
        eprintln!(
            "found {} face(s) in {:?}",
            analysis.faces.len(),
            started.elapsed()
        );
        let face_map = compose_face_mask(&converted, &analysis, &[FacePart::FaceSkin]);
        let regions = compose_skin_regions(width, height, &analysis);

        // the face, with room around it
        let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0, 0);
        for (x, y, p) in face_map.enumerate_pixels() {
            if p[0] > 127 {
                (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            }
        }
        let (pad_x, pad_y) = ((x1 - x0) / 5, (y1 - y0) / 5);
        let (x0, y0) = (x0.saturating_sub(pad_x), y0.saturating_sub(pad_y));
        let (x1, y1) = ((x1 + pad_x).min(width - 1), (y1 + pad_y).min(height - 1));
        let (cw, ch) = (x1 - x0 + 1, y1 - y0 + 1);
        eprintln!("crop {x0},{y0} {cw}x{ch}");
        let crop = converted.crop_imm(x0, y0, cw, ch).to_rgb32f();
        let face_crop = image::imageops::crop_imm(&face_map, x0, y0, cw, ch).to_image();
        let region_crop = |map: &image::GrayImage| {
            let (rw, rh) = map.dimensions();
            let (fx0, fy0) = (x0 as f32 / width as f32, y0 as f32 / height as f32);
            let (fx1, fy1) = (
                (x1 + 1) as f32 / width as f32,
                (y1 + 1) as f32 / height as f32,
            );
            let (rx0, ry0) = ((fx0 * rw as f32) as u32, (fy0 * rh as f32) as u32);
            let (rx1, ry1) = (
                ((fx1 * rw as f32).ceil() as u32).min(rw),
                ((fy1 * rh as f32).ceil() as u32).min(rh),
            );
            image::imageops::crop_imm(map, rx0, ry0, (rx1 - rx0).max(1), (ry1 - ry0).max(1))
                .to_image()
        };
        let data_url = |map: &image::GrayImage| {
            let mut buf = Cursor::new(Vec::new());
            map.write_to(&mut buf, ImageFormat::Png).unwrap();
            format!(
                "data:image/png;base64,{}",
                STANDARD.encode(buf.into_inner())
            )
        };
        let save = |name: &str, px: &[[f32; 3]]| {
            let mut out = image::RgbImage::new(cw, ch);
            for (i, p) in px.iter().enumerate() {
                let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
                out.put_pixel(
                    i as u32 % cw,
                    i as u32 / cw,
                    image::Rgb([c(p[0]), c(p[1]), c(p[2])]),
                );
            }
            out.save(out_dir.join(format!("{name}.png"))).unwrap();
        };

        let context = gpu();
        let input: Vec<[f32; 3]> = crop.pixels().map(|p| p.0).collect();
        save("before", &input);

        let cases = std::env::var("RRAW_CASES").unwrap_or_else(|_| "lines:100".into());
        for case in cases.split(',') {
            let mut adjustments = json!({
                "skinEnabled": true,
                "skinFaceMap": data_url(&face_crop),
                "skinLinesMap": data_url(&region_crop(&regions.lines)),
                "skinUnderEyeMap": data_url(&region_crop(&regions.under_eye)),
                "skinKeepTexture": 50,
            });
            for setting in case.split('+') {
                let (key, value) = setting.split_once(':').expect("key:value");
                let key = match key {
                    "lines" => "skinLines",
                    "undereye" => "skinUnderEye",
                    "spots" => "skinSpots",
                    "shine" => "skinShine",
                    "redness" => "skinRedness",
                    "texture" => "skinKeepTexture",
                    other => other,
                };
                adjustments[key] = json!(value.parse::<f64>().unwrap());
            }
            let started = std::time::Instant::now();
            let out = render_sized(&context, &crop, adjustments);
            eprintln!("{case}: rendered in {:?}", started.elapsed());
            save(&case.replace([':', '+'], "_"), &out);
        }
    }

    #[test]
    fn a_blemish_is_healed_in_colour_as_well_as_brightness() {
        let context = gpu();
        let img = portrait_with_folds();
        let input: Vec<[f32; 3]> = img.pixels().map(|p| p.0).collect();
        let out = render(&context, &img, with_regions(json!({ "skinSpots": 100 })));
        assert!(out.iter().all(|p| p.iter().all(|v| v.is_finite())));

        let beside = mean(&input, (330.0, 150.0), 6.0);
        let mole_in = mean(&input, MOLE, 2.0);
        let mole_out = mean(&out, MOLE, 2.0);
        eprintln!("mole in {mole_in:?} out {mole_out:?} beside {beside:?}");
        for c in 0..3 {
            assert!(
                (mole_out[c] - beside[c]).abs() < (mole_in[c] - beside[c]).abs() * 0.5,
                "channel {c} not healed"
            );
        }

        // and the skin far from it is left alone
        let far_in = mean(&input, (320.0, 60.0), 25.0);
        let far_out = mean(&out, (320.0, 60.0), 25.0);
        for c in 0..3 {
            assert!((far_in[c] - far_out[c]).abs() < 0.02, "clean skin changed");
        }
    }

    #[test]
    fn the_face_fraction_follows_the_largest_blob() {
        let mut map = image::GrayImage::new(400, 400);
        for y in 100..300 {
            for x in 50..250 {
                map.put_pixel(x, y, image::Luma([255]));
            }
        }
        for y in 10..30 {
            for x in 300..390 {
                map.put_pixel(x, y, image::Luma([255]));
            }
        }
        let fraction = super::face_fraction(&map);
        assert!((fraction - 0.5).abs() < 0.05, "fraction {fraction}");
        assert!(super::face_fraction(&image::GrayImage::new(64, 64)) > 0.0);
    }
}
