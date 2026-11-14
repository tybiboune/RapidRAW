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
            let spot = (-(d(SPOT) / 7.0).powi(2)).exp();
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
            for px in line.chunks_exact(8) {
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
        assert_eq!(IMAGE_FORMAT, wgpu::TextureFormat::Rgba16Float);
        let dynamic = DynamicImage::ImageRgb32F(img.clone());
        let slot = compute::upload_image(context, (N, N), &compute::rgba_halves(&dynamic));
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
}
