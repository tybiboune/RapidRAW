use crate::ai::base_cache::{BaseCache, entry_key, source_string};
use crate::ai::denoise::DenoiseMethod;
use crate::app::state::{AppState, LoadedImage};
use crate::render::adjustment_utils::is_effect_enabled;
use image::imageops::{self, FilterType};
use image::{DynamicImage, GenericImageView, Rgb32FImage};
use rayon::prelude::*;
use std::borrow::Cow;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tauri::Manager;

const MODEL_TAG: &str = "upscale-v3";
// the four-times model is another network, so its results are cached apart
const MODEL_TAG_4X: &str = "upscale4-realplksr-v1";
/// The longest side an upscaled picture may have: the render works on GPU textures, which most cards cap
/// at this many pixels. A picture that would come out larger is upscaled two times instead of four.
const MAX_OUTPUT_SIDE: u32 = 16384;
const SUPERFOCUS_TAG: &str = "-superfocus-v1";
/// Superfocus upscales the picture this many rows at a time, with this much of the picture either side
/// for the network to see, and brings each band straight back down: a 24 MP picture four times larger
/// would be about 5 GB whole.
const SUPERFOCUS_BAND: u32 = 256;
const SUPERFOCUS_MARGIN: u32 = 16;
const GAMMA: f32 = 2.2;
const DETAIL_GAIN: f32 = 1.5;
const DETAIL_NOISE_FLOOR: f32 = 2.0;
const DETAIL_OVERSHOOT: f32 = 0.25;
const DETAIL_BAND: usize = 32;
const DETAIL_SAMPLE_STEP: usize = 8;

static CACHE: BaseCache = BaseCache::new(
    "ai_upscale",
    "rrup",
    b"RRUP0002",
    b"RRUP0001",
    "ai-upscale",
    "AI upscale",
);

type Session = Arc<Mutex<ort::session::Session>>;

pub fn initialize_cache_dir(cache_dir: &Path) {
    CACHE.initialize_dir(cache_dir);
}

pub fn set_disk_cache_limit_gb(limit_gb: u32) {
    CACHE.set_disk_limit_gb(limit_gb);
}

pub fn cache_epoch() -> u64 {
    CACHE.epoch()
}

/// Whether the picture goes through an upscaler at all: to be made larger, or for Superfocus.
pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    enlarges(adjustments) || superfocus(adjustments)
}

fn enlarges(adjustments: &serde_json::Value) -> bool {
    is_effect_enabled(adjustments, "superResolution", "aiUpscaleEnabled")
}

/// Superfocus: the picture is upscaled and brought straight back down to its own size, so the detail
/// the network draws is kept and the size is not. A picture also asked to be made larger is made
/// larger instead, which already carries that detail.
fn superfocus(adjustments: &serde_json::Value) -> bool {
    !enlarges(adjustments) && is_effect_enabled(adjustments, "superfocus", "superfocusEnabled")
}

fn superfocus_amount(adjustments: &serde_json::Value) -> f32 {
    (adjustments["superfocusAmount"].as_f64().unwrap_or(100.0) as f32 / 100.0).clamp(0.0, 1.0)
}

/// How many times larger the picture comes out: one for Superfocus, which keeps its size.
pub fn output_factor(adjustments: &serde_json::Value) -> u32 {
    if enlarges(adjustments) {
        requested_factor(adjustments)
    } else {
        1
    }
}

/// What the upscaler is asked for: the factor it runs at, and whether the result is brought back down
/// to the picture's own size (Superfocus).
#[derive(Clone, Copy)]
struct Plan {
    factor: u32,
    keep_size: bool,
}

fn plan(adjustments: &serde_json::Value, size: (u32, u32)) -> Plan {
    if superfocus(adjustments) {
        // brought back down band by band, so no larger picture is ever whole and the card's texture
        // limit does not apply: the network asked for is the one that runs
        let factor = if adjustments["superfocusUpscaler"].as_u64() == Some(4) {
            4
        } else {
            2
        };
        Plan {
            factor,
            keep_size: true,
        }
    } else {
        Plan {
            factor: factor(adjustments, size),
            keep_size: false,
        }
    }
}

fn denoise_input(adjustments: &serde_json::Value) -> Option<DenoiseMethod> {
    crate::ai::denoise::is_enabled(adjustments)
        .then(|| DenoiseMethod::from_adjustments(adjustments))
}

/// The factor asked for: four when the owner chose it, otherwise two.
pub fn requested_factor(adjustments: &serde_json::Value) -> u32 {
    if adjustments["aiUpscaleFactor"].as_u64() == Some(4) {
        4
    } else {
        2
    }
}

/// The factor a picture of this size is upscaled by: four when it is asked for and the result fits on
/// the card, otherwise two.
pub fn factor(adjustments: &serde_json::Value, (w, h): (u32, u32)) -> u32 {
    if requested_factor(adjustments) == 4 && w.max(h).saturating_mul(4) <= MAX_OUTPUT_SIDE {
        4
    } else {
        2
    }
}

fn key(path: &str, denoise: Option<DenoiseMethod>, plan: Plan) -> String {
    let input = denoise.map_or("none", DenoiseMethod::tag);
    let model = if plan.factor == 4 {
        MODEL_TAG_4X
    } else {
        MODEL_TAG
    };
    // a Superfocus result is the picture at its own size, so it is kept apart from the enlarged one
    let kind = if plan.keep_size { SUPERFOCUS_TAG } else { "" };
    entry_key(&source_string(path), &format!("{model}{kind}+{input}"))
}

/// What the picture is made of before it is upscaled, beyond its development and denoise: the negative
/// conversion, the skin renewal and the brushed repairs. None when none of them changes anything.
fn content_tag(adjustments: &serde_json::Value) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    if !crate::decode::image_loader::has_content_steps(adjustments) {
        return None;
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    if let Some(all) = adjustments.as_object() {
        for (name, value) in all {
            // the negative's settings, and where the repairs' masks fall on the turned or cropped picture
            let geometry = matches!(
                name.as_str(),
                "orientationSteps" | "flipHorizontal" | "flipVertical" | "rotation" | "crop"
            ) || name.starts_with("transform");
            if name.starts_with("negative") || geometry {
                name.hash(&mut hasher);
                value.to_string().hash(&mut hasher);
            }
        }
    }
    crate::effects::skin_method::hash_inputs(adjustments, &mut hasher, false);
    // only the repairs that show and have a result: one still being made, or hidden, changes nothing,
    // and its flags must not set the upscaler going again
    for patch in adjustments["aiPatches"].as_array().into_iter().flatten() {
        let visible = patch["visible"].as_bool().unwrap_or(true);
        let data = &patch["patchData"];
        if visible && data["color"].as_str().is_some_and(|c| !c.is_empty()) {
            patch["id"].to_string().hash(&mut hasher);
            data.to_string().hash(&mut hasher);
            patch["subMasks"].to_string().hash(&mut hasher);
            patch["invert"].to_string().hash(&mut hasher);
        }
    }
    Some(hasher.finish())
}

/// The key of a result made from the picture with its content steps, `tag` being their hash.
fn content_key(path: &str, denoise: Option<DenoiseMethod>, plan: Plan, tag: Option<u64>) -> String {
    match tag {
        None => key(path, denoise, plan),
        Some(tag) => format!("{}-c{tag:016x}", key(path, denoise, plan)),
    }
}

/// Paths whose upscale with repairs is being made: their keys depend on the repairs, so the path is kept.
static CONTENT_PENDING: Mutex<Vec<String>> = Mutex::new(Vec::new());

const REFRESH_BLOCK: u32 = 128;
const REFRESH_MARGIN: u32 = 24;

/// Whether any pixel of the block at (x0, y0) differs between the two pictures.
fn block_differs(a: &Rgb32FImage, b: &Rgb32FImage, x0: u32, y0: u32) -> bool {
    let (w, h) = a.dimensions();
    let (x1, y1) = ((x0 + REFRESH_BLOCK).min(w), (y0 + REFRESH_BLOCK).min(h));
    let row = w as usize * 3;
    (y0..y1).any(|y| {
        let span = y as usize * row + x0 as usize * 3..y as usize * row + x1 as usize * 3;
        a.as_raw()[span.clone()]
            .iter()
            .zip(&b.as_raw()[span])
            .any(|(p, q)| (p - q).abs() > 1e-6)
    })
}

/// The upscale of `after`, made from `out` (the upscale of `before`) by running the network again only
/// where the two pictures differ, block by block, with a little of the picture around for context. A
/// brush stroke costs its own area, not the whole picture.
fn refresh_changed(
    mut out: Rgb32FImage,
    before: &Rgb32FImage,
    after: &Rgb32FImage,
    upscale: &Upscale,
    scale: u32,
    on_progress: &(dyn Fn(f32) + Sync),
) -> Result<Rgb32FImage, String> {
    let (w, h) = after.dimensions();
    let (bw, bh) = (w.div_ceil(REFRESH_BLOCK), h.div_ceil(REFRESH_BLOCK));
    let differs: Vec<bool> = (0..bw * bh)
        .into_par_iter()
        .map(|i| {
            block_differs(
                before,
                after,
                (i % bw) * REFRESH_BLOCK,
                (i / bw) * REFRESH_BLOCK,
            )
        })
        .collect();
    // a block beside a changed one is done again too: the network sees across the border
    let changed = |bx: u32, by: u32| {
        (by.saturating_sub(1)..=(by + 1).min(bh - 1)).any(|y| {
            (bx.saturating_sub(1)..=(bx + 1).min(bw - 1)).any(|x| differs[(y * bw + x) as usize])
        })
    };
    let mut spans = Vec::new();
    for by in 0..bh {
        let mut bx = 0;
        while bx < bw {
            if changed(bx, by) {
                let start = bx;
                while bx < bw && changed(bx, by) {
                    bx += 1;
                }
                spans.push((by, start, bx));
            } else {
                bx += 1;
            }
        }
    }
    let count = spans.len().max(1) as f32;
    let out_row = out.width() as usize * 3;
    for (i, &(by, start, end)) in spans.iter().enumerate() {
        let (x0, y0) = (start * REFRESH_BLOCK, by * REFRESH_BLOCK);
        let (x1, y1) = (
            (end * REFRESH_BLOCK).min(w),
            ((by + 1) * REFRESH_BLOCK).min(h),
        );
        let (cx0, cy0) = (
            x0.saturating_sub(REFRESH_MARGIN),
            y0.saturating_sub(REFRESH_MARGIN),
        );
        let (cx1, cy1) = ((x1 + REFRESH_MARGIN).min(w), (y1 + REFRESH_MARGIN).min(h));
        let crop = imageops::crop_imm(after, cx0, cy0, cx1 - cx0, cy1 - cy0).to_image();
        let done = i as f32;
        let up = upscale(&crop, &|p: f32| on_progress((done + p) / count))?;
        let up_row = up.width() as usize * 3;
        let dst: &mut [f32] = &mut out;
        for oy in y0 * scale..y1 * scale {
            let sy = (oy - cy0 * scale) as usize;
            let from = &up.as_raw()[sy * up_row + ((x0 - cx0) * scale) as usize * 3..]
                [..((x1 - x0) * scale) as usize * 3];
            let at = oy as usize * out_row + (x0 * scale) as usize * 3;
            dst[at..at + from.len()].copy_from_slice(from);
        }
    }
    on_progress(1.0);
    Ok(out)
}

/// The upscale of the picture with its content steps (negative, renewal, repairs) done first. The plain
/// upscale is kept as well, so that the next repair only redoes its own area.
#[allow(clippy::too_many_arguments)]
fn run_with_content(
    base: &DynamicImage,
    path: &str,
    input: Option<DenoiseMethod>,
    plan: Plan,
    tag: Option<u64>,
    adjustments: &serde_json::Value,
    session: &Session,
    on_progress: &(dyn Fn(f32) + Sync),
) -> Result<Rgb32FImage, String> {
    let plain_input = input_for(base, path, input)?;
    let Some(_) = tag else {
        return run(&plain_input, session, plan, on_progress);
    };
    CONTENT_PENDING.lock().unwrap().push(path.to_string());
    let result = (|| {
        let content_input = crate::decode::image_loader::apply_content_steps(
            Cow::Owned(DynamicImage::ImageRgb32F(plain_input.clone())),
            base.dimensions(),
            path,
            adjustments,
        )
        .map_err(|e| e.to_string())?
        .into_owned()
        .into_rgb32f();
        let source = source_string(path);
        let size = upscaled_size(plan, base.dimensions());
        let plain_key = key(path, input, plan);
        let plain = match CACHE
            .load(&plain_key, &source)
            .filter(|entry| (entry.width, entry.height) == size)
        {
            Some(entry) => entry.to_rgb32f(),
            None => {
                let made = run(&plain_input, session, plan, &|p| on_progress(p * 0.5))?;
                CACHE.store(
                    crate::ai::base_cache::Job {
                        key: plain_key,
                        source,
                        fingerprint: crate::ai::base_cache::base_fingerprints(base)[0],
                    },
                    made.clone(),
                );
                made
            }
        };
        let scale = size.0 / base.width().max(1);
        refresh_changed(
            plain,
            &plain_input,
            &content_input,
            &|piece, progress| run(piece, session, plan, progress),
            scale,
            &|p| on_progress(0.5 + p * 0.5),
        )
    })();
    CONTENT_PENDING.lock().unwrap().retain(|p| p != path);
    result
}

fn upscaled_size(plan: Plan, (w, h): (u32, u32)) -> (u32, u32) {
    if plan.keep_size {
        (w, h)
    } else {
        (w * plan.factor, h * plan.factor)
    }
}

pub fn scale_between(base: &DynamicImage, prepared: &DynamicImage) -> f32 {
    if base.width() == 0 {
        1.0
    } else {
        prepared.width() as f32 / base.width() as f32
    }
}

// Where output pixel `o` of an upsampling by `factor` reads from a line of `len` source pixels: the two
// source pixels either side of its centre, and how far it is from the first to the second.
fn taps(o: usize, len: usize, factor: usize) -> (usize, usize, f32) {
    let x = ((o as f32 + 0.5) / factor as f32 - 0.5).max(0.0);
    let i = (x.floor() as usize).min(len - 1);
    (i, (i + 1).min(len - 1), (x - i as f32).min(1.0))
}

fn upsample_add(dst: &mut [f32], src: &[f32], width: usize, height: usize, factor: usize) {
    let out_w = width * factor;
    dst.par_chunks_exact_mut(out_w * 3)
        .enumerate()
        .for_each(|(oy, row)| {
            let (y0, y1, wy) = taps(oy, height, factor);
            let (r0, r1) = (&src[y0 * width * 3..], &src[y1 * width * 3..]);
            for ox in 0..out_w {
                let (x0, x1, wx) = taps(ox, width, factor);
                for c in 0..3 {
                    let near = r0[x0 * 3 + c] + (r0[x1 * 3 + c] - r0[x0 * 3 + c]) * wx;
                    let far = r1[x0 * 3 + c] + (r1[x1 * 3 + c] - r1[x0 * 3 + c]) * wx;
                    row[ox * 3 + c] += near + (far - near) * wy;
                }
            }
        });
}

fn luma(p: &[f32]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

fn detail_amount(adjustments: &serde_json::Value) -> f32 {
    (adjustments["aiUpscaleDetail"].as_f64().unwrap_or(30.0) as f32 / 100.0).clamp(0.0, 1.0)
}

struct DetailField<'a> {
    source_luma: &'a [f32],
    width: usize,
    height: usize,
    factor: usize,
}

struct DetailScratch {
    width: usize,
    blend: Vec<f32>,
    signal: Vec<f32>,
    residual: Vec<f32>,
    lo: Vec<f32>,
    hi: Vec<f32>,
    energy: Vec<f32>,
}

impl DetailScratch {
    fn new(width: usize, rows: usize, factor: usize) -> Self {
        let out_w = width * factor;
        let n = rows * out_w;
        Self {
            width: out_w,
            blend: vec![0.0; width],
            signal: vec![0.0; n],
            residual: vec![0.0; n],
            lo: vec![0.0; n],
            hi: vec![0.0; n],
            energy: vec![0.0; n],
        }
    }

    fn span(&self, slot: usize) -> std::ops::Range<usize> {
        slot * self.width..(slot + 1) * self.width
    }

    fn load(&mut self, slot: usize, (signal, residual): &(Vec<f32>, Vec<f32>)) {
        let span = self.span(slot);
        self.signal[span.clone()].copy_from_slice(signal);
        self.residual[span].copy_from_slice(residual);
        self.local_stats(slot);
    }

    fn local_stats(&mut self, slot: usize) {
        let span = self.span(slot);
        let (signal, residual) = (&self.signal[span.clone()], &self.residual[span.clone()]);
        let (lo, hi, energy) = (
            &mut self.lo[span.clone()],
            &mut self.hi[span.clone()],
            &mut self.energy[span],
        );
        let last = signal.len() - 1;
        for x in 0..signal.len() {
            let (l, r) = (x.saturating_sub(1), (x + 1).min(last));
            lo[x] = signal[l].min(signal[x]).min(signal[r]);
            hi[x] = signal[l].max(signal[x]).max(signal[r]);
            energy[x] =
                residual[l] * residual[l] + residual[x] * residual[x] + residual[r] * residual[r];
        }
    }
}

impl DetailField<'_> {
    fn out_width(&self) -> usize {
        self.width * self.factor
    }

    fn row(&self, oy: usize, pixels: &[f32], scratch: &mut DetailScratch, slot: usize) {
        let (y0, y1, wy) = taps(oy, self.height, self.factor);
        let near = &self.source_luma[y0 * self.width..][..self.width];
        let far = &self.source_luma[y1 * self.width..][..self.width];
        for ((b, n), f) in scratch.blend.iter_mut().zip(near).zip(far) {
            *b = n + (f - n) * wy;
        }
        let span = scratch.span(slot);
        let blend = &scratch.blend;
        let signal = scratch.signal[span.clone()].iter_mut();
        let residual = scratch.residual[span].iter_mut();
        for (ox, ((pixel, s), r)) in pixels
            .as_chunks::<3>()
            .0
            .iter()
            .zip(signal)
            .zip(residual)
            .enumerate()
        {
            let (x0, x1, wx) = taps(ox, self.width, self.factor);
            let up = blend[x0] + (blend[x1] - blend[x0]) * wx;
            *s = luma(pixel).max(0.0).sqrt();
            *r = *s - up.max(0.0).sqrt();
        }
    }

    fn row_owned(&self, oy: usize, pixels: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let mut scratch = DetailScratch::new(self.width, 1, self.factor);
        self.row(oy, pixels, &mut scratch, 0);
        (scratch.signal, scratch.residual)
    }

    fn noise_sigma(&self, pixels: &[f32]) -> f32 {
        let stride = self.out_width() * 3;
        let rows = pixels.len() / stride;
        let mut samples: Vec<f32> = (0..rows.div_ceil(DETAIL_SAMPLE_STEP))
            .into_par_iter()
            .flat_map_iter(|i| {
                let oy = i * DETAIL_SAMPLE_STEP;
                self.row_owned(oy, &pixels[oy * stride..][..stride])
                    .1
                    .into_iter()
                    .step_by(DETAIL_SAMPLE_STEP)
                    .map(f32::abs)
            })
            .collect();
        if samples.is_empty() {
            return 0.0;
        }
        let mid = samples.len() / 2;
        1.4826 * *samples.select_nth_unstable_by(mid, f32::total_cmp).1
    }
}

fn soft_limit(value: f32, lo: f32, hi: f32, knee: f32) -> f32 {
    if knee <= 1e-6 {
        value.clamp(lo, hi)
    } else if value > hi {
        hi + knee * (1.0 - (-(value - hi) / knee).exp())
    } else if value < lo {
        lo - knee * (1.0 - (-(lo - value) / knee).exp())
    } else {
        value
    }
}

fn enhance_detail(out: &mut Rgb32FImage, source: &Rgb32FImage, amount: f32) {
    let source_luma: Vec<f32> = source.as_raw().par_chunks_exact(3).map(luma).collect();
    let field = DetailField {
        source_luma: &source_luma,
        width: source.width() as usize,
        height: source.height() as usize,
        factor: (out.width() / source.width().max(1)) as usize,
    };
    let out_w = field.out_width();
    let stride = out_w * 3;
    let band_len = DETAIL_BAND * stride;

    let pixels = out.as_raw();
    let noise_energy = 9.0 * DETAIL_NOISE_FLOOR * field.noise_sigma(pixels).powi(2);
    let edges: Vec<[(Vec<f32>, Vec<f32>); 2]> = pixels
        .par_chunks(band_len)
        .enumerate()
        .map(|(band, chunk)| {
            let last = chunk.len() / stride - 1;
            [0, last]
                .map(|i| field.row_owned(band * DETAIL_BAND + i, &chunk[i * stride..][..stride]))
        })
        .collect();

    let strength = DETAIL_GAIN * amount;
    out.par_chunks_mut(band_len).enumerate().for_each_init(
        || DetailScratch::new(field.width, 3, field.factor),
        |scratch, (band, chunk)| {
            let first = band * DETAIL_BAND;
            let rows = chunk.len() / stride;
            scratch.load(
                0,
                if band > 0 {
                    &edges[band - 1][1]
                } else {
                    &edges[band][0]
                },
            );
            field.row(first, &chunk[..stride], scratch, 1);
            scratch.local_stats(1);
            for i in 0..rows {
                let next = (i + 2) % 3;
                if i + 1 < rows {
                    field.row(
                        first + i + 1,
                        &chunk[(i + 1) * stride..][..stride],
                        scratch,
                        next,
                    );
                    scratch.local_stats(next);
                } else {
                    scratch.load(next, edges.get(band + 1).map_or(&edges[band][1], |e| &e[0]));
                }
                let [a, b, c] = [i % 3, (i + 1) % 3, next].map(|slot| slot * out_w);
                let row = &mut chunk[i * stride..][..stride];
                for (x, px) in row.as_chunks_mut::<3>().0.iter_mut().enumerate() {
                    let s = scratch.signal[b + x];
                    let local_energy =
                        scratch.energy[a + x] + scratch.energy[b + x] + scratch.energy[c + x];
                    if s <= 1e-4 || local_energy <= noise_energy {
                        continue;
                    }
                    let weight = 1.0 - noise_energy / local_energy;
                    let lo = scratch.lo[a + x]
                        .min(scratch.lo[b + x])
                        .min(scratch.lo[c + x]);
                    let hi = scratch.hi[a + x]
                        .max(scratch.hi[b + x])
                        .max(scratch.hi[c + x]);
                    let target = s + strength * weight * scratch.residual[b + x];
                    let limited = soft_limit(target, lo, hi, DETAIL_OVERSHOOT * (hi - lo));
                    let gain = (limited.max(0.0) / s).powi(2);
                    px.iter_mut().for_each(|v| *v *= gain);
                }
            }
        },
    );
}

fn inputs(adjustments: &serde_json::Value) -> Vec<Option<DenoiseMethod>> {
    match denoise_input(adjustments) {
        Some(method) => vec![Some(method), None],
        None => vec![None],
    }
}

/// The picture through the upscaler (enlarged, or Superfocus), and whether the steps that belong before
/// it (negative conversion, skin renewal, brushed repairs) are already in it. While the result with this
/// set of repairs is being made, the one without them is given and the repairs go on top as before.
pub fn apply_ai_upscale<'a>(
    base: &DynamicImage,
    image: Cow<'a, DynamicImage>,
    path: &str,
    adjustments: &serde_json::Value,
) -> (Cow<'a, DynamicImage>, bool) {
    if !is_enabled(adjustments) {
        return (image, false);
    }
    let (w, h) = base.dimensions();
    if image.dimensions() != (w, h) {
        return (image, false);
    }
    let source = source_string(path);
    let plan = plan(adjustments, (w, h));
    let size = upscaled_size(plan, (w, h));
    let strength = crate::ai::denoise::blend_strength(adjustments);
    let pick = |tag: Option<u64>| {
        let load = |input| {
            CACHE
                .load(&content_key(path, input, plan, tag), &source)
                .filter(|entry| (entry.width, entry.height) == size)
        };
        match denoise_input(adjustments).filter(|_| strength.0 > 0.0) {
            None => load(None).map(|entry| entry.to_rgb32f()),
            Some(method) => load(Some(method)).and_then(|denoised| {
                if strength.0 >= 1.0 && strength.1 <= 0.0 {
                    return Some(denoised.to_rgb32f());
                }
                let mut out = load(None)?.to_rgb32f();
                crate::ai::denoise::blend_into(&mut out, &denoised.data, strength);
                Some(out)
            }),
        }
    };
    let tag = content_tag(adjustments);
    let (upscaled, content_done) = match tag.and_then(|tag| pick(Some(tag))) {
        Some(out) => (Some(out), tag.is_some()),
        None => (pick(None), false),
    };
    let Some(mut out) = upscaled else {
        return (image, false);
    };
    // what the result is compared with or blended with is the picture it was made from
    let source_image = if content_done {
        match crate::decode::image_loader::apply_content_steps(image, (w, h), path, adjustments) {
            Ok(made) => made,
            Err(e) => {
                log::warn!("The repairs could not be laid on the picture: {e}");
                return (Cow::Owned(DynamicImage::ImageRgb32F(out)), true);
            }
        }
    } else {
        image
    };
    if plan.keep_size {
        let amount = superfocus_amount(adjustments);
        if amount < 1.0 {
            let before = source_image.to_rgb32f();
            out.par_iter_mut()
                .zip(before.as_raw().par_iter())
                .for_each(|(v, &b)| *v = b + (*v - b) * amount);
        }
        return (Cow::Owned(DynamicImage::ImageRgb32F(out)), content_done);
    }
    let amount = detail_amount(adjustments);
    if amount > 0.0 {
        match source_image.as_rgb32f() {
            Some(source) => enhance_detail(&mut out, source, amount),
            None => enhance_detail(&mut out, &source_image.to_rgb32f(), amount),
        }
    }
    (Cow::Owned(DynamicImage::ImageRgb32F(out)), content_done)
}

fn input_for(
    base: &DynamicImage,
    path: &str,
    denoise: Option<DenoiseMethod>,
) -> Result<Rgb32FImage, String> {
    let Some(method) = denoise else {
        return Ok(base.to_rgb32f());
    };
    crate::ai::denoise::full_strength(path, method)
        .filter(|entry| (entry.width, entry.height) == base.dimensions())
        .map(|entry| entry.to_rgb32f())
        .ok_or_else(|| "AI denoise result is not available".to_string())
}

fn run_upscaler(
    input: &Rgb32FImage,
    session: &Session,
    factor: u32,
    on_progress: &(dyn Fn(f32) + Sync),
) -> Result<Rgb32FImage, String> {
    let (w, h) = input.dimensions();
    let mut encoded = input.clone();
    encoded
        .par_iter_mut()
        .for_each(|v| *v = v.clamp(0.0, 1.0).powf(1.0 / GAMMA));

    let mut output = crate::ai::tiling::run_ai_upscale(&encoded, session, factor, on_progress)
        .map_err(|e| e.to_string())?;
    drop(encoded);
    if output.dimensions() != (w * factor, h * factor) {
        return Err(format!(
            "Upscale model returned {}x{} for a {}x{} input",
            output.width(),
            output.height(),
            w,
            h
        ));
    }
    output.par_iter_mut().for_each(|v| *v = v.powf(GAMMA));

    let excess: Vec<f32> = input
        .as_raw()
        .par_iter()
        .map(|&v| v - v.clamp(0.0, 1.0))
        .collect();
    upsample_add(
        &mut output,
        &excess,
        w as usize,
        h as usize,
        factor as usize,
    );
    Ok(output)
}

/// Superfocus: the picture upscaled by `factor` a band at a time, each band brought back down to its own
/// size with a Lanczos filter in linear light, so what is left is the network's finer detail and edges
/// at the picture's own size.
fn run_superfocus(
    input: &Rgb32FImage,
    session: &Session,
    factor: u32,
    on_progress: &(dyn Fn(f32) + Sync),
) -> Result<Rgb32FImage, String> {
    superfocus_bands(input, on_progress, &|piece, progress| {
        run_upscaler(piece, session, factor, progress)
    })
}

type Upscale<'a> = dyn Fn(&Rgb32FImage, &(dyn Fn(f32) + Sync)) -> Result<Rgb32FImage, String> + 'a;

fn superfocus_bands(
    input: &Rgb32FImage,
    on_progress: &(dyn Fn(f32) + Sync),
    upscale: &Upscale,
) -> Result<Rgb32FImage, String> {
    let (w, h) = input.dimensions();
    let mut out = Rgb32FImage::new(w, h);
    let bands = h.div_ceil(SUPERFOCUS_BAND).max(1);
    for band in 0..bands {
        let (y0, y1) = (
            band * SUPERFOCUS_BAND,
            ((band + 1) * SUPERFOCUS_BAND).min(h),
        );
        // the network sees a little either side, so a band's edges are drawn as the middle is
        let (top, bottom) = (
            y0.saturating_sub(SUPERFOCUS_MARGIN),
            (y1 + SUPERFOCUS_MARGIN).min(h),
        );
        let piece = imageops::crop_imm(input, 0, top, w, bottom - top).to_image();
        let done = band as f32;
        let up = upscale(&piece, &|p: f32| on_progress((done + p) / bands as f32))?;
        let down = imageops::resize(&up, w, bottom - top, FilterType::Lanczos3);
        drop(up);
        let row = w as usize * 3;
        let from = &down.as_raw()[(y0 - top) as usize * row..(y1 - top) as usize * row];
        let dst: &mut [f32] = &mut out;
        dst[y0 as usize * row..y1 as usize * row].copy_from_slice(from);
    }
    on_progress(1.0);
    Ok(out)
}

fn run(
    input: &Rgb32FImage,
    session: &Session,
    plan: Plan,
    on_progress: &(dyn Fn(f32) + Sync),
) -> Result<Rgb32FImage, String> {
    if plan.keep_size {
        run_superfocus(input, session, plan.factor, on_progress)
    } else {
        run_upscaler(input, session, plan.factor, on_progress)
    }
}

async fn load_session(app_handle: tauri::AppHandle, factor: u32) -> Result<Session, String> {
    let state = app_handle.state::<AppState>();
    let (ai_state, lock) = (&state.ai_state, &state.ai_init_lock);
    let session = if factor == 4 {
        crate::ai::models::get_or_init_upscale4_model(&app_handle, ai_state, lock).await
    } else {
        crate::ai::models::get_or_init_upscale_model(&app_handle, ai_state, lock).await
    };
    session.map_err(|e| e.to_string())
}

fn waiting_for_denoise(path: &str, denoise: Option<DenoiseMethod>) -> bool {
    denoise.is_some_and(|method| crate::ai::denoise::full_strength(path, method).is_none())
}

pub fn ensure_generated(
    app_handle: &tauri::AppHandle,
    loaded_image: &LoadedImage,
    adjustments: &serde_json::Value,
) {
    if !is_enabled(adjustments) {
        return;
    }
    let path = loaded_image.path.clone();
    let plan = plan(adjustments, loaded_image.image.dimensions());
    let tag = content_tag(adjustments);
    for input in inputs(adjustments) {
        if waiting_for_denoise(&path, input) {
            continue;
        }
        let key = content_key(&path, input, plan, tag);
        if CACHE.is_pending(&key) {
            return;
        }
        let Some(job) = CACHE.job(key, &loaded_image.image, source_string(&path)) else {
            continue;
        };
        let path = path.clone();
        let adjustments = adjustments.clone();
        CACHE.spawn(
            app_handle,
            loaded_image,
            job,
            false,
            load_session(app_handle.clone(), plan.factor),
            move |session, base, progress| {
                run_with_content(
                    base,
                    &path,
                    input,
                    plan,
                    tag,
                    &adjustments,
                    &session,
                    progress,
                )
            },
        );
        return;
    }
}

// The size is not known here, so a picture asked to go four times is taken as done when either factor
// is there: one too big for four is only ever made at two. Superfocus always runs the network asked for.
pub fn has_cached_entry(path: &str, adjustments: &serde_json::Value) -> bool {
    let source = source_string(path);
    let plans: Vec<Plan> = if superfocus(adjustments) {
        vec![plan(adjustments, (0, 0))]
    } else if requested_factor(adjustments) == 4 {
        [4, 2]
            .map(|factor| Plan {
                factor,
                keep_size: false,
            })
            .to_vec()
    } else {
        vec![Plan {
            factor: 2,
            keep_size: false,
        }]
    };
    let tag = content_tag(adjustments);
    plans.into_iter().any(|plan| {
        inputs(adjustments)
            .into_iter()
            .all(|input| CACHE.has_entry(&content_key(path, input, plan, tag), &source))
    })
}

pub fn is_generating(path: &str) -> bool {
    if CONTENT_PENDING.lock().unwrap().iter().any(|p| p == path) {
        return true;
    }
    std::iter::once(None)
        .chain(DenoiseMethod::ALL.into_iter().map(Some))
        .any(|input| {
            [2, 4].iter().any(|&factor| {
                [false, true].iter().any(|&keep_size| {
                    CACHE.is_pending(&key(path, input, Plan { factor, keep_size }))
                })
            })
        })
}

pub fn generate_for_base(
    app_handle: &tauri::AppHandle,
    path: &str,
    base: &DynamicImage,
    adjustments: &serde_json::Value,
) -> Result<bool, String> {
    let plan = plan(adjustments, base.dimensions());
    let tag = content_tag(adjustments);
    for input in inputs(adjustments) {
        if waiting_for_denoise(path, input) {
            return if crate::ai::denoise::is_generating(path) {
                Ok(false)
            } else {
                Err("AI denoise result is not available".to_string())
            };
        }
        let generated = CACHE.generate_blocking(
            app_handle,
            path,
            base,
            content_key(path, input, plan, tag),
            load_session(app_handle.clone(), plan.factor),
            |session, base, progress| {
                run_with_content(
                    base,
                    path,
                    input,
                    plan,
                    tag,
                    adjustments,
                    &session,
                    progress,
                )
            },
        )?;
        if !generated {
            return Ok(false);
        }
    }
    Ok(true)
}

#[tauri::command]
pub fn clear_ai_upscale_cache() -> Result<(), String> {
    CACHE.clear()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn picture(w: u32, h: u32) -> Rgb32FImage {
        Rgb32FImage::from_fn(w, h, |x, y| {
            let v = 0.5 + 0.3 * ((x as f32 * 0.37).sin() * (y as f32 * 0.23).cos());
            image::Rgb([v, v * 0.8, v * 0.6])
        })
    }

    // An "upscaler" that only enlarges, standing in for the network: brought back down, the picture
    // has to come out as it went in, with nothing to see where one band meets the next.
    #[test]
    fn superfocus_bands_meet_without_a_seam() {
        let input = picture(97, 3 * SUPERFOCUS_BAND + 41);
        let enlarge = |piece: &Rgb32FImage, _: &(dyn Fn(f32) + Sync)| {
            Ok(imageops::resize(
                piece,
                piece.width() * 4,
                piece.height() * 4,
                FilterType::Triangle,
            ))
        };
        let out = superfocus_bands(&input, &|_| {}, &enlarge).unwrap();
        assert_eq!(out.dimensions(), input.dimensions());
        let worst = out
            .as_raw()
            .iter()
            .zip(input.as_raw())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 0.04, "the picture changed by {worst}");
        for edge in (1..4).map(|b| b * SUPERFOCUS_BAND) {
            for x in 0..input.width() {
                let (a, b) = (out.get_pixel(x, edge - 1)[0], out.get_pixel(x, edge)[0]);
                let (c, d) = (input.get_pixel(x, edge - 1)[0], input.get_pixel(x, edge)[0]);
                assert!(((a - b) - (c - d)).abs() < 0.04, "seam at row {edge}");
            }
        }
    }

    // A repair changes a small area: redoing only the blocks it touches must give what upscaling the
    // whole repaired picture gives, with an "upscaler" that looks at each pixel's neighbours.
    #[test]
    fn a_repair_redoes_only_its_own_area() {
        let before = picture(500, 330);
        let mut after = before.clone();
        for y in 140..170 {
            for x in 300..340 {
                after.put_pixel(x, y, image::Rgb([0.9, 0.1, 0.2]));
            }
        }
        // each output pixel from a 3x3 neighbourhood, so the border of a piece would show if done wrong
        let enlarge = |piece: &Rgb32FImage, _: &(dyn Fn(f32) + Sync)| {
            let blurred = imageops::blur(piece, 1.0);
            Ok(imageops::resize(
                &blurred,
                piece.width() * 2,
                piece.height() * 2,
                FilterType::Nearest,
            ))
        };
        let whole_before = enlarge(&before, &|_| {}).unwrap();
        let whole_after = enlarge(&after, &|_| {}).unwrap();
        let calls = std::sync::atomic::AtomicUsize::new(0);
        let counted = |piece: &Rgb32FImage, p: &(dyn Fn(f32) + Sync)| {
            calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            enlarge(piece, p)
        };
        let out = refresh_changed(whole_before, &before, &after, &counted, 2, &|_| {}).unwrap();
        let worst = out
            .as_raw()
            .iter()
            .zip(whole_after.as_raw())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(worst < 0.02, "the repaired upscale differs by {worst}");
        // one row of blocks around the repair, not the picture
        assert!(calls.load(std::sync::atomic::Ordering::Relaxed) <= 3);
    }

    #[test]
    fn superfocus_keeps_the_size_and_runs_the_chosen_network() {
        let on = json!({ "superfocusEnabled": true, "superfocusUpscaler": 4 });
        assert!(is_enabled(&on));
        assert_eq!(output_factor(&on), 1);
        let p = plan(&on, (6048, 4032));
        // four times even where a four-times enlargement would not fit on the card
        assert!(p.keep_size && p.factor == 4);
        assert_eq!(upscaled_size(p, (6048, 4032)), (6048, 4032));

        let builtin = json!({ "superfocusEnabled": true, "superfocusUpscaler": 2 });
        assert_eq!(plan(&builtin, (100, 100)).factor, 2);

        // asked to be made larger as well, the picture is made larger: that already has the detail
        let both =
            json!({ "superfocusEnabled": true, "aiUpscaleEnabled": true, "aiUpscaleFactor": 2 });
        assert_eq!(output_factor(&both), 2);
        assert!(!plan(&both, (100, 100)).keep_size);

        // its results are kept apart from the enlarged ones
        let enlarged = Plan {
            factor: 4,
            keep_size: false,
        };
        assert_ne!(key("a.nef", None, p), key("a.nef", None, enlarged));
    }
}
