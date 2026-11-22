use crate::ai::base_cache::{BaseCache, entry_key, source_string};
use crate::ai::denoise::DenoiseMethod;
use crate::app::state::{AppState, LoadedImage};
use crate::render::adjustment_utils::is_effect_enabled;
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

pub fn is_enabled(adjustments: &serde_json::Value) -> bool {
    is_effect_enabled(adjustments, "superResolution", "aiUpscaleEnabled")
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

fn key(path: &str, denoise: Option<DenoiseMethod>, factor: u32) -> String {
    let input = denoise.map_or("none", DenoiseMethod::tag);
    let model = if factor == 4 { MODEL_TAG_4X } else { MODEL_TAG };
    entry_key(&source_string(path), &format!("{model}+{input}"))
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

pub fn apply_ai_upscale<'a>(
    base: &DynamicImage,
    image: Cow<'a, DynamicImage>,
    path: &str,
    adjustments: &serde_json::Value,
) -> Cow<'a, DynamicImage> {
    if !is_enabled(adjustments) {
        return image;
    }
    let (w, h) = base.dimensions();
    if image.dimensions() != (w, h) {
        return image;
    }
    let source = source_string(path);
    let factor = factor(adjustments, (w, h));
    let load = |input| {
        CACHE
            .load(&key(path, input, factor), &source)
            .filter(|entry| (entry.width, entry.height) == (w * factor, h * factor))
    };

    let strength = crate::ai::denoise::blend_strength(adjustments);
    let upscaled = match denoise_input(adjustments).filter(|_| strength.0 > 0.0) {
        None => load(None).map(|entry| entry.to_rgb32f()),
        Some(method) => load(Some(method)).and_then(|denoised| {
            if strength.0 >= 1.0 && strength.1 <= 0.0 {
                return Some(denoised.to_rgb32f());
            }
            let mut out = load(None)?.to_rgb32f();
            crate::ai::denoise::blend_into(&mut out, &denoised.data, strength);
            Some(out)
        }),
    };
    let Some(mut out) = upscaled else {
        return image;
    };
    let amount = detail_amount(adjustments);
    if amount > 0.0 {
        match image.as_rgb32f() {
            Some(source) => enhance_detail(&mut out, source, amount),
            None => enhance_detail(&mut out, &image.to_rgb32f(), amount),
        }
    }
    Cow::Owned(DynamicImage::ImageRgb32F(out))
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
    let factor = factor(adjustments, loaded_image.image.dimensions());
    for input in inputs(adjustments) {
        if waiting_for_denoise(&path, input) {
            continue;
        }
        let key = key(&path, input, factor);
        if CACHE.is_pending(&key) {
            return;
        }
        let Some(job) = CACHE.job(key, &loaded_image.image, source_string(&path)) else {
            continue;
        };
        let path = path.clone();
        CACHE.spawn(
            app_handle,
            loaded_image,
            job,
            false,
            load_session(app_handle.clone(), factor),
            move |session, base, progress| {
                run_upscaler(&input_for(base, &path, input)?, &session, factor, progress)
            },
        );
        return;
    }
}

// The size is not known here, so a picture asked to go four times is taken as done when either factor
// is there: one too big for four is only ever made at two.
pub fn has_cached_entry(path: &str, adjustments: &serde_json::Value) -> bool {
    let source = source_string(path);
    let factors: &[u32] = if requested_factor(adjustments) == 4 {
        &[4, 2]
    } else {
        &[2]
    };
    factors.iter().any(|&factor| {
        inputs(adjustments)
            .into_iter()
            .all(|input| CACHE.has_entry(&key(path, input, factor), &source))
    })
}

pub fn is_generating(path: &str) -> bool {
    std::iter::once(None)
        .chain(DenoiseMethod::ALL.into_iter().map(Some))
        .any(|input| {
            [2, 4]
                .iter()
                .any(|&f| CACHE.is_pending(&key(path, input, f)))
        })
}

pub fn generate_for_base(
    app_handle: &tauri::AppHandle,
    path: &str,
    base: &DynamicImage,
    adjustments: &serde_json::Value,
) -> Result<bool, String> {
    let factor = factor(adjustments, base.dimensions());
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
            key(path, input, factor),
            load_session(app_handle.clone(), factor),
            |session, base, progress| {
                run_upscaler(&input_for(base, path, input)?, &session, factor, progress)
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
