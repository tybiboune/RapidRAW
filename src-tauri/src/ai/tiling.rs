use crate::ai::models::{GpuOutOfMemory, gpu_inference, make_room_on_gpu};
use anyhow::Result;
use image::{DynamicImage, Rgb32FImage};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Clone, Copy)]
struct TileParams {
    cs: usize,
    margin: usize,
    overlap: usize,
}

const DENOISE_TILE: TileParams = TileParams {
    cs: 504,
    margin: 12,
    overlap: 64,
};

pub const UPSCALE_TILE_SIZE: usize = 544;

const UPSCALE_TILE: TileParams = TileParams {
    cs: UPSCALE_TILE_SIZE,
    margin: 16,
    overlap: 16,
};

/// The four-times model takes a square of exactly this size.
pub const UPSCALE4_TILE_SIZE: usize = 256;

const UPSCALE4_TILE: TileParams = TileParams {
    cs: UPSCALE4_TILE_SIZE,
    margin: 16,
    overlap: 16,
};

#[inline]
fn mirror_coord(c: i32, size: i32) -> i32 {
    if c < 0 {
        (-c).min(size - 1)
    } else if c >= size {
        (2 * size - 1 - c).max(0)
    } else {
        c
    }
}

fn extract_tile_mirror(img: &Rgb32FImage, x0: i32, y0: i32, cs: usize) -> Array4<f32> {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let src = img.as_raw();
    let plane = cs * cs;
    let mut data = vec![0.0f32; 3 * plane];
    let (r, rest) = data.split_at_mut(plane);
    let (g, b) = rest.split_at_mut(plane);
    r.par_chunks_exact_mut(cs)
        .zip(g.par_chunks_exact_mut(cs))
        .zip(b.par_chunks_exact_mut(cs))
        .enumerate()
        .for_each(|(dy, ((r, g), b))| {
            let sy = mirror_coord(y0 + dy as i32, h) as usize;
            let row = &src[sy * w as usize * 3..(sy + 1) * w as usize * 3];
            for dx in 0..cs {
                let sx = mirror_coord(x0 + dx as i32, w) as usize * 3;
                r[dx] = row[sx];
                g[dx] = row[sx + 1];
                b[dx] = row[sx + 2];
            }
        });
    Array4::from_shape_vec((1, 3, cs, cs), data).expect("tile buffer matches its shape")
}

fn tile_origins(len: usize, params: TileParams) -> Vec<i32> {
    let valid = params.cs - 2 * params.margin;
    let step = (valid - params.overlap).max(1);
    let mut origins = Vec::new();
    let mut start = 0;
    loop {
        origins.push(start as i32 - params.margin as i32);
        if start + valid >= len {
            break;
        }
        start += step;
    }
    origins
}

fn tile_blend_weights(params: TileParams) -> Vec<f32> {
    let valid = params.cs - 2 * params.margin;
    let ramp = |d: usize| ((d as f32 + 0.5) / params.overlap.max(1) as f32).min(1.0);
    (0..valid)
        .map(|i| {
            let t = ramp(i).min(ramp(valid - 1 - i));
            t * t * (3.0 - 2.0 * t)
        })
        .collect()
}

/// Runs a network on one tile (1Ã—3Ã—csÃ—cs, floats) and returns its picture of the same or scaled size.
type Infer<'a> = &'a (dyn Fn(Array4<f32>) -> Result<Array4<f32>> + Sync);

fn run_plain(session: &Mutex<Session>, tile: Array4<f32>) -> Result<Array4<f32>> {
    let t_input = Tensor::from_array(tile)?;
    let mut sess = session.lock().unwrap();
    let outputs = gpu_inference(sess.run(ort::inputs![t_input]))?;
    outputs[0]
        .try_extract_array::<f32>()?
        .as_standard_layout()
        .into_owned()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))
}

fn out_of_memory(result: &Result<Array4<f32>>) -> bool {
    result
        .as_ref()
        .is_err_and(|e| e.downcast_ref::<GpuOutOfMemory>().is_some())
}

/// One tile through the network. When the GPU has run out of video memory, the cached sessions are
/// dropped: DirectML cannot go on with a session that ran out, so the next attempt builds a fresh one,
/// with the memory the others held free for it.
fn infer_within_memory(infer: Infer, tile: Array4<f32>) -> Result<Array4<f32>> {
    let result = infer(tile);
    if out_of_memory(&result) {
        make_room_on_gpu();
    }
    result
}

fn run_tiled(
    img: &Rgb32FImage,
    infer: Infer,
    params: TileParams,
    scale: u32,
    on_progress: &dyn Fn(f32),
) -> Result<Rgb32FImage> {
    let s = scale as usize;
    let (width, height) = (img.width() as usize, img.height() as usize);
    let (out_w, out_h) = (width * s, height * s);
    let valid = params.cs - 2 * params.margin;
    let weights = tile_blend_weights(TileParams {
        cs: params.cs * s,
        margin: params.margin * s,
        overlap: params.overlap * s,
    });
    let xs = tile_origins(width, params);
    let ys = tile_origins(height, params);
    let total = xs.len() * ys.len();

    let mut accumulator = vec![0.0f32; out_w * out_h * 3];
    let mut weight_sum = vec![0.0f32; out_w * out_h];
    let origins: Vec<(i32, i32)> = ys
        .iter()
        .flat_map(|&y| xs.iter().map(move |&x| (y, x)))
        .collect();

    std::thread::scope(|scope| -> Result<()> {
        let (sender, receiver) = std::sync::mpsc::sync_channel::<Result<Array4<f32>>>(1);
        let origins_ref = &origins;
        scope.spawn(move || {
            for &(y0, x0) in origins_ref {
                let tile = infer_within_memory(infer, extract_tile_mirror(img, x0, y0, params.cs));
                let failed = tile.is_err();
                if sender.send(tile).is_err() || failed {
                    break;
                }
            }
        });

        for (i, (&(y0, x0), tile)) in origins.iter().zip(receiver).enumerate() {
            on_progress(i as f32 / total as f32);
            let tile = tile?;
            let (tile_h, tile_w) = (tile.dim().2, tile.dim().3);
            if tile_h < params.cs * s || tile_w < params.cs * s {
                return Err(anyhow::anyhow!(
                    "Unexpected output tile size {:?} for a {} px input",
                    tile.dim(),
                    params.cs
                ));
            }
            let data = tile
                .as_slice()
                .ok_or_else(|| anyhow::anyhow!("Output tile is not contiguous"))?;
            let plane = tile_h * tile_w;

            let (ox, oy, om) = (x0 * scale as i32, y0 * scale as i32, params.margin * s);
            let top = oy + om as i32;
            let first = top.max(0) as usize;
            let last = (top + (valid * s) as i32).clamp(0, out_h as i32) as usize;
            if first >= last {
                continue;
            }
            accumulator[first * out_w * 3..last * out_w * 3]
                .par_chunks_exact_mut(out_w * 3)
                .zip(weight_sum[first * out_w..last * out_w].par_chunks_exact_mut(out_w))
                .enumerate()
                .for_each(|(row, (acc, wsum))| {
                    let ty = (first + row) as i32 - top;
                    let ty = ty as usize;
                    let line = (om + ty) * tile_w + om;
                    for tx in 0..valid * s {
                        let gx = ox + (om + tx) as i32;
                        if gx < 0 || gx >= out_w as i32 {
                            continue;
                        }
                        let w = weights[ty] * weights[tx];
                        let gx = gx as usize;
                        for c in 0..3 {
                            let v = data[c * plane + line + tx].clamp(0.0, 1.0);
                            acc[gx * 3 + c] += v * w;
                        }
                        wsum[gx] += w;
                    }
                });
        }
        Ok(())
    })?;

    accumulator
        .par_chunks_exact_mut(3)
        .zip(weight_sum.par_iter())
        .for_each(|(px, &w)| {
            let inv = if w > 0.0 { 1.0 / w } else { 0.0 };
            for v in px {
                *v = (*v * inv).clamp(0.0, 1.0);
            }
        });
    Rgb32FImage::from_raw(out_w as u32, out_h as u32, accumulator)
        .ok_or_else(|| anyhow::anyhow!("Tiled output does not match its dimensions"))
}

pub fn run_ai_denoise(
    rgb_img: &Rgb32FImage,
    session: &Mutex<Session>,
    on_progress: &dyn Fn(f32),
) -> Result<DynamicImage> {
    let infer = |tile| run_plain(session, tile);
    let out = run_tiled(rgb_img, &infer, DENOISE_TILE, 1, on_progress)?;
    Ok(DynamicImage::ImageRgb32F(out))
}

pub fn run_ai_upscale(
    rgb_img: &Rgb32FImage,
    session: &Mutex<Session>,
    scale: u32,
    on_progress: &dyn Fn(f32),
) -> Result<Rgb32FImage> {
    let infer = |tile| run_plain(session, tile);
    let tile = if scale == 4 {
        UPSCALE4_TILE
    } else {
        UPSCALE_TILE
    };
    run_tiled(rgb_img, &infer, tile, scale, on_progress)
}

/// The deblurring networks look at a whole tile at once (NAFNet's channel attention pools over it), so
/// a tile about the size they were tested at keeps them doing what they learned.
const DEBLUR_TILE: TileParams = TileParams {
    cs: 384,
    margin: 32,
    overlap: 32,
};

/// The tile sizes a deblurring network can be run at, largest first. Both take any multiple of 16.
pub const DEBLUR_TILE_SIZES: [usize; 5] = [384, 320, 256, 192, 128];

/// The tile sizes worth trying for a network that needs about `bytes_per_pixel` of video memory per
/// pixel of a tile, when `free` is what the GPU has left: the largest that fits first, and at most three,
/// each smaller than the last, for when one runs out after all.
pub fn deblur_tiles_for(free: Option<u64>, bytes_per_pixel: u64) -> Vec<usize> {
    let fits =
        |cs: usize| free.is_none_or(|free| (cs * cs) as u64 * bytes_per_pixel * 5 / 4 < free);
    let first = DEBLUR_TILE_SIZES
        .iter()
        .position(|&cs| fits(cs))
        .unwrap_or(DEBLUR_TILE_SIZES.len() - 1);
    DEBLUR_TILE_SIZES[first..].iter().copied().take(3).collect()
}

pub fn run_ai_deblur(
    rgb_img: &Rgb32FImage,
    session: &Mutex<Session>,
    tile: usize,
    on_progress: &dyn Fn(f32),
) -> Result<Rgb32FImage> {
    let infer = |tile| run_plain(session, tile);
    let params = TileParams {
        cs: tile,
        ..DEBLUR_TILE
    };
    run_tiled(rgb_img, &infer, params, 1, on_progress)
}

pub const RAWFORGE_TILE_SIZE: usize = 256;

const RAWFORGE_TILE: TileParams = TileParams {
    cs: RAWFORGE_TILE_SIZE,
    margin: 8,
    overlap: 48,
};

/// RawForge's networks were trained on linear Rec. 2020 pictures, while the develop pipeline works in
/// linear sRGB: the picture crosses over for the network and comes back after.
const SRGB_TO_REC2020: [[f32; 3]; 3] = [
    [0.627_403_9, 0.329_283, 0.043_313_1],
    [0.069_097_3, 0.919_540_4, 0.011_362_3],
    [0.016_391_4, 0.088_013_3, 0.895_595_3],
];

const REC2020_TO_SRGB: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5, 1.132_899_9, -0.008_349_4],
    [-0.018_150_8, -0.100_578_9, 1.118_729_7],
];

/// Applies a 3Ã—3 colour matrix to a 1Ã—3Ã—hÃ—w tile, optionally keeping the result inside 0..1.
fn convert_tile(tile: &mut Array4<f32>, matrix: &[[f32; 3]; 3], clamp: bool) {
    let (h, w) = (tile.dim().2, tile.dim().3);
    for y in 0..h {
        for x in 0..w {
            let px = [tile[[0, 0, y, x]], tile[[0, 1, y, x]], tile[[0, 2, y, x]]];
            for (c, row) in matrix.iter().enumerate() {
                let v = row[0] * px[0] + row[1] * px[1] + row[2] * px[2];
                tile[[0, c, y, x]] = if clamp { v.clamp(0.0, 1.0) } else { v };
            }
        }
    }
}

/// The ISO is passed to the network as ISO / 6400, in half precision, as RawForge does.
pub fn run_rawforge_denoise(
    rgb_img: &Rgb32FImage,
    session: &Mutex<Session>,
    iso: u32,
    on_progress: &dyn Fn(f32),
) -> Result<DynamicImage> {
    use half::f16;

    let cond = ndarray::Array2::from_elem((1, 1), f16::from_f32(iso as f32 / 6400.0));
    let infer = |mut tile: Array4<f32>| -> Result<Array4<f32>> {
        convert_tile(&mut tile, &SRGB_TO_REC2020, true);
        let input = Tensor::from_array(tile.mapv(f16::from_f32))?;
        let cond = Tensor::from_array(cond.clone())?;
        let mut sess = session.lock().unwrap();
        let outputs = gpu_inference(sess.run(ort::inputs!["input" => input, "cond" => cond]))?;
        let mut out = outputs[0]
            .try_extract_array::<f16>()?
            .mapv(|v| v.to_f32())
            .as_standard_layout()
            .into_owned()
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| anyhow::anyhow!("Unexpected output shape: {}", e))?;
        convert_tile(&mut out, &REC2020_TO_SRGB, false);
        Ok(out)
    };
    let out = run_tiled(rgb_img, &infer, RAWFORGE_TILE, 1, on_progress)?;
    Ok(DynamicImage::ImageRgb32F(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srgb_to_linear(v: f32) -> f32 {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    }

    fn linear_to_srgb(v: f32) -> f32 {
        let v = v.clamp(0.0, 1.0);
        if v <= 0.003_130_8 {
            v * 12.92
        } else {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        }
    }

    /// Runs RawForge on a PNG (RRAW_PHOTO, sRGB) with the model RRAW_MODEL at the ISO RRAW_ISO and writes
    /// the result to RRAW_OUT. Needs the ONNX runtime library (ORT_DYLIB_PATH).
    #[test]
    #[ignore]
    fn rawforge_on_a_picture() {
        let photo = std::env::var("RRAW_PHOTO").expect("RRAW_PHOTO");
        let out = std::env::var("RRAW_OUT").expect("RRAW_OUT");
        let model = std::env::var("RRAW_MODEL").expect("RRAW_MODEL");
        let iso: u32 = std::env::var("RRAW_ISO")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6400);

        let mut img = image::open(&photo).expect("open picture").to_rgb32f();
        img.iter_mut().for_each(|v| *v = srgb_to_linear(*v));

        let _ = ort::init().commit();
        let session = Mutex::new(
            Session::builder()
                .unwrap()
                .commit_from_file(&model)
                .expect("load model"),
        );
        let start = std::time::Instant::now();
        let result = run_rawforge_denoise(&img, &session, iso, &|_| {})
            .expect("denoise")
            .into_rgb32f();
        println!("rawforge took {:.1}s", start.elapsed().as_secs_f32());

        let bytes: Vec<u8> = result
            .iter()
            .map(|v| (linear_to_srgb(*v) * 255.0 + 0.5) as u8)
            .collect();
        image::RgbImage::from_raw(result.width(), result.height(), bytes)
            .unwrap()
            .save(&out)
            .unwrap();
    }

    // A deblurring network is run at the largest tile the free video memory allows, with smaller ones to
    // fall back on, and gives the picture back whole at any of them.
    #[test]
    fn deblur_tiles_follow_the_free_video_memory() {
        // Restormer needs about 22 KB a pixel: with 3 GB free, 384 px does not fit and 320 px does
        assert_eq!(
            deblur_tiles_for(Some(3_000_000_000), 22_000),
            vec![320, 256, 192]
        );
        assert_eq!(deblur_tiles_for(None, 22_000), vec![384, 320, 256]);
        // nothing fits: the smallest is still tried
        assert_eq!(deblur_tiles_for(Some(1), 22_000), vec![128]);

        let img = Rgb32FImage::from_fn(700, 450, |x, y| {
            let v = 0.5 + 0.4 * ((x as f32 * 0.05).sin() * (y as f32 * 0.07).cos());
            image::Rgb([v, 1.0 - v, v * 0.5])
        });
        let same = |tile: Array4<f32>| -> Result<Array4<f32>> { Ok(tile) };
        for cs in DEBLUR_TILE_SIZES {
            let params = TileParams { cs, ..DEBLUR_TILE };
            let out = run_tiled(&img, &same, params, 1, &|_| {}).unwrap();
            let worst = out
                .as_raw()
                .iter()
                .zip(img.as_raw())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            assert!(worst < 1e-4, "{cs} px tiles changed the picture by {worst}");
        }
    }
}
