//! Focus map: how blurred each edge of the photo is, measured in pixels of the full-size photo, once per
//! photo and geometry. Focus peaking draws from it instead of guessing from the screen.
//!
//! Peaking that reads the picture on screen judges contrast, not focus: a bright edge out of focus beats
//! a faint one in focus, noise looks sharp, the photo scaled down to the screen looks sharper than it is,
//! and the user's own sharpening and clarity count as focus. This reads the photo itself, before any edit,
//! at full size, and measures the blur of each edge directly:
//!
//! - **Brightness** is taken as the square root of linear light for raw files, which makes the noise about
//!   as strong in the shadows as in the highlights (shot noise grows with the square root of the light).
//! - **Re-blur ratio** (Zhuo & Sim, "Defocus map estimation from a single image", 2011): an edge blurred by
//!   `s` has a gradient peak proportional to `1/s`. Blurring the photo again by a known `s0` makes it
//!   `1/sqrt(s^2 + s0^2)`, so the ratio of the two peaks gives `s` back, whatever the edge's contrast.
//! - **Noise** is measured on the photo: the gradient level of its flattest areas. An edge counts once it
//!   stands well above it, so every ISO gets the same treatment without a fixed threshold.
//!
//! The result is kept as a pyramid in which every coarser level holds the sharpest of the 2x2 below it, so
//! the display reads the level that matches its zoom and loses no edge to the scaling.

use std::borrow::Cow;

use image::DynamicImage;
use rayon::prelude::*;

/// Blur of an edge as stored per texel: `round(blur * BLUR_SCALE)`, up to 254 (just under 8 pixels).
pub const BLUR_SCALE: f32 = 32.0;
/// Stored where there is no edge.
pub const NO_EDGE: u8 = 255;

/// The light smoothing every gradient is read through, against noise, in pixels; taken back out of the
/// estimate at the end.
const PRE_SIGMA: f32 = 0.6;
/// The known blur added to read how much an edge already has, in pixels. Large enough that an edge of a
/// few pixels still changes measurably (3 pixels: by a quarter in squared ratio), small enough not to
/// merge neighbouring edges.
const REBLUR_SIGMA: f32 = 1.5;
/// The gradient powers are summed around each edge with a Gaussian of this size: wide enough to take in
/// the whole width of an edge of a few pixels, re-blurred.
const POOL_SIGMA: f32 = 3.0;
/// Reading a gradient as the difference of the two neighbours blurs it a little more again: measured on
/// synthetic edges of 0.5 to 3 pixels, a constant 0.4 square pixels, taken out like the smoothing.
const DIFFERENCE_VARIANCE: f32 = 0.4;
/// An edge counts once its gradient stands this many times above the gradient of the flattest areas.
const EDGE_OVER_NOISE: f32 = 5.0;
/// ... and above this in any case (in square-root brightness), for photos with hardly any noise.
const EDGE_FLOOR: f32 = 0.004;
/// The noise is read on blocks of this size ...
const NOISE_BLOCK: usize = 32;
/// ... as the median gradient of the block at this rank, from the flattest up.
const NOISE_RANK: f32 = 0.1;

pub struct FocusLevel {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct FocusMap {
    /// Level 0 is the full photo; each next one is half the size and holds the sharpest of the 2x2 below.
    /// Sizes follow the GPU's mip rule, `max(1, size >> level)`.
    pub levels: Vec<FocusLevel>,
    /// Gradient level of the flattest areas, in square-root brightness per pixel.
    pub noise: f32,
    /// How many edge texels were measured.
    pub edges: usize,
}

pub fn build(image: &DynamicImage, is_raw: bool) -> FocusMap {
    let (width, height, plane) = brightness(image, is_raw);
    let (data, noise, edges) = blur_map(plane, width as usize, height as usize);
    let mut levels = vec![FocusLevel {
        width,
        height,
        data,
    }];
    while let Some(last) = levels.last()
        && (last.width > 1 || last.height > 1)
    {
        let next = halve(last);
        levels.push(next);
    }
    FocusMap {
        levels,
        noise,
        edges,
    }
}

fn brightness(image: &DynamicImage, is_raw: bool) -> (u32, u32, Vec<f32>) {
    let rgb = match image.as_rgb32f() {
        Some(buffer) => Cow::Borrowed(buffer),
        None => Cow::Owned(image.to_rgb32f()),
    };
    let (width, height) = rgb.dimensions();
    let plane = rgb
        .as_raw()
        .par_chunks_exact(3)
        .map(|p| {
            let light = (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).max(0.0);
            // other files are already gamma-encoded, which does much the same
            if is_raw { light.sqrt() } else { light }
        })
        .collect();
    (width, height, plane)
}

fn kernel(sigma: f32) -> Vec<f32> {
    let radius = (3.0 * sigma).ceil() as i32;
    let mut weights: Vec<f32> = (-radius..=radius)
        .map(|x| (-(x * x) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= total);
    weights
}

fn gaussian(src: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
    let weights = kernel(sigma);
    let radius = (weights.len() / 2) as isize;
    let last_x = width as isize - 1;
    let last_y = height as isize - 1;

    let mut across = vec![0.0f32; width * height];
    across
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let line = &src[y * width..(y + 1) * width];
            for (x, out) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for (i, w) in weights.iter().enumerate() {
                    let sx = (x as isize + i as isize - radius).clamp(0, last_x) as usize;
                    sum += w * line[sx];
                }
                *out = sum;
            }
        });

    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        for (i, w) in weights.iter().enumerate() {
            let sy = (y as isize + i as isize - radius).clamp(0, last_y) as usize;
            let line = &across[sy * width..(sy + 1) * width];
            for (o, v) in row.iter_mut().zip(line) {
                *o += w * v;
            }
        }
    });
    out
}

fn gradient_magnitude(plane: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; width * height];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        if y == 0 || y + 1 == height {
            return;
        }
        for x in 1..width.saturating_sub(1) {
            let i = y * width + x;
            let gx = 0.5 * (plane[i + 1] - plane[i - 1]);
            let gy = 0.5 * (plane[i + width] - plane[i - width]);
            row[x] = (gx * gx + gy * gy).sqrt();
        }
    });
    out
}

/// The gradient level of the flattest areas: the median gradient of each block, at a low rank among blocks.
fn noise_level(magnitude: &[f32], width: usize, height: usize) -> f32 {
    let blocks_x = width / NOISE_BLOCK;
    let blocks_y = height / NOISE_BLOCK;
    if blocks_x == 0 || blocks_y == 0 {
        let mut all = magnitude.to_vec();
        let mid = all.len() / 2;
        return *all.select_nth_unstable_by(mid, f32::total_cmp).1;
    }
    let mut medians: Vec<f32> = (0..blocks_x * blocks_y)
        .into_par_iter()
        .map(|b| {
            let (bx, by) = (b % blocks_x, b / blocks_x);
            let mut values = Vec::with_capacity(NOISE_BLOCK * NOISE_BLOCK);
            for y in by * NOISE_BLOCK..(by + 1) * NOISE_BLOCK {
                let start = y * width + bx * NOISE_BLOCK;
                values.extend_from_slice(&magnitude[start..start + NOISE_BLOCK]);
            }
            let mid = values.len() / 2;
            *values.select_nth_unstable_by(mid, f32::total_cmp).1
        })
        .collect();
    let rank = ((medians.len() as f32 * NOISE_RANK) as usize).min(medians.len() - 1);
    *medians.select_nth_unstable_by(rank, f32::total_cmp).1
}

/// The blur of every edge texel of a brightness plane, stored as `round(blur * BLUR_SCALE)` or `NO_EDGE`,
/// with the measured gradient noise and the number of edge texels.
fn blur_map(plane: Vec<f32>, width: usize, height: usize) -> (Vec<u8>, f32, usize) {
    if width < 3 || height < 3 {
        return (vec![NO_EDGE; width * height], 0.0, 0);
    }
    let fine = gaussian(&plane, width, height, PRE_SIGMA);
    let coarse = gaussian(
        &plane,
        width,
        height,
        (PRE_SIGMA * PRE_SIGMA + REBLUR_SIGMA * REBLUR_SIGMA).sqrt(),
    );
    drop(plane);
    let magnitude = gradient_magnitude(&fine, width, height);
    let noise = noise_level(&magnitude, width, height);
    // Noise adds to every gradient, more to the finer one, which would make a blurred edge in noise read
    // sharp. Its power is taken out of both before their ratio is read: in flat areas the gradient
    // magnitude follows a Rayleigh law, whose median is 1.177 sigma and mean square 2 sigma^2.
    let noise_power = |median: f32| 2.0 * (median / 1.177).powi(2);
    let fine_noise = noise_power(noise);
    let coarse_magnitude = gradient_magnitude(&coarse, width, height);
    let coarse_noise = noise_power(noise_level(&coarse_magnitude, width, height));
    drop(coarse);
    let threshold = (EDGE_OVER_NOISE * noise).max(EDGE_FLOOR);

    // Every texel's gradient power less the noise in it, summed over the neighbourhood. Across an edge
    // blurred by s the squared gradient sums to a constant over s, so the ratio of the fine sum to the
    // coarse one is s_coarse / s_fine, and the noise, taken out texel by texel, sums to nothing. Nothing
    // here depends on which texel is picked as the crest below, so the noise that decides that pick
    // cannot slant the measure.
    let fine_pooled = gaussian(
        &magnitude
            .par_iter()
            .map(|m| m * m - fine_noise)
            .collect::<Vec<f32>>(),
        width,
        height,
        POOL_SIGMA,
    );
    let coarse_pooled = gaussian(
        &coarse_magnitude
            .par_iter()
            .map(|m| m * m - coarse_noise)
            .collect::<Vec<f32>>(),
        width,
        height,
        POOL_SIGMA,
    );
    drop(coarse_magnitude);
    let ratios = pooled_ratios();

    let mut out = vec![NO_EDGE; width * height];
    let edges: usize = out
        .par_chunks_mut(width)
        .enumerate()
        .map(|(y, row)| {
            if y == 0 || y + 1 == height {
                return 0;
            }
            let mut count = 0;
            for x in 1..width - 1 {
                let i = y * width + x;
                let m = magnitude[i];
                if m < threshold {
                    continue;
                }
                // the crest of an edge, where the gradient peaks across it, is where its blur is written
                let gx = fine[i + 1] - fine[i - 1];
                let gy = fine[i + width] - fine[i - width];
                let (before, after) = if gx.abs() > 2.414 * gy.abs() {
                    (magnitude[i - 1], magnitude[i + 1])
                } else if gy.abs() > 2.414 * gx.abs() {
                    (magnitude[i - width], magnitude[i + width])
                } else if gx * gy > 0.0 {
                    (magnitude[i - width - 1], magnitude[i + width + 1])
                } else {
                    (magnitude[i - width + 1], magnitude[i + width - 1])
                };
                if m < before || m <= after {
                    continue;
                }
                row[x] = if coarse_pooled[i] <= 0.0 || fine_pooled[i] <= 0.0 {
                    254
                } else {
                    stored_blur(&ratios, fine_pooled[i] / coarse_pooled[i])
                };
                count += 1;
            }
            count
        })
        .sum();
    (out, noise, edges)
}

/// The ratio of the pooled fine to coarse gradient power that an edge of each stored blur gives, falling as
/// the blur grows. Across an edge blurred by `s` (its own blur with the smoothing and the difference), the
/// squared gradient is `c^2 / (2 pi s^2) * exp(-x^2 / s^2)`; summed under the pooling Gaussian it comes to
/// `c^2 / (2 sqrt(pi) s) / sqrt(1 + s^2 / (2 pool^2))`. The contrast `c` drops out of the ratio.
fn pooled_ratios() -> Vec<f32> {
    let pooled = |seen_squared: f32| {
        1.0 / (seen_squared.sqrt() * (1.0 + seen_squared / (2.0 * POOL_SIGMA * POOL_SIGMA)).sqrt())
    };
    (0..=254)
        .map(|code| {
            let blur = code as f32 / BLUR_SCALE;
            let fine = blur * blur + PRE_SIGMA * PRE_SIGMA + DIFFERENCE_VARIANCE;
            pooled(fine) / pooled(fine + REBLUR_SIGMA * REBLUR_SIGMA)
        })
        .collect()
}

/// The stored blur whose ratio is nearest above the one measured: the edge is at least that blurred.
fn stored_blur(ratios: &[f32], ratio: f32) -> u8 {
    ratios.partition_point(|&r| r > ratio).min(254) as u8
}

/// The next level: half the size (the GPU's `max(1, size / 2)`), each texel the sharpest of those below it.
fn halve(level: &FocusLevel) -> FocusLevel {
    let (w, h) = (level.width as usize, level.height as usize);
    let (w2, h2) = ((w / 2).max(1), (h / 2).max(1));
    let span = |i: usize, half: usize, full: usize| {
        let start = (2 * i).min(full - 1);
        // the last texel also takes the odd row or column left over
        let end = if i + 1 == half {
            full
        } else {
            (2 * i + 2).min(full)
        };
        start..end
    };
    let mut data = vec![NO_EDGE; w2 * h2];
    data.par_chunks_mut(w2).enumerate().for_each(|(y2, row)| {
        for (x2, out) in row.iter_mut().enumerate() {
            let mut sharpest = NO_EDGE;
            for y in span(y2, h2, h) {
                for x in span(x2, w2, w) {
                    sharpest = sharpest.min(level.data[y * w + x]);
                }
            }
            *out = sharpest;
        }
    });
    FocusLevel {
        width: w2 as u32,
        height: h2 as u32,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Abramowitz & Stegun 7.1.26, good to 1.5e-7
    fn erf(x: f32) -> f32 {
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let y = 1.0
            - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                + 0.254829592)
                * t
                * (-x * x).exp();
        if x < 0.0 { -y } else { y }
    }

    // a repeatable normal noise source
    struct Noise(u64);
    impl Noise {
        fn uniform(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 + 0.5) / (1u64 << 24) as f32
        }
        fn normal(&mut self) -> f32 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos()
        }
    }

    const W: usize = 128;
    const H: usize = 96;

    // a vertical edge of the given contrast, blurred by `blur` pixels, with noise of `sigma`
    fn edge(blur: f32, contrast: f32, sigma: f32, seed: u64) -> Vec<f32> {
        let mut noise = Noise(seed);
        let mut plane = Vec::with_capacity(W * H);
        for _y in 0..H {
            for x in 0..W {
                let d = x as f32 - W as f32 / 2.0 + 0.3;
                let step = 0.5 * (1.0 + erf(d / (blur * std::f32::consts::SQRT_2)));
                plane.push(0.3 + contrast * step + sigma * noise.normal());
            }
        }
        plane
    }

    // the blurs measured on the edge's crest in the middle rows, sorted
    fn measured(map: &[u8]) -> Vec<f32> {
        let mut found: Vec<f32> = (10..H - 10)
            .filter_map(|y| {
                (W / 2 - 4..W / 2 + 4)
                    .map(|x| map[y * W + x])
                    .filter(|&v| v != NO_EDGE)
                    .min()
            })
            .map(|v| v as f32 / BLUR_SCALE)
            .collect();
        found.sort_by(f32::total_cmp);
        found
    }

    #[test]
    fn an_edge_reads_as_blurred_as_it_is_whatever_its_contrast() {
        for &blur in &[0.7f32, 1.0, 2.0, 3.0] {
            for &contrast in &[0.05f32, 0.4] {
                let (map, _, _) = blur_map(edge(blur, contrast, 0.0, 1), W, H);
                let found = measured(&map);
                assert!(
                    found.len() > (H - 20) * 9 / 10,
                    "edge of blur {blur} not found"
                );
                let median = found[found.len() / 2];
                assert!(
                    (median - blur).abs() <= 0.2 + 0.1 * blur,
                    "blur {blur} at contrast {contrast} read as {median}"
                );
            }
        }
    }

    #[test]
    fn noise_alone_has_no_edges() {
        let mut noise = Noise(7);
        let plane: Vec<f32> = (0..W * H).map(|_| 0.3 + 0.02 * noise.normal()).collect();
        let (_, _, edges) = blur_map(plane, W, H);
        assert!(
            edges <= W * H / 1000,
            "{edges} texels of noise taken for edges"
        );
    }

    #[test]
    fn an_edge_in_noise_is_still_found_and_measured() {
        let (map, _, _) = blur_map(edge(1.0, 0.2, 0.005, 3), W, H);
        let found = measured(&map);
        assert!(found.len() > (H - 20) * 8 / 10, "edge lost in the noise");
        let median = found[found.len() / 2];
        assert!(
            (median - 1.0).abs() <= 0.4,
            "blur 1 in noise read as {median}"
        );
    }

    // noise raises the finer gradient more than the coarser one; left in, it makes a blurred edge read
    // sharp, the wrong way round for peaking
    #[test]
    fn a_blurred_edge_in_noise_does_not_read_sharp() {
        let (map, _, _) = blur_map(edge(2.0, 0.1, 0.01, 9), W, H);
        let found = measured(&map);
        if !found.is_empty() {
            let median = found[found.len() / 2];
            assert!(median >= 1.5, "blur 2 in noise read as {median}");
        }
    }

    #[test]
    fn each_level_keeps_the_sharpest_below_it_and_follows_the_gpu_sizes() {
        let mut data = vec![NO_EDGE; 5 * 3];
        data[2 * 5 + 4] = 40; // the odd last column and row
        data[0] = 90;
        data[1] = 30;
        let level = FocusLevel {
            width: 5,
            height: 3,
            data,
        };
        let next = halve(&level);
        assert_eq!((next.width, next.height), (2, 1));
        assert_eq!(next.data, vec![30, 40]);
    }
}
