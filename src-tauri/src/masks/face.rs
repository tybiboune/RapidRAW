use std::f32::consts::FRAC_PI_2;
use std::sync::Mutex;

use anyhow::Result;
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, RgbImage, imageops};
use ndarray::Array4;
use ort::session::Session;
use ort::value::Tensor;
use rayon::prelude::*;

use crate::ai::guided_filter::fast_color_guided_filter;

const LANDMARK_INPUT_SIZE: usize = 256;
const LANDMARK_COUNT: usize = 478;
const LANDMARK_MARGIN: f32 = 1.5;
const LANDMARK_PASSES: usize = 4;
const LANDMARK_CONVERGED_SCORE: f32 = 12.0;
const FACE_PRESENCE_THRESHOLD: f32 = 6.0;
const DUPLICATE_BOX_OVERLAP: f32 = 0.5;
const DUPLICATE_FACE_DISTANCE: f32 = 0.3;
const SEGMENTER_INPUT_SIZE: usize = 256;
const SEGMENTER_CLASSES: usize = 6;
const SEGMENTER_SPAN: f32 = 2.6;
const SEGMENTER_OFFSET: f32 = 0.2;
const SEGMENTER_OUTPUTS: [usize; 3] = [1, 3, 2];
const BODY_SPAN: f32 = 6.0;
const BODY_OFFSET: f32 = 2.0;
const BODY_WORK_SIZE: f32 = 1024.0;
const BODY_REFINE_DIVISOR: f32 = 8.0;
const BODY_REFINE_RADIUS: [f32; 2] = [2.0, 16.0];
const SEGMENT_LOW: f32 = 0.3;
const SEGMENT_HIGH: f32 = 0.7;
const CROP_MARGIN: f32 = 0.6;
const CROP_MARGIN_BOTTOM: f32 = 1.2;
const WORK_SIZE: f32 = 768.0;
const OVAL_EXPANSION: f32 = 1.12;
const OVAL_SOFTNESS: f32 = 0.03;
const EDGE_FADE: f32 = 0.03;
const REFINE_DIVISOR: f32 = 60.0;
const REFINE_EPS: f32 = 0.001;
const SUPERSAMPLING: usize = 4;

const RIGHT_EYE_CORNER: usize = 33;
const LEFT_EYE_CORNER: usize = 263;
const LIPS_OUTER: [usize; 20] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
];
const LIPS_INNER: [usize; 20] = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
];
const LEFT_EYE: [usize; 16] = [
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
];
const RIGHT_EYE: [usize; 16] = [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
];
const LEFT_BROW: [usize; 10] = [276, 283, 282, 295, 285, 336, 296, 334, 293, 300];
const RIGHT_BROW: [usize; 10] = [46, 53, 52, 65, 55, 107, 66, 105, 63, 70];
const NOSE: [usize; 26] = [
    168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115,
    48, 64, 193, 417,
];
const FACE_OVAL: [usize; 36] = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
    148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
];
const LEFT_IRIS: [usize; 5] = [473, 474, 475, 476, 477];
const RIGHT_IRIS: [usize; 5] = [468, 469, 470, 471, 472];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FacePart {
    FaceSkin,
    BodySkin,
    Eyebrows,
    EyeWhites,
    Iris,
    Lips,
    Teeth,
    Nose,
    Hair,
}

impl FacePart {
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "faceSkin" => Some(FacePart::FaceSkin),
            "bodySkin" => Some(FacePart::BodySkin),
            "eyebrows" => Some(FacePart::Eyebrows),
            "eyeWhites" => Some(FacePart::EyeWhites),
            "iris" => Some(FacePart::Iris),
            "lips" => Some(FacePart::Lips),
            "teeth" => Some(FacePart::Teeth),
            "nose" => Some(FacePart::Nose),
            "hair" => Some(FacePart::Hair),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct FaceFrame {
    center: [f32; 2],
    side: f32,
    angle: f32,
}

fn rotate([x, y]: [f32; 2], angle: f32) -> [f32; 2] {
    let (sin, cos) = angle.sin_cos();
    [cos * x - sin * y, sin * x + cos * y]
}

impl FaceFrame {
    fn to_image(self, size: usize, x: f32, y: f32) -> [f32; 2] {
        let k = self.side / size as f32;
        let half = size as f32 / 2.0;
        let [dx, dy] = rotate([(x - half) * k, (y - half) * k], self.angle);
        [self.center[0] + dx, self.center[1] + dy]
    }

    fn to_frame(self, size: usize, p: [f32; 2]) -> [f32; 2] {
        let k = size as f32 / self.side;
        let half = size as f32 / 2.0;
        let [dx, dy] = rotate([p[0] - self.center[0], p[1] - self.center[1]], -self.angle);
        [dx * k + half, dy * k + half]
    }
}

pub struct DetectedFace {
    landmarks: Vec<[f32; 2]>,
    center: [f32; 2],
    angle: f32,
    size: f32,
    segment_frame: FaceFrame,
    hair: Vec<f32>,
    skin: Vec<f32>,
}

pub struct FaceAnalysis {
    pub path_hash: String,
    pub faces: Vec<DetectedFace>,
    body: Vec<(FaceFrame, Vec<f32>)>,
}

fn inside(x: f32, y: f32, width: usize, height: usize) -> bool {
    x >= -0.5 && y >= -0.5 && x <= width as f32 - 0.5 && y <= height as f32 - 0.5
}

fn bilinear(x: f32, y: f32, width: usize, height: usize) -> [(usize, f32); 4] {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(width - 1), (y0 + 1).min(height - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    [
        (y0 * width + x0, (1.0 - fx) * (1.0 - fy)),
        (y0 * width + x1, fx * (1.0 - fy)),
        (y1 * width + x0, (1.0 - fx) * fy),
        (y1 * width + x1, fx * fy),
    ]
}

fn sample_rgb(image: &RgbImage, x: f32, y: f32, replicate: bool) -> [f32; 3] {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if !replicate && !inside(x, y, w, h) {
        return [0.0; 3];
    }
    let taps = bilinear(x, y, w, h);
    let raw = image.as_raw();
    std::array::from_fn(|c| taps.iter().map(|&(i, k)| raw[i * 3 + c] as f32 * k).sum())
}

fn frame_tensor(
    image: &RgbImage,
    frame: &FaceFrame,
    size: usize,
    replicate: bool,
    normalize: impl Fn(f32) -> f32 + Sync,
) -> Array4<f32> {
    let mut tensor = Array4::<f32>::zeros((1, size, size, 3));
    tensor
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(size * 3)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..size {
                let [sx, sy] = frame.to_image(size, x as f32, y as f32);
                let rgb = sample_rgb(image, sx, sy, replicate);
                for c in 0..3 {
                    row[x * 3 + c] = normalize(rgb[c]);
                }
            }
        });
    tensor
}

fn run_landmarks(
    session: &Mutex<Session>,
    image: &RgbImage,
    frame: &FaceFrame,
) -> Result<(Vec<[f32; 2]>, f32)> {
    let tensor = frame_tensor(image, frame, LANDMARK_INPUT_SIZE, false, |v| v / 255.0);
    let mut session = session.lock().unwrap();
    let outputs = session.run(ort::inputs![Tensor::from_array(tensor)?])?;
    let points = outputs[0].try_extract_array::<f32>()?;
    let score = outputs[1]
        .try_extract_array::<f32>()?
        .iter()
        .next()
        .copied()
        .unwrap_or(f32::MIN);
    let points: Vec<f32> = points.iter().copied().collect();
    let landmarks = (0..LANDMARK_COUNT)
        .map(|i| frame.to_image(LANDMARK_INPUT_SIZE, points[i * 3], points[i * 3 + 1]))
        .collect();
    Ok((landmarks, score))
}

fn upright_extent(landmarks: &[[f32; 2]], angle: f32) -> ([f32; 2], f32) {
    let (min, max) = bounds(landmarks);
    let center = [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0];
    let upright: Vec<[f32; 2]> = landmarks
        .iter()
        .map(|p| rotate([p[0] - center[0], p[1] - center[1]], -angle))
        .collect();
    let (lo, hi) = bounds(&upright);
    (center, (hi[0] - lo[0]).max(hi[1] - lo[1]))
}

fn bounds(points: &[[f32; 2]]) -> ([f32; 2], [f32; 2]) {
    points
        .iter()
        .fold(([f32::MAX; 2], [f32::MIN; 2]), |(min, max), p| {
            (
                [min[0].min(p[0]), min[1].min(p[1])],
                [max[0].max(p[0]), max[1].max(p[1])],
            )
        })
}

fn fit_landmarks(
    session: &Mutex<Session>,
    image: &RgbImage,
    face_box: [f32; 4],
    angle: f32,
) -> Result<(Vec<[f32; 2]>, f32, f32)> {
    let mut frame = FaceFrame {
        center: [
            (face_box[0] + face_box[2]) / 2.0,
            (face_box[1] + face_box[3]) / 2.0,
        ],
        side: (face_box[2] - face_box[0]).max(face_box[3] - face_box[1]) * LANDMARK_MARGIN,
        angle,
    };
    let (landmarks, score) = run_landmarks(session, image, &frame)?;
    let mut best = (landmarks, score, angle);
    let mut current = best.0.clone();
    for _ in 1..LANDMARK_PASSES {
        let (right, left) = (current[RIGHT_EYE_CORNER], current[LEFT_EYE_CORNER]);
        let angle = (left[1] - right[1]).atan2(left[0] - right[0]);
        let (center, extent) = upright_extent(&current, angle);
        frame = FaceFrame {
            center,
            side: extent * LANDMARK_MARGIN,
            angle,
        };
        let (landmarks, score) = run_landmarks(session, image, &frame)?;
        current = landmarks;
        if score > best.1 {
            best = (current.clone(), score, angle);
        }
        if score > LANDMARK_CONVERGED_SCORE {
            break;
        }
    }
    Ok(best)
}

fn run_segmenter(
    session: &Mutex<Session>,
    image: &RgbImage,
    frame: &FaceFrame,
) -> Result<[Vec<f32>; 3]> {
    let tensor = frame_tensor(image, frame, SEGMENTER_INPUT_SIZE, true, |v| {
        v / 127.5 - 1.0
    });
    let mut session = session.lock().unwrap();
    let outputs = session.run(ort::inputs![Tensor::from_array(tensor)?])?;
    let logits = outputs[0].try_extract_array::<f32>()?;
    let logits = logits
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("Unexpected face segmentation output layout"))?;
    let mut probabilities: [Vec<f32>; 3] = Default::default();
    for classes in logits.as_chunks::<SEGMENTER_CLASSES>().0 {
        let max = classes.iter().fold(f32::MIN, |m, &v| m.max(v));
        let sum: f32 = classes.iter().map(|&v| (v - max).exp()).sum();
        for (out, &class) in probabilities.iter_mut().zip(&SEGMENTER_OUTPUTS) {
            out.push((classes[class] - max).exp() / sum);
        }
    }
    Ok(probabilities)
}

fn box_overlap(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let smaller = ((a[2] - a[0]) * (a[3] - a[1])).min((b[2] - b[0]) * (b[3] - b[1]));
    w * h / smaller.max(f32::EPSILON)
}

pub fn analyze_faces(
    image: &DynamicImage,
    detections: &[(f32, [f32; 4])],
    landmark_session: &Mutex<Session>,
    segmenter_session: &Mutex<Session>,
    base_angle: f32,
    path_hash: String,
) -> Result<FaceAnalysis> {
    let (width, height) = image.dimensions();
    let mut boxes: Vec<[f32; 4]> = Vec::new();
    for (_, b) in detections {
        let b = [
            b[0] * width as f32,
            b[1] * height as f32,
            b[2] * width as f32,
            b[3] * height as f32,
        ];
        if !boxes
            .iter()
            .any(|kept| box_overlap(kept, &b) > DUPLICATE_BOX_OVERLAP)
        {
            boxes.push(b);
        }
    }
    let mut analysis = FaceAnalysis {
        path_hash,
        faces: Vec::new(),
        body: Vec::new(),
    };
    if boxes.is_empty() {
        return Ok(analysis);
    }

    let rgb = image.to_rgb8();
    let faces = &mut analysis.faces;
    for face_box in boxes {
        let mut fitted = None;
        for quarter in 0..4 {
            let angle = base_angle + quarter as f32 * FRAC_PI_2;
            let (landmarks, score, angle) = fit_landmarks(landmark_session, &rgb, face_box, angle)?;
            if score >= FACE_PRESENCE_THRESHOLD {
                fitted = Some((landmarks, angle));
                break;
            }
        }
        let Some((landmarks, angle)) = fitted else {
            continue;
        };

        let (center, size) = upright_extent(&landmarks, angle);
        if faces.iter().any(|face| {
            (face.center[0] - center[0]).hypot(face.center[1] - center[1])
                < DUPLICATE_FACE_DISTANCE * size.max(face.size)
        }) {
            continue;
        }

        let [dx, dy] = rotate([0.0, SEGMENTER_OFFSET * size], angle);
        let segment_frame = FaceFrame {
            center: [center[0] + dx, center[1] + dy],
            side: size * SEGMENTER_SPAN,
            angle,
        };
        let [hair, skin, _] = run_segmenter(segmenter_session, &rgb, &segment_frame)?;
        let [dx, dy] = rotate([0.0, BODY_OFFSET * size], angle);
        let body_frame = FaceFrame {
            center: [center[0] + dx, center[1] + dy],
            side: size * BODY_SPAN,
            angle,
        };
        let [_, _, body] = run_segmenter(segmenter_session, &rgb, &body_frame)?;
        analysis.body.push((body_frame, body));
        faces.push(DetectedFace {
            landmarks,
            center,
            angle,
            size,
            segment_frame,
            hair,
            skin,
        });
    }
    if !analysis.faces.is_empty() {
        let frame = FaceFrame {
            center: [width as f32 / 2.0, height as f32 / 2.0],
            side: width.max(height) as f32,
            angle: base_angle,
        };
        let [_, _, body] = run_segmenter(segmenter_session, &rgb, &frame)?;
        analysis.body.push((frame, body));
    }
    Ok(analysis)
}

fn rasterize(
    width: usize,
    height: usize,
    (top, bottom): (f32, f32),
    spans: impl Fn(f32) -> Vec<[f32; 2]> + Sync,
) -> Vec<f32> {
    let mut mask = vec![0.0; width * height];
    let y0 = (top.floor().max(0.0) as usize).min(height);
    let y1 = (bottom.ceil().max(0.0) as usize + 1).min(height);
    let samples = (width * SUPERSAMPLING) as f32;
    let scale = SUPERSAMPLING as f32;
    mask.par_chunks_mut(width)
        .enumerate()
        .skip(y0)
        .take(y1.saturating_sub(y0))
        .for_each(|(y, row)| {
            for sy in 0..SUPERSAMPLING {
                for [a, b] in spans(y as f32 + (sy as f32 + 0.5) / scale - 0.5) {
                    let start = ((a + 0.5) * scale - 0.5).ceil().clamp(0.0, samples) as usize;
                    let end = ((b + 0.5) * scale - 0.5).ceil().clamp(0.0, samples) as usize;
                    for n in start..end {
                        row[n / SUPERSAMPLING] += 1.0 / (scale * scale);
                    }
                }
            }
        });
    mask
}

fn polygon_mask(width: usize, height: usize, polygon: &[[f32; 2]]) -> Vec<f32> {
    let (min, max) = bounds(polygon);
    let n = polygon.len();
    rasterize(width, height, (min[1], max[1]), |y| {
        let mut crossings: Vec<f32> = (0..n)
            .map(|i| (polygon[i], polygon[(i + n - 1) % n]))
            .filter(|(a, b)| (a[1] > y) != (b[1] > y))
            .map(|(a, b)| a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
            .collect();
        crossings.sort_by(f32::total_cmp);
        crossings.as_chunks::<2>().0.to_vec()
    })
}

fn convex_hull(points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut points = points.to_vec();
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut hull: Vec<[f32; 2]> = Vec::with_capacity(points.len() * 2);
    for pass in 0..2 {
        let start = hull.len();
        let ordered: Box<dyn Iterator<Item = &[f32; 2]>> = if pass == 0 {
            Box::new(points.iter())
        } else {
            Box::new(points.iter().rev())
        };
        for &p in ordered {
            while hull.len() >= start + 2
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

fn scaled_polygon(points: &[[f32; 2]], scale: f32) -> Vec<[f32; 2]> {
    let n = points.len() as f32;
    let cx = points.iter().map(|p| p[0]).sum::<f32>() / n;
    let cy = points.iter().map(|p| p[1]).sum::<f32>() / n;
    points
        .iter()
        .map(|p| [cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale])
        .collect()
}

fn disc_mask(width: usize, height: usize, landmarks: &[[f32; 2]], iris: &[usize]) -> Vec<f32> {
    let c = landmarks[iris[0]];
    let r = iris[1..]
        .iter()
        .map(|&i| (landmarks[i][0] - c[0]).hypot(landmarks[i][1] - c[1]))
        .sum::<f32>()
        / (iris.len() - 1) as f32;
    rasterize(width, height, (c[1] - r, c[1] + r), |y| {
        let half = (r * r - (y - c[1]) * (y - c[1])).max(0.0).sqrt();
        vec![[c[0] - half, c[0] + half]]
    })
}

fn to_gray(values: &[f32], width: usize, height: usize) -> GrayImage {
    GrayImage::from_raw(
        width as u32,
        height as u32,
        values
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect(),
    )
    .unwrap()
}

fn from_gray(image: &GrayImage) -> Vec<f32> {
    image.as_raw().iter().map(|&v| v as f32 / 255.0).collect()
}

fn refine(guide: &RgbImage, mask: &[f32], radius: usize) -> Vec<f32> {
    let (width, height) = guide.dimensions();
    from_gray(&fast_color_guided_filter(
        guide,
        &to_gray(mask, width as usize, height as usize),
        radius.max(1),
        REFINE_EPS,
    ))
}

fn sharpen(probabilities: &[f32]) -> Vec<f32> {
    probabilities
        .iter()
        .map(|&p| (p - SEGMENT_LOW) / (SEGMENT_HIGH - SEGMENT_LOW))
        .collect()
}

fn maximum(a: &mut [f32], b: &[f32]) {
    a.iter_mut().zip(b).for_each(|(a, &b)| *a = a.max(b));
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, f32::total_cmp);
    Some(values[mid])
}

fn teeth_mask(rgb: &RgbImage, luma: &[f32], lips: &[f32], mouth: &[f32]) -> Vec<f32> {
    let saturation: Vec<f32> = rgb
        .pixels()
        .map(|p| {
            let max = p.0.iter().copied().max().unwrap_or(0) as f32;
            let min = p.0.iter().copied().min().unwrap_or(0) as f32;
            if max > 0.0 { (max - min) / max } else { 0.0 }
        })
        .collect();
    let lip_pixels = || {
        lips.iter()
            .enumerate()
            .filter(|&(_, &v)| v > 0.5)
            .map(|(i, _)| i)
    };
    let (Some(lip_luma), Some(lip_saturation)) = (
        median(lip_pixels().map(|i| luma[i]).collect()),
        median(lip_pixels().map(|i| saturation[i]).collect()),
    ) else {
        return vec![0.0; mouth.len()];
    };
    mouth
        .iter()
        .zip(luma)
        .zip(&saturation)
        .map(|((&m, &l), &s)| {
            let bright = ((l - lip_luma - 0.04) / 0.12).clamp(0.0, 1.0);
            let pale = ((lip_saturation * 0.9 + 0.05 - s) / 0.15).clamp(0.0, 1.0);
            m * bright * pale
        })
        .collect()
}

fn face_region(face: &DetectedFace, width: u32, height: u32) -> (u32, u32, u32, u32) {
    let near = (0.5 + CROP_MARGIN) * face.size;
    let far = (0.5 + CROP_MARGIN_BOTTOM) * face.size;
    let corners: Vec<[f32; 2]> = [[-near, -near], [near, -near], [-near, far], [near, far]]
        .into_iter()
        .map(|corner| {
            let [dx, dy] = rotate(corner, face.angle);
            [face.center[0] + dx, face.center[1] + dy]
        })
        .collect();
    let (min, max) = bounds(&corners);
    (
        min[0].floor().clamp(0.0, width as f32) as u32,
        min[1].floor().clamp(0.0, height as f32) as u32,
        max[0].ceil().clamp(0.0, width as f32) as u32,
        max[1].ceil().clamp(0.0, height as f32) as u32,
    )
}

fn face_parts_mask(
    image: &DynamicImage,
    face: &DetectedFace,
    parts: &[FacePart],
    region: (u32, u32, u32, u32),
) -> GrayImage {
    let (x0, y0, x1, y1) = region;
    let (crop_w, crop_h) = (x1 - x0, y1 - y0);
    let scale = (WORK_SIZE / crop_w.max(crop_h) as f32).min(1.0);
    let w = ((crop_w as f32 * scale).round() as usize).max(1);
    let h = ((crop_h as f32 * scale).round() as usize).max(1);

    let crop = image.crop_imm(x0, y0, crop_w, crop_h);
    let guide_full = crop.to_rgb8();
    let work = crop.resize_exact(w as u32, h as u32, imageops::FilterType::Triangle);
    let work_rgb = work.to_rgb8();
    let luma = from_gray(&work.to_luma8());

    let points: Vec<[f32; 2]> = face
        .landmarks
        .iter()
        .map(|p| [(p[0] - x0 as f32) * scale, (p[1] - y0 as f32) * scale])
        .collect();
    let select = |indices: &[usize]| indices.iter().map(|&i| points[i]).collect::<Vec<_>>();
    let size = face.size * scale;
    let radius = (size / REFINE_DIVISOR).round().max(1.0) as usize;
    let fine_radius = (radius / 2).max(1);

    let pair = |mut left: Vec<f32>, right: &[f32]| {
        maximum(&mut left, right);
        left
    };
    let polygons = |a: &[usize], b: &[usize]| {
        pair(
            polygon_mask(w, h, &select(a)),
            &polygon_mask(w, h, &select(b)),
        )
    };
    let eyes = polygons(&LEFT_EYE, &RIGHT_EYE);
    let brows = polygons(&LEFT_BROW, &RIGHT_BROW);
    let mut iris = pair(
        disc_mask(w, h, &points, &LEFT_IRIS),
        &disc_mask(w, h, &points, &RIGHT_IRIS),
    );
    iris.iter_mut().zip(&eyes).for_each(|(i, &e)| *i *= e);
    let lips_outer = polygon_mask(w, h, &select(&LIPS_OUTER));
    let mouth = polygon_mask(w, h, &select(&LIPS_INNER));
    let lips: Vec<f32> = lips_outer
        .iter()
        .zip(&mouth)
        .map(|(&o, &i)| (o - i).max(0.0))
        .collect();

    let mut combined = vec![0.0; w * h];
    for &part in parts {
        let mask = match part {
            FacePart::EyeWhites => {
                let refined = refine(&work_rgb, &eyes, fine_radius);
                let iris = refine(&work_rgb, &iris, fine_radius);
                refined
                    .iter()
                    .zip(&iris)
                    .map(|(&e, &i)| (e - i).max(0.0))
                    .collect()
            }
            FacePart::Iris => refine(&work_rgb, &iris, fine_radius),
            FacePart::Eyebrows => refine(&work_rgb, &brows, radius),
            FacePart::Lips => refine(&work_rgb, &lips, radius),
            FacePart::Teeth => {
                let inner = polygon_mask(w, h, &scaled_polygon(&select(&LIPS_INNER), 0.95));
                let teeth = teeth_mask(&work_rgb, &luma, &lips, &inner);
                refine(&work_rgb, &teeth, fine_radius)
                    .iter()
                    .zip(&inner)
                    .map(|(&t, &m)| t * m)
                    .collect()
            }
            FacePart::Nose => refine(
                &work_rgb,
                &polygon_mask(w, h, &convex_hull(&select(&NOSE))),
                radius * 2,
            ),
            FacePart::BodySkin => continue,
            FacePart::Hair => refine(
                &work_rgb,
                &sharpen(&segment_probabilities(
                    &face.segment_frame,
                    &face.hair,
                    w,
                    h,
                    x0,
                    y0,
                    scale,
                )),
                radius * 2,
            ),
            FacePart::FaceSkin => {
                let oval = scaled_polygon(&select(&FACE_OVAL), OVAL_EXPANSION);
                let oval = ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(
                    w as u32,
                    h as u32,
                    polygon_mask(w, h, &oval),
                )
                .unwrap();
                let oval = imageops::blur(&oval, (size * OVAL_SOFTNESS).max(0.5)).into_raw();
                let skin: Vec<f32> =
                    segment_probabilities(&face.segment_frame, &face.skin, w, h, x0, y0, scale)
                        .iter()
                        .zip(&oval)
                        .map(|(&s, &o)| s * o)
                        .collect();
                let features = refine(
                    &work_rgb,
                    &pair(pair(brows.clone(), &eyes), &lips_outer),
                    fine_radius,
                );
                refine(&work_rgb, &sharpen(&skin), radius * 2)
                    .iter()
                    .zip(&features)
                    .map(|(&s, &f)| s * (1.0 - f))
                    .collect()
            }
        };
        maximum(&mut combined, &mask);
    }

    let fade = EDGE_FADE * size;
    let fades = [x0 > 0, y0 > 0, x1 < image.width(), y1 < image.height()];
    combined.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, v) in row.iter_mut().enumerate() {
            let distances = [x as f32, y as f32, (w - 1 - x) as f32, (h - 1 - y) as f32];
            let edge = distances
                .iter()
                .zip(fades)
                .filter(|&(_, faded)| faded)
                .fold(1.0f32, |acc, (&d, _)| acc.min(d / fade.max(1.0)));
            *v *= edge.min(1.0);
        }
    });

    let combined = to_gray(&combined, w, h);
    if (w as u32, h as u32) == (crop_w, crop_h) {
        combined
    } else {
        fast_color_guided_filter(&guide_full, &combined, fine_radius, REFINE_EPS)
    }
}

fn segment_probabilities(
    frame: &FaceFrame,
    grid: &[f32],
    width: usize,
    height: usize,
    x0: u32,
    y0: u32,
    scale: f32,
) -> Vec<f32> {
    let mut values = vec![0.0; width * height];
    values
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let p = [x0 as f32 + x as f32 / scale, y0 as f32 + y as f32 / scale];
                let [u, v] = frame.to_frame(SEGMENTER_INPUT_SIZE, p);
                if inside(u, v, SEGMENTER_INPUT_SIZE, SEGMENTER_INPUT_SIZE) {
                    *out = bilinear(u, v, SEGMENTER_INPUT_SIZE, SEGMENTER_INPUT_SIZE)
                        .iter()
                        .map(|&(i, k)| grid[i] * k)
                        .sum();
                }
            }
        });
    values
}

/// What a face does around its skin, for the skin tools: where it tends to fold, and the band under
/// each eye that tends to go dark. Both are soft, so they are kept small and read at any size.
pub struct SkinRegions {
    /// how much a valley counts as a line at each place, 0 to 1: most where a face folds anyway - beside the
    /// nose, under the eyes, across the forehead - something everywhere else on the skin, and little where
    /// the shadows are the shape of the face and not wrinkles, under the nose
    pub lines: GrayImage,
    /// the shadowed band under each eye
    pub under_eye: GrayImage,
}

const REGION_MAP_SIDE: f32 = 1024.0;
const RIGHT_EYE_INNER_CORNER: usize = 133;
const LEFT_EYE_INNER_CORNER: usize = 362;
const MOUTH_CORNERS: [usize; 2] = [61, 291];
const NOSE_TIP: usize = 1;
const NOSE_BASE: usize = 2;
const NOSTRIL_CORNERS: [usize; 2] = [98, 327];
// what a valley counts for on skin nothing says anything about, and under the nose
const PLAIN_LINE_WEIGHT: f32 = 0.45;
const NOSE_SHADOW_LINE_WEIGHT: f32 = 0.12;

fn mean_point(points: &[[f32; 2]], indices: impl Iterator<Item = usize>) -> [f32; 2] {
    let (sum, count) = indices.fold(([0.0f32; 2], 0.0f32), |(sum, count), i| {
        ([sum[0] + points[i][0], sum[1] + points[i][1]], count + 1.0)
    });
    [sum[0] / count.max(1.0), sum[1] / count.max(1.0)]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn smooth_step(low: f32, high: f32, x: f32) -> f32 {
    let t = ((x - low) / (high - low).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn segment_distance(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let t =
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy).max(1e-6)).clamp(0.0, 1.0);
    distance(p, [a[0] + dx * t, a[1] + dy * t])
}

/// The regions are laid out from the fitted landmarks in the face's own upright frame, in proportion
/// to the face: the width between the eyes sets the size of everything else, so they land the same way
/// on a small face and a large one.
pub fn compose_skin_regions(width: u32, height: u32, analysis: &FaceAnalysis) -> SkinRegions {
    let scale = (REGION_MAP_SIDE / width.max(height) as f32).min(1.0);
    let w = ((width as f32 * scale).round() as usize).max(1);
    let h = ((height as f32 * scale).round() as usize).max(1);
    let mut lines = vec![PLAIN_LINE_WEIGHT; w * h];
    let mut counted = vec![false; w * h];
    let mut under_eye = vec![0.0f32; w * h];

    for face in &analysis.faces {
        let origin = [face.center[0] * scale, face.center[1] * scale];
        let upright: Vec<[f32; 2]> = face
            .landmarks
            .iter()
            .map(|p| {
                rotate(
                    [p[0] * scale - origin[0], p[1] * scale - origin[1]],
                    -face.angle,
                )
            })
            .collect();

        let right_eye = mean_point(&upright, RIGHT_EYE.iter().copied());
        let left_eye = mean_point(&upright, LEFT_EYE.iter().copied());
        let eye_span = distance(right_eye, left_eye).max(1.0);
        let size = eye_span * 2.6;
        let eye_radius = ((distance(upright[RIGHT_EYE_CORNER], upright[RIGHT_EYE_INNER_CORNER])
            + distance(upright[LEFT_EYE_CORNER], upright[LEFT_EYE_INNER_CORNER]))
            / 4.0)
            .max(1.0);
        let brow_y = mean_point(&upright, LEFT_BROW.iter().chain(RIGHT_BROW.iter()).copied())[1];
        let mouth = mean_point(&upright, MOUTH_CORNERS.iter().copied());
        let mouth_half = distance(upright[MOUTH_CORNERS[0]], upright[MOUTH_CORNERS[1]]) / 2.0;
        let nose = upright[NOSE_TIP];
        // The shadow under the nose and around the nostrils is the shape of the nose, not a wrinkle, and a
        // line healed out of it floats on the skin as a pale ribbon.
        let base = upright[NOSE_BASE];
        let nostril_half = distance(upright[NOSTRIL_CORNERS[0]], upright[NOSTRIL_CORNERS[1]]) / 2.0;
        let nose_shadow = [base[0], base[1] + nostril_half * 0.15];

        let blob = |u: [f32; 2], centre: [f32; 2], rx: f32, ry: f32| {
            let (dx, dy) = ((u[0] - centre[0]) / rx, (u[1] - centre[1]) / ry);
            (-(dx * dx + dy * dy)).exp()
        };
        // a soft band from beside the nose down to the corner of the mouth, one on each side
        let fold = |u: [f32; 2], side: f32| {
            let from = [nose[0] + side * size * 0.09, nose[1] + size * 0.03];
            let to = [mouth[0] + side * mouth_half, mouth[1] + mouth_half * 0.4];
            let d = segment_distance(u, from, to) / (size * 0.05);
            (-(d * d)).exp()
        };

        let (x0, y0, x1, y1) = face_region(face, width, height);
        let (x0, y0) = ((x0 as f32 * scale) as usize, (y0 as f32 * scale) as usize);
        let (x1, y1) = (
            ((x1 as f32 * scale).ceil() as usize).min(w),
            ((y1 as f32 * scale).ceil() as usize).min(h),
        );
        for y in y0..y1 {
            for x in x0..x1 {
                let u = rotate([x as f32 - origin[0], y as f32 - origin[1]], -face.angle);
                let forehead = 1.0 - smooth_step(brow_y - size * 0.06, brow_y, u[1]);
                let under = blob(
                    u,
                    [right_eye[0], right_eye[1] + eye_radius * 1.35],
                    eye_radius * 1.5,
                    eye_radius * 0.85,
                )
                .max(blob(
                    u,
                    [left_eye[0], left_eye[1] + eye_radius * 1.35],
                    eye_radius * 1.5,
                    eye_radius * 0.85,
                ));
                let folds = fold(u, 1.0).max(fold(u, -1.0));
                let fold_share = (folds + under * 0.8 + forehead * 0.6).clamp(0.0, 1.0);
                let weight = PLAIN_LINE_WEIGHT + (1.0 - PLAIN_LINE_WEIGHT) * fold_share;
                let guard = blob(u, nose_shadow, nostril_half * 1.4, nostril_half * 0.5);
                let weight = weight + (NOSE_SHADOW_LINE_WEIGHT - weight) * guard;
                let i = y * w + x;
                lines[i] = if counted[i] {
                    lines[i].max(weight)
                } else {
                    weight
                };
                counted[i] = true;
                under_eye[i] = under_eye[i].max(under);
            }
        }
    }

    SkinRegions {
        lines: to_gray(&lines, w, h),
        under_eye: to_gray(&under_eye, w, h),
    }
}

const WRINKLE_INPUT_SIZE: usize = 512;
const WRINKLE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const WRINKLE_STD: [f32; 3] = [0.229, 0.224, 0.225];
// in the model's own pixels: the lines come back hard-edged, and a correction that follows a hard edge
// leaves one
const WRINKLE_SOFTNESS: f32 = 1.2;
// below this the picture is shrunk properly before it is sampled, so a face many times the size of the
// model's input is averaged rather than picked at
const PRESHRINK: f32 = 0.75;
// ABPN's blemish detector (Alibaba's skin retouching): a square 1.5 times the face around it, scaled to
// -1..1, in; how likely each pixel is to be part of a spot, a mole, a milium or an acne mark, out
const BLEMISH_INPUT_SIZE: usize = 768;
const BLEMISH_SPAN: f32 = 1.5;
const BLEMISH_SOFTNESS: f32 = 1.0;

// The square the FFHQ faces were cut to, which is what the wrinkle model learned on: built from the eyes
// and the mouth exactly as that dataset's alignment builds it, so a face here is framed, sized and turned
// the way every face it was trained on was.
fn ffhq_frame(face: &DetectedFace) -> FaceFrame {
    let points = &face.landmarks;
    let eye_a = mean_point(points, RIGHT_EYE.iter().copied());
    let eye_b = mean_point(points, LEFT_EYE.iter().copied());
    let mouth = mean_point(points, MOUTH_CORNERS.iter().copied());
    let eye_avg = [(eye_a[0] + eye_b[0]) / 2.0, (eye_a[1] + eye_b[1]) / 2.0];
    let eye_to_eye = [eye_b[0] - eye_a[0], eye_b[1] - eye_a[1]];
    let eye_to_mouth = [mouth[0] - eye_avg[0], mouth[1] - eye_avg[1]];
    let across = [
        eye_to_eye[0] + eye_to_mouth[1],
        eye_to_eye[1] - eye_to_mouth[0],
    ];
    let half = (eye_to_eye[0].hypot(eye_to_eye[1]) * 2.0)
        .max(eye_to_mouth[0].hypot(eye_to_mouth[1]) * 1.8);
    FaceFrame {
        center: [
            eye_avg[0] + eye_to_mouth[0] * 0.1,
            eye_avg[1] + eye_to_mouth[1] * 0.1,
        ],
        side: (half * 2.0).max(1.0),
        angle: across[1].atan2(across[0]),
    }
}

// The square of the blemish detector: the face's own frame, upright, half as wide again as the face, which
// is how the pipeline it comes from cuts a face out around its box.
fn blemish_frame(face: &DetectedFace) -> FaceFrame {
    FaceFrame {
        center: face.center,
        side: (face.size * BLEMISH_SPAN).max(1.0),
        angle: face.angle,
    }
}

// The face cut out along its frame at `size` pixels a side, channels first, each pixel's 0-255 RGB put
// through `normalize`. A face much bigger than the model's input is shrunk with a proper filter first
// and then sampled; sampled straight from the full picture, the fine lines and small spots these models
// look for would alias into noise.
fn face_tensor(
    image: &RgbImage,
    frame: &FaceFrame,
    size: usize,
    normalize: impl Fn([f32; 3]) -> [f32; 3] + Sync,
) -> Array4<f32> {
    let factor = size as f32 / frame.side;
    let shrunk = if factor < PRESHRINK {
        let corners: Vec<[f32; 2]> = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            .into_iter()
            .map(|[u, v]| frame.to_image(size, u * size as f32, v * size as f32))
            .collect();
        let (min, max) = bounds(&corners);
        let (w, h) = (image.width() as f32, image.height() as f32);
        let x0 = min[0].floor().clamp(0.0, w) as u32;
        let y0 = min[1].floor().clamp(0.0, h) as u32;
        let x1 = max[0].ceil().clamp(0.0, w) as u32;
        let y1 = max[1].ceil().clamp(0.0, h) as u32;
        (x1 > x0 + 1 && y1 > y0 + 1).then(|| {
            let (cw, ch) = (x1 - x0, y1 - y0);
            let sw = ((cw as f32 * factor).round() as u32).max(1);
            let sh = ((ch as f32 * factor).round() as u32).max(1);
            let crop = imageops::crop_imm(image, x0, y0, cw, ch).to_image();
            let small = imageops::resize(&crop, sw, sh, imageops::FilterType::Triangle);
            let (fx, fy) = (sw as f32 / cw as f32, sh as f32 / ch as f32);
            let small_frame = FaceFrame {
                center: [
                    (frame.center[0] - x0 as f32 + 0.5) * fx - 0.5,
                    (frame.center[1] - y0 as f32 + 0.5) * fy - 0.5,
                ],
                side: frame.side * (fx + fy) / 2.0,
                angle: frame.angle,
            };
            (small, small_frame)
        })
    } else {
        None
    };
    let (source, frame) = match &shrunk {
        Some((small, small_frame)) => (small, small_frame),
        None => (image, frame),
    };

    let plane = size * size;
    let mut tensor = Array4::<f32>::zeros((1, 3, size, size));
    let data = tensor.as_slice_mut().unwrap();
    let (red, rest) = data.split_at_mut(plane);
    let (green, blue) = rest.split_at_mut(plane);
    red.par_chunks_mut(size)
        .zip(green.par_chunks_mut(size))
        .zip(blue.par_chunks_mut(size))
        .enumerate()
        .for_each(|(y, ((r, g), b))| {
            let row = r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut());
            for (x, ((r, g), b)) in row.enumerate() {
                let [sx, sy] = frame.to_image(size, x as f32, y as f32);
                [*r, *g, *b] = normalize(sample_rgb(source, sx, sy, true));
            }
        });
    tensor
}

// One face through a model that answers with one probability a pixel at its input's size, softened by
// `softness` of the model's own pixels: the answers come back hard-edged, and a correction that follows a
// hard edge leaves one.
fn run_face_model(
    session: &Mutex<Session>,
    tensor: Array4<f32>,
    size: usize,
    softness: f32,
) -> Result<Vec<f32>> {
    let probabilities: Vec<f32> = {
        let mut session = session.lock().unwrap();
        let outputs = session.run(ort::inputs![Tensor::from_array(tensor)?])?;
        outputs[0]
            .try_extract_array::<f32>()?
            .iter()
            .copied()
            .collect()
    };
    if probabilities.len() != size * size {
        return Err(anyhow::anyhow!(
            "Unexpected face model output of {} values",
            probabilities.len()
        ));
    }
    let grid =
        ImageBuffer::<Luma<f32>, Vec<f32>>::from_raw(size as u32, size as u32, probabilities)
            .unwrap();
    Ok(imageops::blur(&grid, softness).into_raw())
}

fn run_wrinkles(session: &Mutex<Session>, image: &RgbImage, frame: &FaceFrame) -> Result<Vec<f32>> {
    let tensor = face_tensor(image, frame, WRINKLE_INPUT_SIZE, |rgb| {
        std::array::from_fn(|c| (rgb[c] / 255.0 - WRINKLE_MEAN[c]) / WRINKLE_STD[c])
    });
    run_face_model(session, tensor, WRINKLE_INPUT_SIZE, WRINKLE_SOFTNESS)
}

fn run_blemishes(session: &Mutex<Session>, image: &RgbImage, frame: &FaceFrame) -> Result<Vec<f32>> {
    let tensor = face_tensor(image, frame, BLEMISH_INPUT_SIZE, |rgb| {
        rgb.map(|v| v / 127.5 - 1.0)
    });
    run_face_model(session, tensor, BLEMISH_INPUT_SIZE, BLEMISH_SOFTNESS)
}

// One map over the whole frame from a model run on each face in turn: `find` gives the frame it cut the
// face out along, the size of the model's square and its answer there, and every pixel of the face's
// region on the map takes the largest answer any face gave it. The map is at most `long_side` pixels
// across, which is all the skin tool looks for faults at.
fn compose_face_map(
    image: &DynamicImage,
    analysis: &FaceAnalysis,
    long_side: u32,
    find: impl Fn(&RgbImage, &DetectedFace) -> Result<(FaceFrame, usize, Vec<f32>)>,
) -> Result<GrayImage> {
    let (width, height) = image.dimensions();
    let scale = (long_side as f32 / width.max(height) as f32).min(1.0);
    let w = ((width as f32 * scale).round() as usize).max(1);
    let h = ((height as f32 * scale).round() as usize).max(1);
    let rgb = image.to_rgb8();
    let mut map = vec![0.0f32; w * h];
    for face in &analysis.faces {
        let (frame, size, probabilities) = find(&rgb, face)?;
        // the face's region, in pixels of the map
        let (x0, y0, x1, y1) = face_region(face, width, height);
        let to_map = |v: u32, limit: usize| ((v as f32 * scale) as usize).min(limit);
        let (mx0, my0) = (to_map(x0, w), to_map(y0, h));
        let (mx1, my1) = (
            ((x1 as f32 * scale).ceil() as usize).min(w),
            ((y1 as f32 * scale).ceil() as usize).min(h),
        );
        map.par_chunks_mut(w)
            .enumerate()
            .skip(my0)
            .take(my1.saturating_sub(my0))
            .for_each(|(y, row)| {
                for (x, out) in row.iter_mut().enumerate().take(mx1).skip(mx0) {
                    // where this pixel of the map lies on the picture
                    let p = [
                        (x as f32 + 0.5) / scale - 0.5,
                        (y as f32 + 0.5) / scale - 0.5,
                    ];
                    let [u, v] = frame.to_frame(size, p);
                    if inside(u, v, size, size) {
                        let found: f32 = bilinear(u, v, size, size)
                            .iter()
                            .map(|&(i, k)| probabilities[i] * k)
                            .sum();
                        *out = out.max(found);
                    }
                }
            });
    }
    Ok(to_gray(&map, w, h))
}

/// Where the lines are on every face in the picture, as one map over the whole frame: how likely each
/// place is to be a wrinkle, 0 to 1, found by the FFHQ-Wrinkle model on each face in turn. It is not cut
/// to the skin here; the skin tool reads it through its own face-skin weight, which already leaves out
/// the eyes, brows and lips the model sometimes takes for lines.
pub fn compose_wrinkle_map(
    image: &DynamicImage,
    analysis: &FaceAnalysis,
    session: &Mutex<Session>,
    long_side: u32,
) -> Result<GrayImage> {
    compose_face_map(image, analysis, long_side, |rgb, face| {
        let frame = ffhq_frame(face);
        let found = run_wrinkles(session, rgb, &frame)?;
        Ok((frame, WRINKLE_INPUT_SIZE, found))
    })
}

/// Where the blemishes are on every face in the picture - spots, moles, milia, acne marks - as one map
/// over the whole frame, how likely each place is to be one, 0 to 1, found by ABPN's blemish detector on
/// each face in turn. Like the lines, it is read through the skin tool's own face-skin weight.
pub fn compose_blemish_map(
    image: &DynamicImage,
    analysis: &FaceAnalysis,
    session: &Mutex<Session>,
    long_side: u32,
) -> Result<GrayImage> {
    compose_face_map(image, analysis, long_side, |rgb, face| {
        let frame = blemish_frame(face);
        let found = run_blemishes(session, rgb, &frame)?;
        Ok((frame, BLEMISH_INPUT_SIZE, found))
    })
}

pub fn compose_face_mask(
    image: &DynamicImage,
    analysis: &FaceAnalysis,
    parts: &[FacePart],
) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut mask = GrayImage::new(width, height);
    if parts.contains(&FacePart::BodySkin) {
        let scale = (BODY_WORK_SIZE / width.max(height) as f32).min(1.0);
        let w = ((width as f32 * scale).round() as usize).max(1);
        let h = ((height as f32 * scale).round() as usize).max(1);
        let mut body = vec![0.0; w * h];
        for (frame, grid) in &analysis.body {
            maximum(
                &mut body,
                &segment_probabilities(frame, grid, w, h, 0, 0, scale),
            );
        }
        let smallest_face = analysis
            .faces
            .iter()
            .map(|face| face.size)
            .fold(f32::MAX, f32::min);
        let radius = (smallest_face * scale / BODY_REFINE_DIVISOR)
            .clamp(BODY_REFINE_RADIUS[0], BODY_REFINE_RADIUS[1]);
        mask = fast_color_guided_filter(
            &image.to_rgb8(),
            &to_gray(&sharpen(&body), w, h),
            radius.round() as usize,
            REFINE_EPS,
        );
    }
    if parts.iter().all(|&part| part == FacePart::BodySkin) {
        return mask;
    }
    let regions: Vec<_> = analysis
        .faces
        .iter()
        .map(|face| face_region(face, width, height))
        .collect();
    let face_masks: Vec<GrayImage> = analysis
        .faces
        .par_iter()
        .zip(&regions)
        .map(|(face, &region)| face_parts_mask(image, face, parts, region))
        .collect();
    for (face_mask, &(x0, y0, _, _)) in face_masks.iter().zip(&regions) {
        for (x, y, pixel) in face_mask.enumerate_pixels() {
            let out = mask.get_pixel_mut(x0 + x, y0 + y);
            out[0] = out[0].max(pixel[0]);
        }
    }
    mask
}
